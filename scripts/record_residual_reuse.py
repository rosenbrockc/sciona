#!/usr/bin/env python3
"""Record exact, dimension-preserving residual reuse without claiming physical equivalence."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
from uuid import UUID,uuid5
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sciona.ghost.symbolic import deserialize_expr
from sciona.physics_ingest.pdg_evidence import prepare_pdg_evidence,_digest
from sciona.physics_ingest.residual_equivalence import equivalent_residual_mapping


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbol-file',required=True,type=Path)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();symbols=args.symbol_file.read_bytes();counts=Counter()
    root=Path(__file__).resolve().parents[1]
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        rows=db.execute("SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload,v.content_hash FROM artifact_symbolic_expressions e JOIN physics_equation_candidates q USING(candidate_id) JOIN physics_ingest_snapshots s USING(snapshot_id) JOIN artifact_versions v ON v.version_id=e.version_id WHERE e.evidence_json->'dimensional_analysis'->>'status'='passed' ORDER BY e.expression_id FOR SHARE OF e,q,s,v").fetchall()
        variables=db.execute('SELECT expression_id,symbol_name,dim_signature FROM artifact_symbolic_variables FOR SHARE').fetchall()
        dimensions=defaultdict(dict)
        for variable in variables:dimensions[variable['expression_id']][variable['symbol_name']]=variable['dim_signature']
        fresh={};expressions={};targets=[]
        for row in rows:
            key=row['expression_id'];comparison=prepare_pdg_evidence(row,symbols)
            if comparison!=row['evidence_json'].get('pdg_source_comparison'):raise ValueError('source evidence changed')
            fresh[key]=comparison;expressions[key]=deserialize_expr(row['sympy_srepr'])
            runtime=row['evidence_json'].get('numpy_runtime') or {}
            if runtime.get('runner_version')=='series-source-validation.v1':
                if runtime.get('tests_passed') is not True or runtime['source_comparison_sha256']!=_digest(comparison):raise ValueError('target validation mismatch')
                for path,expected in runtime['implementation_hashes'].items():
                    if hashlib.sha256((root/path).read_bytes()).hexdigest()!=expected:raise ValueError('target implementation changed')
                if not any(equivalent_residual_mapping(expressions[key],expressions[t['expression_id']],dimensions[key],dimensions[t['expression_id']]) is not None for t in targets):targets.append(row)
        counts['distinct_validated_implementations']=len(targets)
        for row in rows:
            key=row['expression_id'];counts['examined']+=1
            for target in targets:
                target_key=target['expression_id']
                try:mapping=equivalent_residual_mapping(expressions[key],expressions[target_key],dimensions[key],dimensions[target_key])
                except ValueError:
                    counts['outside_proof_scope']+=1;break
                if mapping is None:continue
                details={'scope':'exact polynomial residual reuse with SI-preserving variable renaming; no transfer of physical validity or publication approval',
                         'source_content_hash':row['content_hash'],'source_expression_id':str(key),
                         'target_expression_id':str(target_key),'target_version_id':str(target['version_id']),'target_content_hash':target['content_hash'],
                         'source_to_target_symbols':mapping,'source_dimensions':dimensions[key],'target_dimensions':dimensions[target_key],
                         'source_comparison_sha256':_digest(fresh[key]),'target_comparison_sha256':_digest(fresh[target_key]),
                         'target_runtime_evidence_sha256':_digest(target['evidence_json']['numpy_runtime']),
                         'implementation_sha256':hashlib.sha256((root/'sciona/physics_ingest/residual_equivalence.py').read_bytes()).hexdigest()}
                evidence_id=uuid5(UUID(str(row['version_id'])),'residual-reuse.v1:'+str(target_key))
                old=db.execute('SELECT details FROM artifact_audit_evidence WHERE evidence_id=%s FOR UPDATE',(evidence_id,)).fetchone()
                if old:
                    if old['details']!=details:raise ValueError('existing reuse proof changed')
                    counts['unchanged']+=1
                else:
                    db.execute("INSERT INTO artifact_audit_evidence(evidence_id,artifact_id,version_id,audit_type,passed,status,source_kind,runner_version,details) VALUES(%s,%s,%s,'semantic_audit',true,'completed','automated','residual-reuse.v1',%s)",(evidence_id,row['artifact_id'],row['version_id'],Jsonb(details)))
                    counts['proofs_recorded']+=1
                counts['matched']+=1;break
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'counts':dict(counts)},indent=2))


if __name__=='__main__':main()
