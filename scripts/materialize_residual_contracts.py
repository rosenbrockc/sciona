#!/usr/bin/env python3
"""Materialize validated residual interfaces and version-bound automated review evidence."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from uuid import UUID,uuid5
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sciona.physics_ingest.pdg_evidence import prepare_pdg_evidence,_digest
from sciona.physics_ingest.residual_contract import build_residual_contract
from sciona.physics_ingest.review import assess_publishability


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbol-file',required=True,type=Path)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();symbols=args.symbol_file.read_bytes();counts=Counter()
    root=Path(__file__).resolve().parents[1]
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        rows=db.execute("SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload,v.content_hash FROM artifact_symbolic_expressions e JOIN physics_equation_candidates q USING(candidate_id) JOIN physics_ingest_snapshots s USING(snapshot_id) JOIN artifact_versions v ON v.version_id=e.version_id WHERE e.evidence_json->'numpy_runtime'->>'runner_version'='series-source-validation.v1' ORDER BY e.expression_id FOR UPDATE OF e,v FOR SHARE OF q,s").fetchall()
        for row in rows:
            fresh=prepare_pdg_evidence(row,symbols)
            if fresh!=row['evidence_json'].get('pdg_source_comparison'):raise ValueError('source evidence drift')
            runtime=row['evidence_json']['numpy_runtime']
            if runtime['source_comparison_sha256']!=_digest(fresh):raise ValueError('runtime source evidence drift')
            for relative,expected in runtime['implementation_hashes'].items():
                if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=expected:raise ValueError('validation implementation drift')
            variables=db.execute('SELECT * FROM artifact_symbolic_variables WHERE expression_id=%s ORDER BY ordinal FOR SHARE',(row['expression_id'],)).fetchall()
            contract=build_residual_contract(row,variables)
            candidate=db.execute('SELECT * FROM physics_equation_candidates WHERE candidate_id=%s',(row['candidate_id'],)).fetchone()
            references=db.execute('SELECT * FROM artifact_references WHERE artifact_id=%s FOR SHARE',(row['artifact_id'],)).fetchall()
            bounds=db.execute('SELECT * FROM artifact_validity_bounds WHERE expression_id=%s FOR SHARE',(row['expression_id'],)).fetchall()
            assessment=assess_publishability(candidate=candidate,expression=row,variables=variables,references=references,validity_bounds=bounds,io_specs=contract['ports'])
            if not assessment.publishable:raise ValueError('Community automated assessment failed')
            existing=db.execute('SELECT direction,ordinal,name,type_desc,dim_signature,constraints,required,default_value_repr FROM artifact_io_specs WHERE version_id=%s ORDER BY direction,ordinal FOR UPDATE',(row['version_id'],)).fetchall()
            if existing and existing!=contract['ports']:raise ValueError('existing interface differs; refusing replacement')
            if not existing:
                for port in contract['ports']:
                    fields={**port,'artifact_id':row['artifact_id'],'version_id':row['version_id']}
                    db.execute(sql.SQL('INSERT INTO artifact_io_specs ({}) VALUES ({})').format(sql.SQL(',').join(map(sql.Identifier,fields)),sql.SQL(',').join(sql.Placeholder() for _ in fields)),tuple(fields.values()))
                    counts['io_rows_created']+=1
            duplicates=db.execute('SELECT count(*) AS n FROM artifact_symbolic_expressions WHERE canonical_expr_hash=%s',(row['canonical_expr_hash'],)).fetchone()['n']
            related=db.execute('SELECT count(*) AS n FROM artifact_symbolic_expressions WHERE topology_hash=%s',(row['topology_hash'],)).fetchone()['n']
            details={'publication_tier':3,'scope':'automated residual implementation review; source equation preserved; publication baseline still requires duplication and runtime-loader closure',
                     'content_hash':row['content_hash'],'expression_id':str(row['expression_id']),
                     'contract':contract,'assessment':assessment.to_report().to_dict(),
                     'validation_evidence_sha256':_digest(runtime),'source_comparison_sha256':_digest(fresh),
                     'structural_status':'pass','runtime_status':'pass','semantic_status':'conditional_pass','developer_semantics_status':'conditional_pass',
                     'conditions':contract['limitations'],'duplicate_screen':{'exact_expression_count':duplicates,'same_topology_count':related},
                     'remaining_publication_actions':['Confirm runtime loading through the catalog execution path','Resolve equivalent implementations before publishing a new atom'],
                     'implementation_sha256':hashlib.sha256((root/'sciona/physics_ingest/residual_contract.py').read_bytes()).hexdigest()}
            evidence_id=uuid5(UUID(str(row['version_id'])),'residual-contract-review.v1')
            previous=db.execute('SELECT details FROM artifact_audit_evidence WHERE evidence_id=%s FOR UPDATE',(evidence_id,)).fetchone()
            if previous:
                if previous['details']!=details:raise ValueError('existing review evidence differs')
                counts['unchanged_reviews']+=1
            else:
                db.execute("INSERT INTO artifact_audit_evidence(evidence_id,artifact_id,version_id,audit_type,passed,status,source_kind,runner_version,details) VALUES(%s,%s,%s,'semantic_audit',true,'completed','automated','residual-contract-review.v1',%s)",(evidence_id,row['artifact_id'],row['version_id'],Jsonb(details)))
                counts['reviews_created']+=1
            counts['examined']+=1
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'counts':dict(counts)},indent=2))


if __name__=='__main__':main()
