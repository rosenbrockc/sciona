#!/usr/bin/env python3
"""Attach scientific references and review-pending regimes to exact source dependencies."""
import argparse
from collections import Counter
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
from sciona.physics_ingest.series_source_review import classify_series_equation


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--symbol-file',required=True,type=Path);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    symbols=args.symbol_file.read_bytes();counts=Counter()
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        runtimes=db.execute("SELECT * FROM artifact_audit_evidence WHERE runner_version='series-terminal-runtime.v1' AND passed FOR SHARE").fetchall()
        for runtime in runtimes:
            graph_version=runtime['version_id'];details=runtime['details']
            replay=db.execute('SELECT details FROM artifact_audit_evidence WHERE evidence_id=%s AND passed FOR SHARE',(details['replay_evidence_id'],)).fetchone()
            if not replay or _digest(replay['details'])!=details['replay_details_sha256']:raise ValueError('replay reference drift')
            reference=db.execute('SELECT ref_id,ref_key,title,url,verified,source FROM artifact_references WHERE artifact_id=%s AND url=%s FOR SHARE',(runtime['artifact_id'],details['scientific_reference'])).fetchall()
            if len(reference)!=1 or not reference[0]['verified']:raise ValueError('verified scientific reference required')
            reference=reference[0]
            ids=list(replay['details']['expression_evidence_sha256'])
            rows=db.execute("SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload FROM artifact_symbolic_expressions e JOIN physics_equation_candidates q USING(candidate_id) JOIN physics_ingest_snapshots s USING(snapshot_id) WHERE e.expression_id=ANY(%s::uuid[]) ORDER BY e.expression_id FOR UPDATE OF e FOR SHARE OF q,s",(ids,)).fetchall()
            if len(rows)!=len(ids):raise ValueError('source dependency missing')
            for row in rows:
                if row['review_status']!='needs_human':raise ValueError('existing human review must be preserved')
                fresh=prepare_pdg_evidence(row,symbols)
                if fresh!=row['evidence_json'].get('pdg_source_comparison') or _digest(fresh)!=replay['details']['expression_evidence_sha256'][str(row['expression_id'])]:raise ValueError('source expression evidence changed')
                variables=db.execute('SELECT source_symbol,dim_signature FROM artifact_symbolic_variables WHERE expression_id=%s FOR SHARE',(row['expression_id'],)).fetchall()
                expression=deserialize_expr(fresh['upstream_symbolic']['sympy_srepr'])
                classification=classify_series_equation(expression,{r['source_symbol']:r['dim_signature'] for r in variables})
                classification_evidence={'mechanism':classification,'basis':'exact source AST and SI dimensions in a source-supported ohmic series derivation','scientific_reference':reference['url'],'source_graph_version_id':str(graph_version),'source_runtime_evidence_id':str(runtime['evidence_id']),'source_comparison_sha256':_digest(fresh),'scope':'automated classification and reference association, not human approval','implementation_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'sciona/physics_ingest/series_source_review.py').read_bytes()).hexdigest()}
                evidence=dict(row['evidence_json']);old=evidence.get('mechanism_classification')
                if old is not None and old!=classification_evidence:raise ValueError('existing mechanism evidence conflicts')
                evidence['mechanism_classification']=classification_evidence
                existing_ref=db.execute('SELECT ref_id,url,verified FROM artifact_references WHERE artifact_id=%s AND ref_key=%s',(row['artifact_id'],reference['ref_key'])).fetchall()
                if existing_ref:
                    if existing_ref!=[{k:reference[k] for k in ['ref_id','url','verified']}]:raise ValueError('existing reference conflicts')
                else:
                    db.execute("INSERT INTO artifact_references(artifact_id,ref_id,ref_key,title,url,verified,source,relevance_note,confidence) VALUES(%s,%s,%s,%s,%s,true,'llm_extracted',%s,'high')",(row['artifact_id'],reference['ref_id'],reference['ref_key'],reference['title'],reference['url'],'Supports the classified equation in an ohmic series circuit; publication verification is not human expression approval.'))
                    counts['references_created']+=1
                statement='Review scope for this source derivation: '+details['physical_regime']
                bound_id=uuid5(UUID(str(row['expression_id'])),'series-source-regime:'+str(graph_version))
                bound=db.execute('SELECT validity_statement FROM artifact_validity_bounds WHERE bound_id=%s',(bound_id,)).fetchone()
                if bound:
                    if bound['validity_statement']!=statement:raise ValueError('existing validity regime conflicts')
                else:
                    db.execute("INSERT INTO artifact_validity_bounds(bound_id,artifact_id,version_id,expression_id,scope,bound_kind,validity_statement,evidence_ref_key,review_status,metadata) VALUES(%s,%s,%s,%s,'expression','regime',%s,%s,'needs_human',%s)",(bound_id,row['artifact_id'],row['version_id'],row['expression_id'],statement,reference['ref_key'],Jsonb({'source_graph_version_id':str(graph_version),'classification':classification,'scope':'physical regime pending human review; graph-specific algebraic conditions remain separately required'})))
                    counts['bounds_created']+=1
                if old is None:
                    db.execute('UPDATE artifact_symbolic_expressions SET evidence_json=%s WHERE expression_id=%s',(Jsonb(evidence),row['expression_id']))
                    counts['classifications_created']+=1
                counts['examined']+=1
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'counts':dict(counts)},indent=2))


if __name__=='__main__':main()
