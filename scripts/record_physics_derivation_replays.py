#!/usr/bin/env python3
"""Record conditional source-algebra replay evidence and review-pending bounds."""
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
from sciona.physics_ingest.pdg_rule_contracts import load_pinned_rule_contracts
from sciona.physics_ingest.pdg_graph_replay import replay_derivation
from sciona.physics_ingest.pdg_evidence import prepare_pdg_evidence, _digest
from sciona.ghost.symbolic import deserialize_expr


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rule-file',required=True,type=Path)
    parser.add_argument('--symbol-file',required=True,type=Path)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    rule_bytes=args.rule_file.read_bytes();symbol_bytes=args.symbol_file.read_bytes()
    counts=Counter()
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        versions=db.execute("""SELECT v.version_id FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_cdg_bindings b USING(version_id)
 LEFT JOIN artifacts t ON t.fqdn=b.bound_artifact_fqdn LEFT JOIN artifact_versions tv ON tv.artifact_id=t.artifact_id AND tv.content_hash=b.bound_version_content_hash LEFT JOIN artifact_symbolic_expressions e ON e.version_id=tv.version_id
 WHERE a.fqdn LIKE 'physics.pdg.%' AND v.is_latest AND a.status='draft' GROUP BY v.version_id HAVING count(*)=count(*) FILTER(WHERE e.evidence_json->'dimensional_analysis'->>'status'='passed')""").fetchall()
        for selected in versions:
            version_id=selected['version_id']
            version=db.execute("SELECT v.*,a.status FROM artifact_versions v JOIN artifacts a USING(artifact_id) WHERE v.version_id=%s FOR UPDATE OF v,a",(version_id,)).fetchone()
            if version['status']!='draft':raise ValueError('graph status changed')
            nodes=db.execute('SELECT * FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id FOR SHARE',(version_id,)).fetchall()
            edges=db.execute('SELECT * FROM artifact_cdg_edges WHERE version_id=%s ORDER BY source_id,target_id FOR SHARE',(version_id,)).fetchall()
            bindings=db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s ORDER BY binding_id FOR SHARE',(version_id,)).fetchall()
            ids=set()
            for node in nodes:
                signature=json.loads(node['type_signature']);ids.update(signature['inputs']);ids.add(signature['output'])
            expressions=db.execute("SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload,a.fqdn,v.content_hash FROM artifact_symbolic_expressions e JOIN physics_equation_candidates q USING(candidate_id) JOIN physics_ingest_snapshots s USING(snapshot_id) JOIN artifact_versions v ON v.version_id=e.version_id JOIN artifacts a ON a.artifact_id=e.artifact_id WHERE e.expression_id=ANY(%s::uuid[]) FOR SHARE OF e,q,s,v,a",(list(ids),)).fetchall()
            resolved={};pins=set();evidence_basis={}
            targets={(b['bound_artifact_fqdn'],b['bound_version_content_hash']) for b in bindings if b['status']=='active'}
            for row in expressions:
                if (row['fqdn'],row['content_hash']) not in targets:raise ValueError('expression dependency is not pinned by the graph')
                fresh=prepare_pdg_evidence(row,symbol_bytes)
                if fresh!=row['evidence_json'].get('pdg_source_comparison'):raise ValueError('source evidence is stale')
                if fresh['source_comparison']['correspondence']!='exact_ast_match':raise ValueError('source correspondence is not exact')
                resolved[str(row['expression_id'])]=deserialize_expr(fresh['upstream_symbolic']['sympy_srepr'])
                evidence_basis[str(row['expression_id'])]=_digest(fresh)
                pins.add(row['snapshot_payload']['core_file_sha256']['conversion_of_data_formats/infrules.cypher'])
            if len(pins)!=1:raise ValueError('ambiguous source rule version')
            rules=load_pinned_rule_contracts(rule_bytes,next(iter(pins)))
            try:
                report=replay_derivation(nodes,resolved,rules,edges=[(e['source_id'],e['target_id']) for e in edges])
                passed=True
            except ValueError as exc:
                report={'scope':'source-algebra replay only; not approval','blocker':str(exc),'required_conditions':[]}
                passed=False
            root=Path(__file__).resolve().parents[1]
            report.update({
                'graph_content_hash':version['content_hash'],
                'graph_projection_sha256':_digest({'nodes':[{k:str(v) if k=='version_id' else v for k,v in n.items()} for n in nodes], 'edges':[{k:str(v) if k=='version_id' else v for k,v in e.items()} for e in edges]}),
                'expression_evidence_sha256':evidence_basis,
                'rule_file_sha256':next(iter(pins)),
                'implementation_hashes':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['sciona/physics_ingest/pdg_graph_replay.py','sciona/physics_ingest/pdg_rule_contracts.py','scripts/record_physics_derivation_replays.py']},
            })
            evidence_id=uuid5(UUID(str(version_id)), 'conditional-replay:'+_digest(report))
            existing=db.execute('SELECT details,passed FROM artifact_audit_evidence WHERE evidence_id=%s',(evidence_id,)).fetchone()
            if existing:
                if existing!={'details':report,'passed':passed}:raise ValueError('existing replay evidence drifted')
                for condition in report['required_conditions']:
                    bound=db.execute('SELECT validity_statement,metadata FROM artifact_validity_bounds WHERE bound_id=%s',(uuid5(evidence_id,condition),)).fetchone()
                    if not bound or bound['validity_statement']!='Necessary algebraic condition: '+condition or bound['metadata'].get('replay_evidence_id')!=str(evidence_id):
                        raise ValueError('recorded replay condition is missing or changed')
                counts['unchanged']+=1
                continue
            db.execute("INSERT INTO artifact_audit_evidence(evidence_id,artifact_id,version_id,audit_type,passed,status,details,source_kind,runner_version) VALUES(%s,%s,%s,'determinism_replay',%s,'completed',%s,'automated','pdg-graph-replay.v1')",(evidence_id,version['artifact_id'],version_id,passed,Jsonb(report)))
            for condition in report['required_conditions']:
                bound_id=uuid5(evidence_id,condition)
                db.execute("INSERT INTO artifact_validity_bounds(bound_id,artifact_id,version_id,scope,bound_kind,validity_statement,review_status,metadata) VALUES(%s,%s,%s,'version','assumption',%s,'needs_human',%s)",
                    (bound_id,version['artifact_id'],version_id,'Necessary algebraic condition: '+condition,Jsonb({'replay_evidence_id':str(evidence_id),'condition_srepr':condition,'scope':'algebraic replay domain; physical premises not reviewed'})))
                counts['pending_conditions']+=1
            counts['conditional_replay_passed' if passed else 'replay_blocked']+=1
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'counts':dict(counts)},indent=2))


if __name__=='__main__':main()
