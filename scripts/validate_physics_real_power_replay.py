#!/usr/bin/env python3
"""Revalidate pinned source graphs using explicit forward real-root squaring."""
import argparse
import hashlib
import json
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
from dotenv import dotenv_values
from sciona.ghost.symbolic import deserialize_expr
from sciona.physics_ingest.pdg_evidence import prepare_pdg_evidence, _digest
from sciona.physics_ingest.pdg_rule_contracts import load_pinned_rule_contracts
from sciona.physics_ingest.pdg_graph_replay_v3 import replay_derivation


def validate(root,symbol_file,rule_file):
    reports=[]
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        rows=db.execute("SELECT e.version_id,e.evidence_id,e.details,v.content_hash FROM artifact_audit_evidence e JOIN artifact_versions v USING(version_id) WHERE e.runner_version='pdg-source-graph-replay.v2' AND NOT e.passed ORDER BY e.version_id").fetchall()
        for row in rows:
            nodes=db.execute('SELECT * FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id',(row['version_id'],)).fetchall()
            signatures=[json.loads(n['type_signature']) for n in nodes]
            if not any(s['inference_rule_id']=='111483' for s in signatures):continue
            edges=db.execute('SELECT source_id,target_id FROM artifact_cdg_edges WHERE version_id=%s ORDER BY source_id,target_id',(row['version_id'],)).fetchall()
            bindings=db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s',(row['version_id'],)).fetchall()
            ids={i for s in signatures for i in [*s['inputs'],*s.get('outputs',[s['output']])]}
            expressions=db.execute("SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload,a.fqdn,v.content_hash FROM artifact_symbolic_expressions e JOIN physics_equation_candidates q USING(candidate_id) JOIN physics_ingest_snapshots s USING(snapshot_id) JOIN artifact_versions v ON v.version_id=e.version_id JOIN artifacts a ON a.artifact_id=e.artifact_id WHERE e.expression_id=ANY(%s::uuid[])",(list(ids),)).fetchall()
            if {str(e['expression_id']) for e in expressions}!=ids:raise ValueError('Missing expression')
            resolved={};basis={};pins=set()
            for e in expressions:
                if not any(b['status']=='active' and b['bound_artifact_fqdn']==e['fqdn'] and b['bound_version_content_hash']==e['content_hash'] for b in bindings):raise ValueError('Missing exact expression binding')
                fresh=prepare_pdg_evidence(e,symbol_file.read_bytes())
                if fresh!=e['evidence_json'].get('pdg_source_comparison') or fresh['source_comparison']['correspondence']!='exact_ast_match':raise ValueError('Source evidence differs')
                if e['evidence_json'].get('dimensional_analysis',{}).get('status')!='passed':raise ValueError('Dimension gate missing')
                identity=str(e['expression_id'])
                resolved[identity]=deserialize_expr(fresh['upstream_symbolic']['sympy_srepr'])
                basis[identity]=_digest(fresh)
                pins.add(e['snapshot_payload']['core_file_sha256']['conversion_of_data_formats/infrules.cypher'])
            if len(pins)!=1:raise ValueError('Ambiguous rule pin')
            rules=load_pinned_rule_contracts(rule_file.read_bytes(),next(iter(pins)))
            report=replay_derivation(nodes,resolved,rules,edges=[(e['source_id'],e['target_id']) for e in edges])
            report.update(version_id=str(row['version_id']),content_hash=row['content_hash'],
                prior_failed_evidence_id=str(row['evidence_id']),prior_evidence_sha256=_digest(row['details']),
                expression_evidence_sha256=basis,passed=True)
            reports.append(report)
    files=['sciona/physics_ingest/pdg_graph_replay_v3.py','sciona/physics_ingest/pdg_real_power_rules.py',
        'sciona/physics_ingest/pdg_rational_rules_v2.py','sciona/physics_ingest/pdg_rule_contracts.py',
        'scripts/validate_physics_real_power_replay.py','tests/physics_ingest/test_pdg_real_power_rules.py']
    return dict(read_only=True,approval_applied=False,graphs=reports,
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},
        limitations=['Conditional source-algebra consequences only; no reverse square-root inference or physical regime certification.',
            'Previous failed replay evidence is retained; this validator does not modify catalog state.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file','rule-file','output']:parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args()
    result=validate(Path(__file__).resolve().parents[1],args.symbol_file,args.rule_file)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(graphs_passed=len(result['graphs']),steps=sum(len(g['steps']) for g in result['graphs']),approval_applied=False)))
