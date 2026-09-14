#!/usr/bin/env python3
"""Diagnose elementary arithmetic in pinned multiple-output source proofs."""
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
from sciona.physics_ingest.pdg_step_diagnostics import arithmetic_step_differences
from sciona.physics_ingest.source_symbolic import parse_source_srepr
import sympy as sp


def validate(root,symbol_file,rule_file):
    reports=[]
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        rows=db.execute("SELECT e.version_id,e.evidence_id,e.details,v.content_hash FROM artifact_audit_evidence e JOIN artifact_versions v USING(version_id) WHERE e.runner_version='pdg-source-graph-replay.v2' AND NOT e.passed ORDER BY e.version_id").fetchall()
        for row in rows:
            nodes=db.execute('SELECT * FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id',(row['version_id'],)).fetchall()
            signatures=[json.loads(n['type_signature']) for n in nodes]
            if not any(len(s.get('outputs',[]))>1 for s in signatures):continue
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
            steps=[]
            for node,signature in zip(nodes,signatures):
                rule=rules[signature['inference_rule_id']]
                if len(signature['outputs'])!=1:
                    steps.append(dict(node_id=node['node_id'],status='not_evaluated_multiple_outputs'))
                    continue
                inputs=[resolved[i] for i in signature['inputs']]
                expected=resolved[signature['output']]
                feeds=[parse_source_srepr(f['sympy']) for f in signature['variable_bindings'].get('feeds',[])]
                deltas=arithmetic_step_differences(rule,inputs,feeds,expected)
                if deltas is None:
                    steps.append(dict(node_id=node['node_id'],status='not_evaluated_operation'))
                    continue
                record=dict(node_id=node['node_id'],rule=rule.name,
                    status='exact_arithmetic' if deltas==(0,0) else 'symbolic_difference',
                    side_differences=[sp.srepr(d) for d in deltas])
                assignments={
                    'pdg_step_1':(2,3,1,-1),
                    'pdg_step_2':(2,3,-4,1),
                    'pdg_step_8':(2,3,1,-1),
                    'pdg_step_9':(2,3,1,sp.Rational(-1,2)),
                }
                if node['node_id'] in assignments:
                    names=['pdg0009139','pdg0001939','pdg0004231','pdg0001464']
                    values=dict(zip(map(sp.Symbol,names),map(sp.sympify,assignments[node['node_id']])))
                    premise=sp.simplify(inputs[0].subs(values))
                    conclusion=sp.simplify(expected.subs(values))
                    if premise!=sp.true or conclusion!=sp.false:
                        raise ValueError('Claimed source counterexample does not hold')
                    if any(sp.simplify(f.subs(values)).is_finite is not True for f in feeds):
                        raise ValueError('Nonfinite counterexample feed')
                    record['counterexample']=dict(assignments={str(k):sp.srepr(v) for k,v in values.items()},
                        premise_satisfied=True,conclusion_satisfied=False,
                        scope='Finite real example with nonzero quadratic coefficient and real distinct roots; no extra a=b constraint imposed.')
                steps.append(record)
            report=dict(steps=steps,scope='Independent elementary-operation diagnostics, not composed replay or approval')
            report.update(version_id=str(row['version_id']),content_hash=row['content_hash'],
                prior_failed_evidence_id=str(row['evidence_id']),prior_evidence_sha256=_digest(row['details']),
                expression_evidence_sha256=basis)
            reports.append(report)
    files=['sciona/physics_ingest/pdg_step_diagnostics.py','scripts/audit_physics_branch_steps.py','tests/physics_ingest/test_pdg_step_diagnostics.py']
    return dict(read_only=True,approval_applied=False,graphs=reports,
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},
        limitations=['Independent operation diagnostics; zero differences do not establish a complete valid proof.',
            'Previous failed replay evidence is retained; this validator does not modify catalog state.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file','rule-file','output']:parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args()
    result=validate(Path(__file__).resolve().parents[1],args.symbol_file,args.rule_file)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(graphs_diagnosed=len(result['graphs']),steps=sum(len(g['steps']) for g in result['graphs']),approval_applied=False)))
