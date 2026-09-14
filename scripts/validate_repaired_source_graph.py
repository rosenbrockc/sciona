#!/usr/bin/env python3
"""Select exact source alternatives for the repaired graph and attempt replay."""
import argparse
import hashlib
import json
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
from dotenv import dotenv_values
from sciona.physics_ingest.pdg_evidence import prepare_pdg_evidence,_digest
from sciona.physics_ingest.pdg_symbols import load_pinned_pdg_scalars
from sciona.physics_ingest.source_identity_normalization import normalize_source_identities
from sciona.physics_ingest.pdg_rule_contracts import load_pinned_rule_contracts
from sciona.physics_ingest.pdg_graph_replay_v4 import replay_derivation
from sciona.ghost.symbolic import deserialize_expr

VERSION='76274cb1-ebf6-533d-a777-b505190f3d17'


def validate(root,symbol_file,rule_file):
    symbol_bytes=symbol_file.read_bytes();selected={};substitutions={};pins=set();binding_versions=[]
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,options='-c default_transaction_read_only=on') as db:
        graph=db.execute('SELECT content_hash FROM artifact_versions WHERE version_id=%s',(VERSION,)).fetchone()
        nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id',(VERSION,)).fetchall()
        edges=db.execute('SELECT source_id,target_id FROM artifact_cdg_edges WHERE version_id=%s ORDER BY source_id,target_id',(VERSION,)).fetchall()
        bindings=db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s ORDER BY binding_id',(VERSION,)).fetchall()
        if len(nodes)!=13 or len(bindings)!=31:raise ValueError('Expected repaired graph changed')
        for binding in bindings:
            if binding['status']!='active':raise ValueError('Inactive binding')
            rows=db.execute("SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload,v.content_hash,old.expression_id AS previous_expression_id FROM artifacts a JOIN artifact_versions ov ON ov.artifact_id=a.artifact_id JOIN artifact_symbolic_expressions old ON old.version_id=ov.version_id JOIN artifact_versions v ON v.artifact_id=a.artifact_id AND (v.version_id=ov.version_id OR v.derives_from=ov.version_id) JOIN artifact_symbolic_expressions e ON e.version_id=v.version_id JOIN physics_equation_candidates q ON q.candidate_id=e.candidate_id JOIN physics_ingest_snapshots s ON s.snapshot_id=q.snapshot_id WHERE a.fqdn=%s AND ov.content_hash=%s AND e.evidence_json->'dimensional_analysis'->>'status'='passed'",(binding['bound_artifact_fqdn'],binding['bound_version_content_hash'])).fetchall()
            if len(rows)!=1:raise ValueError('Unique dimensioned expression alternative required')
            row=rows[0];fresh=prepare_pdg_evidence(row,symbol_bytes)
            identity_evidence=row['evidence_json'].get('source_identity_comparison')
            if identity_evidence:
                if identity_evidence['correspondence']!='exact_source_identity_ast' or _digest(fresh['upstream_symbolic']['sympy_srepr'])!=identity_evidence['source_ast_sha256']:raise ValueError('Source-ID evidence differs')
                normalized,labels=normalize_source_identities(fresh['upstream_symbolic']['sympy_srepr'],load_pinned_pdg_scalars(symbol_bytes,fresh['symbol_file_sha256']))
                if normalized.srepr_str!=row['sympy_srepr'] or normalized.dimensional_hash!=row['dimensional_hash']:raise ValueError('Source-ID normalization differs')
                provenance=row['evidence_json']['source_normalization']
                if provenance['display_labels']!=labels:raise ValueError('Source labels differ')
                for p,sha in provenance['implementation_hashes'].items():
                    if hashlib.sha256((root/p).read_bytes()).hexdigest()!=sha:raise ValueError('Normalization implementation differs')
            elif fresh!=row['evidence_json'].get('pdg_source_comparison') or fresh['source_comparison']['correspondence']!='exact_ast_match':
                raise ValueError('Source-label correspondence differs')
            identity=str(row['expression_id']);selected[identity]=dict(source_srepr=fresh['upstream_symbolic']['sympy_srepr'],version_id=str(row['version_id']),content_hash=row['content_hash'],evidence_sha256=_digest(row['evidence_json']))
            substitutions[str(row['previous_expression_id'])]=identity
            pins.add(row['snapshot_payload']['core_file_sha256']['conversion_of_data_formats/infrules.cypher'])
            binding_versions.append(dict(node_id=binding['node_id'],fqdn=binding['bound_artifact_fqdn'],selected_version_id=str(row['version_id']),selected_content_hash=row['content_hash']))
        if len(pins)!=1:raise ValueError('Ambiguous rule pin')
        rules=load_pinned_rule_contracts(rule_file.read_bytes(),next(iter(pins)))
        for node in nodes:
            s=json.loads(node['type_signature']);s['inputs']=[substitutions[i] for i in s['inputs']];s['outputs']=[substitutions[i] for i in s.get('outputs',[s['output']])];s['output']=substitutions[s['output']];node['type_signature']=s
        resolved={i:deserialize_expr(r['source_srepr']) for i,r in selected.items()}
        try:
            replay=replay_derivation(nodes,resolved,rules,edges=[(e['source_id'],e['target_id']) for e in edges]);passed=True
        except ValueError as error:replay=dict(blocker=str(error));passed=False
        operations=[dict(node_id=n['node_id'],rule=rules[n['type_signature']['inference_rule_id']].name) for n in nodes]
    files=['scripts/validate_repaired_source_graph.py','sciona/physics_ingest/pdg_graph_replay_v4.py','sciona/physics_ingest/pdg_rational_rules_v4.py','sciona/physics_ingest/pdg_real_power_rules.py','sciona/physics_ingest/pdg_rational_rules_v2.py','sciona/physics_ingest/pdg_rule_contracts.py']
    return dict(read_only=True,approval_applied=False,source_graph_version_id=VERSION,source_graph_content_hash=graph['content_hash'],
        selected_expressions=selected,expression_substitutions=substitutions,bindings=binding_versions,nodes=nodes,edges=edges,
        replay_passed=passed,replay=replay,operations=operations,rule_file_sha256=next(iter(pins)),
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},
        scope='Exact alternative selection proposal; no catalog rebinding or publication approval.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file','rule-file','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();result=validate(Path(__file__).resolve().parents[1],args.symbol_file,args.rule_file)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(selected_expressions=len(result['selected_expressions']),bindings=len(result['bindings']),replay_passed=result['replay_passed'],replay=result['replay'])))
