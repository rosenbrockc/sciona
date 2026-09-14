#!/usr/bin/env python3
"""Create non-latest source-AST graph versions and replay their exact expression pins."""
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
from sciona.physics_ingest.pdg_rule_contracts import load_pinned_rule_contracts
from sciona.physics_ingest.pdg_graph_replay_v2 import replay_derivation
from scripts.import_residual_execution_drafts import ensure_row,comparable


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--symbol-file',required=True,type=Path);parser.add_argument('--rule-file',required=True,type=Path);parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();symbol_bytes=args.symbol_file.read_bytes();counts=Counter();root=Path(__file__).resolve().parents[1]
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('pdg-source-graph.v1'))")
        versions=db.execute("SELECT v.* FROM artifact_versions v JOIN artifacts a USING(artifact_id) WHERE a.fqdn LIKE 'physics.pdg.%' AND v.is_latest AND a.status='draft' AND NOT a.is_publishable FOR SHARE OF a,v").fetchall()
        for version in versions:
            nodes=db.execute('SELECT * FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id FOR SHARE',(version['version_id'],)).fetchall()
            signatures=[json.loads(n['type_signature']) for n in nodes]
            if not signatures or not all(s.get('source_pdg_step_id') for s in signatures):continue
            bindings=db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s ORDER BY binding_id FOR SHARE',(version['version_id'],)).fetchall()
            selected={};substitutions={};selected_hashes={};valid=bool(bindings)
            for binding in bindings:
                if binding['status']!='active':valid=False;break
                candidates=db.execute("SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload,v.content_hash,old.expression_id AS previous_expression_id FROM artifacts a JOIN artifact_versions ov ON ov.artifact_id=a.artifact_id JOIN artifact_symbolic_expressions old ON old.version_id=ov.version_id JOIN artifact_versions v ON v.artifact_id=a.artifact_id AND (v.version_id=ov.version_id OR v.derives_from=ov.version_id) JOIN artifact_symbolic_expressions e ON e.version_id=v.version_id JOIN physics_equation_candidates q ON q.candidate_id=e.candidate_id JOIN physics_ingest_snapshots s ON s.snapshot_id=q.snapshot_id WHERE a.fqdn=%s AND ov.content_hash=%s AND e.evidence_json->'dimensional_analysis'->>'status'='passed' FOR SHARE OF a,ov,old,v,e,q,s",(binding['bound_artifact_fqdn'],binding['bound_version_content_hash'])).fetchall()
                if len(candidates)!=1:valid=False;break
                chosen=candidates[0];selected[str(chosen['expression_id'])]=chosen;substitutions[str(chosen['previous_expression_id'])]=str(chosen['expression_id'])
                selected_hashes[(binding['bound_artifact_fqdn'],binding['bound_version_content_hash'])]=chosen['content_hash']
                binding['bound_version_content_hash']=chosen['content_hash']
            if not valid or not any(a!=b for a,b in substitutions.items()):continue
            resolved={};evidence_basis={};pins=set()
            for expression_id,row in selected.items():
                fresh=prepare_pdg_evidence(row,symbol_bytes)
                if fresh!=row['evidence_json']['pdg_source_comparison'] or fresh['source_comparison']['correspondence']!='exact_ast_match':raise ValueError('selected source correspondence is not exact')
                resolved[expression_id]=deserialize_expr(fresh['upstream_symbolic']['sympy_srepr']);evidence_basis[expression_id]=_digest(fresh)
                pins.add(row['snapshot_payload']['core_file_sha256']['conversion_of_data_formats/infrules.cypher'])
            if len(pins)!=1:raise ValueError('source rule pins differ')
            rules=load_pinned_rule_contracts(args.rule_file.read_bytes(),next(iter(pins)))
            for node,signature in zip(nodes,signatures):
                signature['inputs']=[substitutions[i] for i in signature['inputs']]
                signature['output']=substitutions[signature['output']]
                signature['outputs']=[substitutions[i] for i in signature.get('outputs',[signature['output']])] if 'outputs' in signature else [signature['output']]
                signature['source_expression_substitutions']={a:b for a,b in substitutions.items() if a!=b}
                node['type_signature']=json.dumps(signature,sort_keys=True)
            edges=db.execute('SELECT * FROM artifact_cdg_edges WHERE version_id=%s ORDER BY source_id,target_id FOR SHARE',(version['version_id'],)).fetchall()
            digest=_digest({'kind':'pdg-source-graph.v1','previous_content_hash':version['content_hash'],'nodes':comparable([{k:v for k,v in n.items() if k!='version_id'} for n in nodes]),'bindings':comparable([{k:v for k,v in b.items() if k not in {'binding_id','version_id'}} for b in bindings])})
            new_id=uuid5(UUID(str(version['artifact_id'])),'pdg-source-graph.v1:'+digest)
            created=ensure_row(db,'artifact_versions',{'version_id':new_id},{'version_id':new_id,'artifact_id':version['artifact_id'],'content_hash':digest,'semver':'0.0.0+source-graph.'+digest[:12],'derives_from':version['version_id'],'is_latest':False,'trust_tier':3})
            for node in nodes:
                node['version_id']=new_id;ensure_row(db,'artifact_cdg_nodes',{'version_id':new_id,'node_id':node['node_id']},node)
            for edge in edges:
                edge.pop('edge_id',None);edge['version_id']=new_id
                ensure_row(db,'artifact_cdg_edges',{k:edge[k] for k in ['version_id','source_id','target_id','input_name','output_name']},edge)
            for binding in bindings:
                old_id=binding.pop('binding_id');binding['binding_id']=uuid5(new_id,str(old_id));binding['version_id']=new_id
                binding['binding_source']='pdg-source-graph.v1';binding['evidence_summary']=Jsonb({'scope':'explicit source-AST expression version selection; not publication approval','previous_graph_version_id':str(version['version_id'])})
                binding['alternatives']=Jsonb(binding['alternatives'])
                ensure_row(db,'artifact_cdg_bindings',{'binding_id':binding['binding_id']},binding)
            for port in db.execute('SELECT * FROM artifact_io_specs WHERE version_id=%s FOR SHARE',(version['version_id'],)).fetchall():
                old_id=port['io_spec_id'];port['io_spec_id']=uuid5(new_id,str(old_id));port['version_id']=new_id
                ensure_row(db,'artifact_io_specs',{'io_spec_id':port['io_spec_id']},port)
            for dependency in db.execute('SELECT * FROM artifact_dependencies WHERE dependent_version_id=%s FOR SHARE',(version['version_id'],)).fetchall():
                old_id=dependency['dependency_id'];dependency['dependency_id']=uuid5(new_id,str(old_id));dependency['dependent_version_id']=new_id
                dependency['dependency_content_hash']=selected_hashes.get((dependency['dependency_artifact_fqdn'],dependency['dependency_content_hash']),dependency['dependency_content_hash'])
                dependency['binding_metadata']=Jsonb(dependency['binding_metadata'])
                ensure_row(db,'artifact_dependencies',{'dependency_id':dependency['dependency_id']},dependency)
            try:
                report=replay_derivation(nodes,resolved,rules,edges=[(e['source_id'],e['target_id']) for e in edges]);passed=True
            except ValueError as error:report={'blocker':str(error),'required_conditions':[]};passed=False
            report.update({'scope':'exact pinned source-AST graph replay; physical validation and agreement with original LaTeX remain separate','content_hash':digest,'previous_version_id':str(version['version_id']),'expression_substitutions':substitutions,'expression_evidence_sha256':evidence_basis,'implementation_hashes':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['sciona/physics_ingest/pdg_graph_replay_v2.py','sciona/physics_ingest/pdg_rational_rules_v2.py','sciona/physics_ingest/pdg_rule_contracts.py']}})
            audit_id=uuid5(new_id,'source-graph-replay.v2')
            ensure_row(db,'artifact_audit_evidence',{'evidence_id':audit_id},{'evidence_id':audit_id,'artifact_id':version['artifact_id'],'version_id':new_id,'audit_type':'determinism_replay','passed':passed,'status':'completed','source_kind':'automated','runner_version':'pdg-source-graph-replay.v2','details':Jsonb(report)})
            for condition in report['required_conditions']:
                bound_id=uuid5(audit_id,condition)
                ensure_row(db,'artifact_validity_bounds',{'bound_id':bound_id},{'bound_id':bound_id,'artifact_id':version['artifact_id'],'version_id':new_id,'scope':'version','bound_kind':'assumption','validity_statement':'Necessary algebraic condition: '+condition,'review_status':'automated_pass','metadata':Jsonb({'replay_evidence_id':str(audit_id),'scope':'required algebraic domain condition, not a physical-regime validation'})})
            counts['created_versions' if created else 'unchanged_versions']+=1;counts['replay_passed' if passed else report['blocker']]+=1
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'counts':dict(counts)},indent=2))


if __name__=='__main__':main()
