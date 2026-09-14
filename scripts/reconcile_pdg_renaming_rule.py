#!/usr/bin/env python3
"""Correct a mislabeled source renaming rule only after exact, dimensioned proof."""
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
from sciona.ghost.symbolic import deserialize_expr
from sciona.physics_ingest.pdg_evidence import prepare_pdg_evidence,_digest
from sciona.physics_ingest.pdg_symbols import load_pinned_pdg_scalars
from sciona.physics_ingest.pdg_rule_contracts import load_pinned_rule_contracts,contract_blockers
from sciona.physics_ingest.pdg_rule_reconciliation import prove_variable_renaming
from sciona.physics_ingest.source_symbolic import parse_source_srepr
from scripts.repair_pdg_input_slots import insert


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbol-file',required=True,type=Path);parser.add_argument('--rule-file',required=True,type=Path)
    parser.add_argument('--apply',action='store_true');args=parser.parse_args();counts=Counter();symbol_bytes=args.symbol_file.read_bytes()
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('pdg-renaming-reconciliation.v1'))")
        pins=db.execute("SELECT DISTINCT payload->'core_file_sha256'->>'conversion_of_data_formats/symbols.cypher' AS symbols,payload->'core_file_sha256'->>'conversion_of_data_formats/infrules.cypher' AS rules FROM physics_ingest_snapshots WHERE payload ? 'core_file_sha256'").fetchall()
        if len(pins)!=1:raise ValueError('source pins must be unambiguous')
        definitions=load_pinned_pdg_scalars(symbol_bytes,pins[0]['symbols'])
        dimensions={name:item.dimension.to_compact() for name,item in definitions.items() if item.dimension is not None}
        rules=load_pinned_rule_contracts(args.rule_file.read_bytes(),pins[0]['rules'])
        replacement_ids=[key for key,r in rules.items() if r.name=='change three variables in expr' and (r.inputs,r.feeds,r.outputs)==(1,6,1)]
        if len(replacement_ids)!=1:raise ValueError('replacement rule must resolve uniquely')
        replacement=replacement_ids[0]
        versions=db.execute("SELECT v.* FROM artifact_versions v JOIN artifacts a USING(artifact_id) WHERE a.fqdn LIKE 'physics.pdg.%' AND a.status='draft' AND NOT a.is_publishable AND v.is_latest FOR UPDATE OF v FOR SHARE OF a").fetchall()
        for version in versions:
            nodes=db.execute('SELECT * FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id FOR SHARE',(version['version_id'],)).fetchall();changes=[]
            for node in nodes:
                signature=json.loads(node['type_signature']);old_rule=rules.get(signature.get('inference_rule_id'))
                if not signature.get('source_pdg_step_id') or not old_rule or old_rule.name!='change four variables in expr':continue
                feeds=signature['variable_bindings'].get('feeds',[])
                if len(feeds)!=6 or len(signature['inputs'])!=1 or len(signature.get('outputs',[]))!=1:continue
                expressions=[];comparison_hashes={}
                for expression_id in [signature['inputs'][0],signature['output']]:
                    row=db.execute('SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload FROM artifact_symbolic_expressions e JOIN physics_equation_candidates q USING(candidate_id) JOIN physics_ingest_snapshots s USING(snapshot_id) WHERE e.expression_id=%s FOR SHARE OF e,q,s',(expression_id,)).fetchone()
                    fresh=prepare_pdg_evidence(row,symbol_bytes)
                    if fresh!=row['evidence_json']['pdg_source_comparison']:raise ValueError('source comparison drift')
                    expressions.append(deserialize_expr(fresh['upstream_symbolic']['sympy_srepr']));comparison_hashes[expression_id]=_digest(fresh)
                mapping=prove_variable_renaming(*expressions,[parse_source_srepr(f['sympy']) for f in feeds],dimensions)
                original=signature['inference_rule_id'];signature['source_inference_rule_id']=original;signature['inference_rule_id']=replacement
                if contract_blockers(signature,rules[replacement]):raise ValueError('replacement rule contract failed')
                changes.append({'node_id':node['node_id'],'source_rule_id':original,'corrected_rule_id':replacement,'symbol_mapping':mapping,'symbol_dimensions':{name:dimensions[name] for name in set(mapping)|set(mapping.values())},'expression_comparison_sha256':comparison_hashes})
                node['type_signature']=json.dumps(signature,sort_keys=True);node['name']=rules[replacement].name
            if not changes:continue
            digest=_digest({'previous_content_hash':version['content_hash'],'corrected_nodes':[{k:v for k,v in n.items() if k!='version_id'} for n in nodes],'repair':'pdg-renaming-reconciliation.v1'})
            new_id=uuid5(UUID(str(version['artifact_id'])),'pdg-renaming-reconciliation.v1:'+digest)
            insert(db,'artifact_versions',{'version_id':new_id,'artifact_id':version['artifact_id'],'content_hash':digest,'semver':'0.0.0+pdg-rule-reconciled.'+digest[:12],'derives_from':version['version_id'],'is_latest':False,'trust_tier':3})
            for node in nodes:insert(db,'artifact_cdg_nodes',{**node,'version_id':new_id})
            for table,primary in [('artifact_cdg_edges','edge_id'),('artifact_cdg_bindings','binding_id'),('artifact_io_specs','io_spec_id')]:
                for row in db.execute(sql.SQL('SELECT * FROM {} WHERE version_id=%s FOR SHARE').format(sql.Identifier(table)),(version['version_id'],)).fetchall():
                    row.pop(primary,None);row['version_id']=new_id;insert(db,table,row)
            for row in db.execute('SELECT * FROM artifact_dependencies WHERE dependent_version_id=%s FOR SHARE',(version['version_id'],)).fetchall():
                row.pop('dependency_id');row['dependent_version_id']=new_id;insert(db,'artifact_dependencies',row)
            insert(db,'artifact_audit_evidence',{'artifact_id':version['artifact_id'],'version_id':new_id,'audit_type':'semantic_audit','passed':True,'status':'completed','source_kind':'automated','runner_version':'pdg-renaming-reconciliation.v1','details':{'scope':'source-level correction of mislabeled renaming only; stored normalization and whole-graph validation remain separate','previous_version_id':str(version['version_id']),'previous_content_hash':version['content_hash'],'changes':changes,'source_file_pins':pins[0],'proof_implementation_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'sciona/physics_ingest/pdg_rule_reconciliation.py').read_bytes()).hexdigest()}})
            db.execute('UPDATE artifact_versions SET is_latest=false WHERE version_id=%s',(version['version_id'],));db.execute('UPDATE artifact_versions SET is_latest=true WHERE version_id=%s',(new_id,))
            counts['corrected_versions']+=1;counts['reconciled_steps']+=len(changes)
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'counts':dict(counts)},indent=2))


if __name__=='__main__':main()
