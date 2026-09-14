#!/usr/bin/env python3
"""Version-preserving correction of source-proven repeated PDG premise slots."""
import argparse
from collections import defaultdict,Counter
import hashlib
import json
import os
from pathlib import Path
from uuid import UUID,uuid5
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sciona.physics_ingest.pdg_cdg import _step_input_node_ids
from sciona.physics_ingest.sources.pdg import PDGInferenceEdge
from sciona.physics_ingest.pdg_evidence import _digest


def insert(db,table,row):
    db.execute(sql.SQL('INSERT INTO {} ({}) VALUES ({})').format(sql.Identifier(table),sql.SQL(',').join(map(sql.Identifier,row)),sql.SQL(',').join(sql.Placeholder() for _ in row)),tuple(Jsonb(v) if isinstance(v,dict) or (table=='artifact_cdg_bindings' and k=='alternatives') else v for k,v in row.items()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');args=parser.parse_args();counts=Counter()
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('pdg-input-slots.v1'))")
        snapshots=db.execute("SELECT snapshot_id,payload FROM physics_ingest_snapshots WHERE payload ? 'core_file_sha256' FOR SHARE").fetchall()
        source_groups=defaultdict(list)
        for snapshot in snapshots:
            groups=defaultdict(list)
            for edge in snapshot['payload']['inference_edges']:
                b=edge['bindings'];groups[(b['derivation_id'],b['step_id'])].append(edge)
            for key,edges in groups.items():
                if not any(entry['edges']==edges for entry in source_groups[key]):source_groups[key].append({'snapshot_id':str(snapshot['snapshot_id']),'edges':edges})
        expressions={str(r['expression_id']):r['source_id'] for r in db.execute("SELECT e.expression_id,q.source_payload->>'id' AS source_id FROM artifact_symbolic_expressions e JOIN physics_equation_candidates q USING(candidate_id) FOR SHARE OF e,q").fetchall()}
        versions=db.execute("SELECT v.* FROM artifact_versions v JOIN artifacts a USING(artifact_id) WHERE a.fqdn LIKE 'physics.pdg.%' AND a.status='draft' AND NOT a.is_publishable AND v.is_latest FOR UPDATE OF v FOR SHARE OF a").fetchall()
        for version in versions:
            nodes=db.execute('SELECT * FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id FOR SHARE',(version['version_id'],)).fetchall()
            changes=[]
            for node in nodes:
                signature=json.loads(node['type_signature'])
                if not signature.get('source_pdg_step_id'):continue
                bindings=signature['variable_bindings'];matches=source_groups[(bindings['derivation_id'],bindings['step_id'])]
                if len(matches)!=1:raise ValueError('source step must resolve unambiguously')
                source=matches[0]
                if [e['id'] for e in source['edges']]!=signature['source_pdg_inference_ids']:raise ValueError('stored source edge projection differs')
                slots=_step_input_node_ids([PDGInferenceEdge.from_payload(e) for e in source['edges']])
                current=signature['inputs']
                if [expressions.get(e) for e in current]==slots:continue
                mapping={expressions.get(e):e for e in current}
                if None in mapping or list(mapping)!=list(dict.fromkeys(slots)):raise ValueError('change is not solely repeated input multiplicity')
                corrected=[mapping[source_id] for source_id in slots]
                changes.append({'node_id':node['node_id'],'previous_inputs':current,'corrected_inputs':corrected,'snapshot_id':source['snapshot_id'],'source_edges_sha256':_digest(source['edges'])})
                signature['inputs']=corrected;node['type_signature']=json.dumps(signature,sort_keys=True)
            if not changes:continue
            projected=[{k:v for k,v in node.items() if k!='version_id'} for node in nodes]
            digest=_digest({'source_content_hash':version['content_hash'],'corrected_nodes':projected,'repair':'pdg-input-slots.v1'})
            new_id=uuid5(UUID(str(version['artifact_id'])),'pdg-input-slots.v1:'+digest)
            if db.execute('SELECT 1 FROM artifact_versions WHERE version_id=%s',(new_id,)).fetchone():raise ValueError('existing correction requires latest-version reconciliation')
            insert(db,'artifact_versions',{'version_id':new_id,'artifact_id':version['artifact_id'],'content_hash':digest,'semver':'0.0.0+pdg-input-slots.'+digest[:12],'derives_from':version['version_id'],'is_latest':False,'trust_tier':3})
            for node in nodes:insert(db,'artifact_cdg_nodes',{**node,'version_id':new_id})
            for table,primary_key in [('artifact_cdg_edges','edge_id'),('artifact_cdg_bindings','binding_id'),('artifact_io_specs','io_spec_id')]:
                for row in db.execute(sql.SQL('SELECT * FROM {} WHERE version_id=%s FOR SHARE').format(sql.Identifier(table)),(version['version_id'],)).fetchall():
                    row.pop(primary_key,None);row['version_id']=new_id;insert(db,table,row)
            for row in db.execute('SELECT * FROM artifact_dependencies WHERE dependent_version_id=%s FOR SHARE',(version['version_id'],)).fetchall():
                row.pop('dependency_id');row['dependent_version_id']=new_id;insert(db,'artifact_dependencies',row)
            insert(db,'artifact_audit_evidence',{'artifact_id':version['artifact_id'],'version_id':new_id,'audit_type':'asset_integrity_check','passed':True,'status':'completed','source_kind':'automated','runner_version':'pdg-input-slots.v1','details':{'scope':'restore source-proven repeated input slots only; no semantic validation or publication approval','previous_version_id':str(version['version_id']),'previous_content_hash':version['content_hash'],'corrected_content_hash':digest,'changes':changes,'implementation_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'sciona/physics_ingest/pdg_cdg.py').read_bytes()).hexdigest()}})
            db.execute('UPDATE artifact_versions SET is_latest=false WHERE version_id=%s',(version['version_id'],))
            db.execute('UPDATE artifact_versions SET is_latest=true WHERE version_id=%s',(new_id,))
            counts['corrected_versions']+=1;counts['restored_steps']+=len(changes)
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'counts':dict(counts)},indent=2))


if __name__=='__main__':main()
