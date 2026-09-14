#!/usr/bin/env python3
"""Create a non-latest graph alternative resolving an internally inconsistent pin."""
import argparse
import hashlib
import json
from pathlib import Path
from uuid import UUID,uuid5
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from dotenv import dotenv_values
from scripts.import_residual_execution_drafts import ensure_row,comparable
from sciona.physics_ingest.pdg_evidence import prepare_pdg_evidence,_digest

GRAPH=UUID('ffb043f3-05a3-5ca6-9d94-87a17ff512d3')
EXPRESSION=UUID('05e12134-25f0-57cd-8e7f-f80aa01888fd')
MISSING='8d7dec2c3f13e7dcc89abc31b3d2e1539fff56acd55b0d78dfd7b36f7feca4cf'
FQDN='physics.pdg.remote_wave.equation.1292735067'


def repair(root,symbol_file,apply=False):
    created=0
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('pdg-explicit-expression-pin.v1'))")
        graph=db.execute('SELECT v.*,a.status,a.is_publishable FROM artifact_versions v JOIN artifacts a USING(artifact_id) WHERE version_id=%s FOR SHARE OF a,v',(GRAPH,)).fetchone()
        if not graph or graph['status']!='draft' or graph['is_publishable']:raise ValueError('Original graph must remain draft')
        if db.execute('SELECT 1 FROM artifact_versions WHERE content_hash=%s',(MISSING,)).fetchone():raise ValueError('Missing pin now exists; reassess repair')
        nodes=db.execute('SELECT * FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id FOR SHARE',(GRAPH,)).fetchall()
        bindings=db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s ORDER BY binding_id FOR SHARE',(GRAPH,)).fetchall()
        target=[b for b in bindings if b['bound_artifact_fqdn']==FQDN]
        if len(target)!=1 or target[0]['bound_version_content_hash']!=MISSING or target[0]['status']!='active':raise ValueError('Expected inconsistent binding differs')
        node=next(n for n in nodes if n['node_id']==target[0]['node_id'])
        signature=json.loads(node['type_signature'])
        if str(EXPRESSION) not in signature['inputs']:raise ValueError('Graph does not explicitly reference replacement expression')
        row=db.execute("SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload,a.fqdn,v.content_hash FROM artifact_symbolic_expressions e JOIN physics_equation_candidates q USING(candidate_id) JOIN physics_ingest_snapshots s USING(snapshot_id) JOIN artifact_versions v ON v.version_id=e.version_id JOIN artifacts a ON a.artifact_id=e.artifact_id WHERE e.expression_id=%s FOR SHARE OF e,q,s,v,a",(EXPRESSION,)).fetchone()
        if not row or row['fqdn']!=FQDN or row['source_expression_id']!='1292735067' or row['source_payload']['id']!='1292735067':raise ValueError('Referenced source identity differs')
        fresh=prepare_pdg_evidence(row,symbol_file.read_bytes())
        if fresh!=row['evidence_json']['pdg_source_comparison']:raise ValueError('Referenced source evidence changed')
        if fresh['upstream_symbolic']['status']!='roundtrip_passed':raise ValueError('Source syntax missing')
        details=dict(kind='pdg-explicit-expression-pin.v1',scope='Internal binding/reference consistency only; no expression equivalence, proof or publication approval.',
            previous_graph_version_id=str(GRAPH),previous_graph_content_hash=graph['content_hash'],node_id=node['node_id'],
            referenced_expression_id=str(EXPRESSION),replacement_version_id=str(row['version_id']),
            missing_content_hash=MISSING,replacement_content_hash=row['content_hash'],
            replacement_source_evidence_sha256=_digest(fresh),original_binding_sha256=_digest(comparable(target[0])),
            source_correspondence=fresh['source_comparison']['correspondence'],
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        digest=_digest(details);new_id=uuid5(graph['artifact_id'],'pdg-explicit-expression-pin.v1:'+digest)
        created+=ensure_row(db,'artifact_versions',{'version_id':new_id},dict(version_id=new_id,artifact_id=graph['artifact_id'],content_hash=digest,semver='0.0.0+pin-repair.'+digest[:12],derives_from=GRAPH,is_latest=False,trust_tier=3))
        for n in nodes:
            n['version_id']=new_id
            created+=ensure_row(db,'artifact_cdg_nodes',{'version_id':new_id,'node_id':n['node_id']},n)
        for edge in db.execute('SELECT * FROM artifact_cdg_edges WHERE version_id=%s FOR SHARE',(GRAPH,)).fetchall():
            edge.pop('edge_id',None);edge['version_id']=new_id
            created+=ensure_row(db,'artifact_cdg_edges',{k:edge[k] for k in ['version_id','source_id','target_id','input_name','output_name']},edge)
        for binding in bindings:
            old=binding['binding_id'];binding['binding_id']=uuid5(new_id,str(old));binding['version_id']=new_id
            if binding['bound_artifact_fqdn']==FQDN:
                binding['bound_version_content_hash']=row['content_hash'];binding['binding_source']='pdg-explicit-expression-pin.v1'
                binding['evidence_summary']={'scope':details['scope'],'referenced_expression_id':str(EXPRESSION),'replacement_version_id':str(row['version_id'])}
            binding['evidence_summary']=Jsonb(binding['evidence_summary']);binding['alternatives']=Jsonb(binding['alternatives'])
            created+=ensure_row(db,'artifact_cdg_bindings',{'binding_id':binding['binding_id']},binding)
        for port in db.execute('SELECT * FROM artifact_io_specs WHERE version_id=%s FOR SHARE',(GRAPH,)).fetchall():
            port['io_spec_id']=uuid5(new_id,str(port['io_spec_id']));port['version_id']=new_id
            created+=ensure_row(db,'artifact_io_specs',{'io_spec_id':port['io_spec_id']},port)
        for dep in db.execute('SELECT * FROM artifact_dependencies WHERE dependent_version_id=%s FOR SHARE',(GRAPH,)).fetchall():
            dep['dependency_id']=uuid5(new_id,str(dep['dependency_id']));dep['dependent_version_id']=new_id
            if dep['dependency_artifact_fqdn']==FQDN:
                if dep['dependency_content_hash']!=MISSING:raise ValueError('Unexpected independent dependency pin')
                dep['dependency_content_hash']=row['content_hash']
            dep['binding_metadata']=Jsonb(dep['binding_metadata'])
            created+=ensure_row(db,'artifact_dependencies',{'dependency_id':dep['dependency_id']},dep)
        evidence_id=uuid5(new_id,'pin-repair-integrity')
        created+=ensure_row(db,'artifact_audit_evidence',{'evidence_id':evidence_id},dict(evidence_id=evidence_id,artifact_id=graph['artifact_id'],version_id=new_id,audit_type='asset_integrity_check',passed=True,status='completed',source_kind='automated',runner_version='pdg-explicit-expression-pin.v1',details=Jsonb(details)))
        if db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s',(graph['artifact_id'],)).fetchone():raise ValueError('Unreviewed graph is served')
        if not apply:db.rollback()
    return dict(applied=apply,rows_created=created,alternative_version_id=str(new_id),original_preserved=True,publication='nonlatest_draft')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--symbol-file',type=Path,required=True);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    print(json.dumps(repair(Path(__file__).resolve().parents[1],args.symbol_file,args.apply)))
