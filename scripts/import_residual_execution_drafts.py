#!/usr/bin/env python3
"""Import exact shared provider identities and tested residual CDGs without replacing history."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from uuid import UUID,uuid5,NAMESPACE_URL
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms,AtomSeedRow
from sciona.architect.handoff import CDGExport
from sciona.physics_ingest.pdg_evidence import _digest
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg


def comparable(value):
    return json.loads(json.dumps(value,default=lambda obj:obj.obj if isinstance(obj,Jsonb) else str(obj)))


def ensure_row(db,table,identity,row):
    fields=list(row)
    predicate=sql.SQL(' AND ').join(sql.SQL('{}=%s').format(sql.Identifier(k)) for k in identity)
    existing=db.execute(sql.SQL('SELECT {} FROM {} WHERE {} FOR UPDATE').format(sql.SQL(',').join(map(sql.Identifier,fields)),sql.Identifier(table),predicate),tuple(identity.values())).fetchall()
    if existing:
        if comparable(existing)!=[comparable(row)]:raise ValueError('existing '+table+' record differs; refusing replacement')
        return False
    db.execute(sql.SQL('INSERT INTO {} ({}) VALUES ({})').format(sql.Identifier(table),sql.SQL(',').join(map(sql.Identifier,fields)),sql.SQL(',').join(sql.Placeholder() for _ in fields)),tuple(row.values()))
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];provider=root.parent/'sciona-atoms';counts=Counter()
    specs={p.fqdn:p for p in _parse_registered_atoms(repo=ProviderRepo('sciona-atoms',provider),artifact_root=provider/'src/sciona/atoms') if p.fqdn.startswith('sciona.atoms.electrical.residuals.')}
    if len(specs)!=4:raise ValueError('expected four shared provider implementations')
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('residual-execution-import.v1'))")
        evidence=db.execute("SELECT e.*,a.fqdn,v.content_hash AS source_hash FROM artifact_audit_evidence e JOIN artifacts a USING(artifact_id) JOIN artifact_versions v ON v.version_id=e.version_id WHERE e.runner_version='residual-cdg-execution.v1' AND e.passed ORDER BY e.evidence_id FOR SHARE OF e,a,v").fetchall()
        owners=db.execute("SELECT DISTINCT a.owner_id,a.source_repo_id FROM atoms a JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name='sciona-atoms'").fetchall()
        if len(owners)!=1:raise ValueError('ambiguous provider ownership')
        owner=owners[0]
        columns={r['column_name'] for r in db.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='artifacts'").fetchall()}
        for fqdn,spec in specs.items():
            related=[e for e in evidence if e['details']['execution_graph']['nodes'][0]['matched_primitive']==fqdn]
            if not related:raise ValueError('provider has no execution evidence')
            sha=hashlib.sha256(spec.file_path.read_bytes()).hexdigest()
            for record in related:
                if record['details']['provider_source_sha256']!=sha:raise ValueError('provider source drift')
            atom_id=uuid5(NAMESPACE_URL,'sciona-provider-draft:'+fqdn)
            fields={name:getattr(spec,name) for name in ['fqdn','namespace_root','namespace_path','repo_name','source_module_path','import_module','source_symbol','description','domain_tags','source_kind','is_ffi']}
            fields.update(source_package=spec.namespace_root,status='flagged',is_publishable=False)
            atom=AtomSeedRow(**fields).as_dict(owner_id=str(owner['owner_id']),source_repo_id=str(owner['source_repo_id']));atom['atom_id']=atom_id
            counts['provider_atoms_created']+=ensure_row(db,'atoms',{'atom_id':atom_id},atom)
            canonical={k:v for k,v in atom.items() if k in columns};canonical.update(artifact_id=atom_id,artifact_kind='atom',status='draft')
            ensure_row(db,'artifacts',{'artifact_id':atom_id},canonical)
            for table,owner_key in [('atom_versions','atom_id'),('artifact_versions','artifact_id')]:
                ensure_row(db,table,{'version_id':spec.version_id},{'version_id':spec.version_id,owner_key:atom_id,'content_hash':spec.content_hash,'semver':spec.semver,'is_latest':True,'s3_key':'','fingerprint':spec.fingerprint,'trust_tier':3})
            node=CDGExport.model_validate(related[0]['details']['execution_graph']).nodes[0]
            for direction,ports in [('input',node.inputs),('output',node.outputs)]:
                for ordinal,port in enumerate(ports):
                    common={'version_id':spec.version_id,'direction':direction,'name':port.name,'ordinal':ordinal,'type_desc':port.type_desc,'constraints':port.constraints,'required':port.required,'default_value_repr':port.default_value_repr}
                    for table,owner_key in [('atom_io_specs','atom_id'),('artifact_io_specs','artifact_id')]:
                        row={owner_key:atom_id,**common}
                        if table=='artifact_io_specs':row['dim_signature']=port.dim_signature
                        counts['provider_io_created']+=ensure_row(db,table,{k:row[k] for k in [owner_key,'version_id','direction','name']},row)
            audit_id=uuid5(UUID(str(spec.version_id)),'residual-provider-intake.v1')
            ensure_row(db,'artifact_audit_evidence',{'evidence_id':audit_id},{'evidence_id':audit_id,'artifact_id':atom_id,'version_id':spec.version_id,'audit_type':'asset_integrity_check','passed':True,'status':'completed','source_kind':'automated','runner_version':'residual-provider-intake.v1','details':Jsonb({'scope':'local provider identity, interfaces and source integrity; draft pending publication review and distribution checks','source_sha256':sha,'content_hash':spec.content_hash,'execution_evidence_ids':[str(r['evidence_id']) for r in related]})})
        for record in evidence:
            report=record['details']
            if report['source_content_hash']!=record['source_hash']:raise ValueError('source version drift')
            for path,expected in report['implementation_hashes'].items():
                if hashlib.sha256((root/path).read_bytes()).hexdigest()!=expected:raise ValueError('execution implementation drift')
            graph=CDGExport.model_validate(report['execution_graph']);digest,nodes,edges=encode_execution_graph(graph)
            if digest!=report['execution_graph_sha256'] or len(nodes)!=1 or edges:raise ValueError('execution graph drift')
            spec=specs[graph.nodes[0].matched_primitive]
            artifact_id=uuid5(UUID(str(record['artifact_id'])),'residual-execution.v1');version_id=uuid5(artifact_id,digest);fqdn=record['fqdn']+'.residual_execution'
            counts['execution_cdgs_created']+=ensure_row(db,'artifacts',{'artifact_id':artifact_id},{'artifact_id':artifact_id,'artifact_kind':'cdg','fqdn':fqdn,'description':'Evaluate a source equation residual through a shared dimensioned implementation; no unknown is solved.','status':'draft','is_publishable':False,'verified_leaf_coverage':0.0})
            ensure_row(db,'artifact_versions',{'version_id':version_id},{'version_id':version_id,'artifact_id':artifact_id,'content_hash':digest,'semver':'0.0.0+execution.'+digest[:12],'is_latest':False,'trust_tier':3})
            for node in nodes:
                row={'version_id':version_id,**{k:node[k] for k in ['node_id','parent_node_id','name','description','concept_type','status','matched_primitive','type_signature']}}
                ensure_row(db,'artifact_cdg_nodes',{'version_id':version_id,'node_id':node['node_id']},row)
            for direction,ports in [('input',graph.nodes[0].inputs),('output',graph.nodes[0].outputs)]:
                for ordinal,port in enumerate(ports):
                    row={'artifact_id':artifact_id,'version_id':version_id,'direction':direction,'name':port.name,'ordinal':ordinal,'type_desc':port.type_desc,'constraints':port.constraints,'required':port.required,'dim_signature':port.dim_signature}
                    ensure_row(db,'artifact_io_specs',{k:row[k] for k in ['artifact_id','version_id','direction','name']},row)
            ensure_row(db,'artifact_cdg_bindings',{'version_id':version_id,'node_id':'residual'},{'version_id':version_id,'node_id':'residual','bound_artifact_fqdn':spec.fqdn,'bound_version_content_hash':spec.content_hash,'binding_confidence':1.0,'binding_source':'residual-execution-import.v1','status':'active','evidence_summary':Jsonb({'scope':'exact tested implementation binding; target pending Community publication','provider_version_id':str(spec.version_id)})})
            ensure_row(db,'artifact_dependencies',{'dependent_version_id':version_id,'dependency_artifact_fqdn':record['fqdn'],'dependency_content_hash':record['source_hash'],'port_name':''},{'dependent_version_id':version_id,'dependency_artifact_fqdn':record['fqdn'],'dependency_content_hash':record['source_hash'],'port_name':'','dependency_role':'logic_atom','optional':False,'binding_metadata':Jsonb({'scope':'source relation provenance and validity dependency','source_version_id':str(record['version_id'])})})
            audit_id=uuid5(version_id,'residual-execution-import.v1')
            ensure_row(db,'artifact_audit_evidence',{'evidence_id':audit_id},{'evidence_id':audit_id,'artifact_id':artifact_id,'version_id':version_id,'audit_type':'asset_integrity_check','passed':True,'status':'completed','source_kind':'automated','runner_version':'residual-execution-import.v1','details':Jsonb({'scope':'lossless tested draft import; not publication approval','source_execution_evidence_id':str(record['evidence_id']),'source_execution_evidence_sha256':_digest(report),'provider_content_hash':spec.content_hash})})
            document=db.execute('SELECT get_artifact_document(%s) AS document',(fqdn,)).fetchone()['document']
            if _artifact_document_to_cdg(document,version_id=str(version_id),content_hash=digest,require_execution_envelope=True)!=graph:raise ValueError('catalog materialization differs from tested graph')
            counts['execution_cdgs_verified']+=1
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'counts':dict(counts)},indent=2))


if __name__=='__main__':main()
