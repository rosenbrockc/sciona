#!/usr/bin/env python3
"""Store a lossless numerical execution draft with a mandatory source-CDG pin."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
from uuid import UUID,uuid5
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sciona.atoms.electrical.series_resistance import series_resistance
from sciona.architect.handoff import CDGExport
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.physics_ingest.pdg_evidence import _digest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    counts={'created':0,'unchanged':0}
    root=Path(__file__).resolve().parents[1]
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        sources=db.execute("SELECT e.*,a.fqdn,v.content_hash AS source_hash FROM artifact_audit_evidence e JOIN artifact_versions v USING(version_id) JOIN artifacts a ON a.artifact_id=v.artifact_id WHERE e.runner_version='series-cdg-execution.v1' AND e.passed AND v.is_latest AND a.status='draft' FOR SHARE OF e,v,a").fetchall()
        for record in sources:
            report=record['details']
            for name,digest in report['implementation_hashes'].items():
                if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('execution evidence implementation is stale')
            if hashlib.sha256(Path(inspect.getfile(series_resistance)).read_bytes()).hexdigest()!=report['provider_implementation_sha256']:raise ValueError('provider implementation changed')
            graph=CDGExport.model_validate(report['execution_cdg'])
            if graph.metadata.get('source_derivation_version_id')!=str(record['version_id']) or not graph.metadata.get('requires_source_dependency_review'):raise ValueError('source review provenance missing')
            digest,nodes,edges=encode_execution_graph(graph)
            if len(nodes)!=1 or edges:raise ValueError('this importer requires the validated single-node realization')
            artifact_id=uuid5(UUID(str(record['artifact_id'])),'numerical-execution.v1')
            version_id=uuid5(artifact_id,digest)
            fqdn=record['fqdn']+'.execution'
            existing=db.execute('SELECT * FROM artifact_versions WHERE version_id=%s FOR UPDATE',(version_id,)).fetchone()
            if not existing:
                if db.execute('SELECT 1 FROM artifacts WHERE artifact_id=%s OR fqdn=%s',(artifact_id,fqdn)).fetchone():raise ValueError('existing execution artifact requires explicit version reconciliation')
                db.execute("INSERT INTO artifacts(artifact_id,artifact_kind,fqdn,description,status,is_publishable,verified_leaf_coverage) VALUES(%s,'cdg',%s,%s,'draft',false,0)",(artifact_id,fqdn,'Numerical realization of a source derivation; provider publication and source dependency review remain required.'))
                db.execute("INSERT INTO artifact_versions(version_id,artifact_id,semver,content_hash,is_latest) VALUES(%s,%s,%s,%s,false)",(version_id,artifact_id,'0.0.0+execution.'+digest[:12],digest))
                for node in nodes:
                    keys=['node_id','parent_node_id','name','description','concept_type','status','matched_primitive','type_signature']
                    db.execute('INSERT INTO artifact_cdg_nodes(version_id,node_id,parent_node_id,name,description,concept_type,status,matched_primitive,type_signature) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)',(version_id,*(node[k] for k in keys)))
                for direction,ports in [('input',graph.nodes[0].inputs),('output',graph.nodes[0].outputs)]:
                    for ordinal,port in enumerate(ports):
                        db.execute('INSERT INTO artifact_io_specs(artifact_id,version_id,direction,name,type_desc,constraints,required,ordinal,dim_signature) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)',(artifact_id,version_id,direction,port.name,port.type_desc,port.constraints,port.required,ordinal,port.dim_signature))
                db.execute("INSERT INTO artifact_dependencies(dependent_version_id,dependency_artifact_fqdn,dependency_content_hash,dependency_role,optional,binding_metadata) VALUES(%s,%s,%s,'cdg',false,%s)",(version_id,record['fqdn'],record['source_hash'],Jsonb({'source_version_id':str(record['version_id']),'scope':'mandatory derivation provenance and review dependency'})))
                db.execute("INSERT INTO artifact_audit_evidence(artifact_id,version_id,audit_type,passed,status,details,source_kind,runner_version) VALUES(%s,%s,'asset_integrity_check',true,'completed',%s,'automated','execution-draft-import.v1')",(artifact_id,version_id,Jsonb({'scope':'lossless draft import only; no publication approval','source_execution_evidence_id':str(record['evidence_id']),'source_execution_evidence_sha256':_digest(report),'snapshot_hash':digest,'publication_blockers':['provider_not_catalog_published','source_derivation_review_pending','execution_version_review_pending']})))
            document=db.execute('SELECT get_artifact_document(%s) AS document',(fqdn,)).fetchone()['document']
            restored=_artifact_document_to_cdg(document,version_id=str(version_id),content_hash=digest,require_execution_envelope=True)
            if restored!=graph:raise ValueError('canonical materialization is not lossless')
            expected_io=[]
            for direction,ports in [('input',graph.nodes[0].inputs),('output',graph.nodes[0].outputs)]:
                for ordinal,port in enumerate(ports):
                    expected_io.append({'direction':direction,'name':port.name,'type_desc':port.type_desc,'constraints':port.constraints,'required':port.required,'ordinal':ordinal,'dim_signature':port.dim_signature})
            actual_io=db.execute('SELECT direction,name,type_desc,constraints,required,ordinal,dim_signature FROM artifact_io_specs WHERE version_id=%s ORDER BY direction,ordinal',(version_id,)).fetchall()
            if actual_io!=expected_io:raise ValueError('canonical boundary IO drift')
            dependency=db.execute('SELECT dependency_artifact_fqdn,dependency_content_hash,optional FROM artifact_dependencies WHERE dependent_version_id=%s',(version_id,)).fetchall()
            if dependency!=[{'dependency_artifact_fqdn':record['fqdn'],'dependency_content_hash':record['source_hash'],'optional':False}]:raise ValueError('mandatory source dependency drift')
            counts['unchanged' if existing else 'created']+=1
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'counts':counts},indent=2))


if __name__=='__main__':main()
