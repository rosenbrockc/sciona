#!/usr/bin/env python3
"""Preserve validated causal execution separately from the conceptual intake."""
import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
from uuid import uuid5
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sciona.architect.handoff import CDGExport
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.physics_ingest.pdg_evidence import _digest
from scripts.import_residual_execution_drafts import ensure_row


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('causal-execution-intake.v1'))")
        records=db.execute("SELECT e.*,a.fqdn,v.content_hash AS source_hash FROM artifact_audit_evidence e JOIN artifacts a USING(artifact_id) JOIN artifact_versions v USING(version_id) WHERE e.runner_version='causal-full-execution.v1' AND e.passed FOR SHARE OF e,a,v").fetchall()
        if len(records)!=1:raise ValueError('unique validated causal execution required')
        record=records[0];details=record['details']
        for path,digest in details['implementation_hashes'].items():
            if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest:raise ValueError('execution implementation changed')
        for module,digest in details['provider_hashes'].items():
            if hashlib.sha256(Path(inspect.getfile(importlib.import_module(module))).read_bytes()).hexdigest()!=digest:raise ValueError('provider implementation changed')
        source=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='competition-intake.v1' FOR SHARE",(record['version_id'],)).fetchone()
        if not source or _digest(source['details']['snapshot'])!=details['source_intake_sha256']:raise ValueError('source intake changed')
        graph=CDGExport.model_validate(details['execution_graph'])
        digest,nodes,edges=encode_execution_graph(graph)
        if digest!=details['execution_graph_sha256']:raise ValueError('validated graph hash differs')
        artifact_id=uuid5(record['artifact_id'],'causal-execution.v1');version_id=uuid5(artifact_id,digest);fqdn=record['fqdn']+'.execution'
        created=ensure_row(db,'artifacts',{'artifact_id':artifact_id},{'artifact_id':artifact_id,'artifact_kind':'cdg','fqdn':fqdn,'status':'draft','is_publishable':False,'description':'Train and execute a three-system causal direction estimator from typed observation pairs. Reconstructed source-compatible feature and model pipeline; synthetic same-library parity established, not competition performance.','verified_leaf_coverage':0.,'leaf_count':len(nodes)})
        ensure_row(db,'artifact_versions',{'version_id':version_id},{'version_id':version_id,'artifact_id':artifact_id,'content_hash':digest,'semver':'0.0.0+execution.'+digest[:12],'is_latest':False,'trust_tier':3})
        for table,rows,keys in [('artifact_cdg_nodes',nodes,['node_id']),('artifact_cdg_edges',edges,['source_id','target_id','output_name','input_name'])]:
            for item in rows:
                if table=='artifact_cdg_nodes':
                    row={k:item[k] for k in ['node_id','parent_node_id','name','description','concept_type','status','matched_primitive','type_signature']}
                else:
                    row={k:item[k] for k in ['source_id','target_id','output_name','input_name']}
                row['version_id']=version_id
                ensure_row(db,table,{k:row[k] for k in ['version_id',*keys]},row)
        consumed={(e.target_id,e.input_name) for e in graph.edges}
        producers={e.source_id for e in graph.edges}
        for direction,ports in [('input',[p for n in graph.nodes for p in n.inputs if (n.node_id,p.name) not in consumed]),('output',[p for n in graph.nodes if n.node_id not in producers for p in n.outputs])]:
            if len({p.name for p in ports})!=len(ports):raise ValueError('ambiguous graph boundary')
            for ordinal,port in enumerate(ports):
                row={'artifact_id':artifact_id,'version_id':version_id,'direction':direction,'name':port.name,'ordinal':ordinal,'type_desc':port.type_desc,'constraints':port.constraints,'required':port.required,'default_value_repr':port.default_value_repr}
                ensure_row(db,'artifact_io_specs',{k:row[k] for k in ['artifact_id','version_id','direction','name']},row)
        ensure_row(db,'artifact_dependencies',{'dependent_version_id':version_id,'dependency_artifact_fqdn':record['fqdn'],'dependency_content_hash':record['source_hash'],'port_name':''},{'dependent_version_id':version_id,'dependency_artifact_fqdn':record['fqdn'],'dependency_content_hash':record['source_hash'],'port_name':'','dependency_role':'cdg','optional':False,'binding_metadata':Jsonb({'scope':'mandatory conceptual source provenance, not an invoked numerical dependency'})})
        audit_id=uuid5(version_id,'causal-execution-intake.v1')
        ensure_row(db,'artifact_audit_evidence',{'evidence_id':audit_id},{'evidence_id':audit_id,'artifact_id':artifact_id,'version_id':version_id,'audit_type':'asset_integrity_check','passed':True,'status':'completed','source_kind':'automated','runner_version':'causal-execution-intake.v1','details':Jsonb({'source_execution_evidence_id':str(record['evidence_id']),'source_execution_evidence_sha256':_digest(details),'scope':'Lossless validated graph intake; provider version bindings and publication review remain pending.'})})
        document=db.execute('SELECT get_artifact_document(%s) AS d',(fqdn,)).fetchone()['d']
        if _artifact_document_to_cdg(document,version_id=str(version_id),content_hash=digest,require_execution_envelope=True)!=graph:raise ValueError('catalog graph round-trip differs')
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'created':created,'nodes':len(nodes),'edges':len(edges),'catalog_roundtrip':'passed'},indent=2))


if __name__=='__main__':main()
