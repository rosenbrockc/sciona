#!/usr/bin/env python3
"""Import exact causal graph providers and bind versions without granting approval."""
import argparse
from collections import Counter
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
from uuid import UUID,uuid5,NAMESPACE_URL
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms,AtomSeedRow
from sciona.architect.handoff import CDGExport
from sciona.physics_ingest.pdg_evidence import _digest
from scripts.import_residual_execution_drafts import ensure_row


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];specs={};counts=Counter()
    for repo in ['sciona-atoms','sciona-atoms-ml']:
        directory=root.parent/repo
        for spec in _parse_registered_atoms(repo=ProviderRepo(repo,directory),artifact_root=directory/'src/sciona/atoms'):
            specs.setdefault((spec.import_module,spec.source_symbol),[]).append(spec)
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('causal-provider-binding.v1'))")
        intakes=db.execute("SELECT * FROM artifact_audit_evidence WHERE runner_version='causal-execution-intake.v1' FOR SHARE").fetchall()
        if len(intakes)!=1:raise ValueError('one causal execution intake required')
        intake=intakes[0]
        executed=db.execute("SELECT * FROM artifact_audit_evidence WHERE evidence_id=%s AND passed FOR SHARE",(intake['details']['source_execution_evidence_id'],)).fetchone()
        if not executed or _digest(executed['details'])!=intake['details']['source_execution_evidence_sha256']:raise ValueError('execution evidence mismatch')
        details=executed['details'];graph=CDGExport.model_validate(details['execution_graph'])
        columns={r['column_name'] for r in db.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='artifacts'").fetchall()}
        for node in graph.nodes:
            module,symbol=node.matched_primitive.rsplit('.',1)
            matches=specs.get((module,symbol),[])
            if len(matches)!=1:raise ValueError('unique parsed provider identity required')
            spec=matches[0];fn=getattr(importlib.import_module(module),symbol)
            if fn.__module__+'.'+fn.__name__!=node.matched_primitive:raise ValueError('runtime callable identity differs')
            sha=hashlib.sha256(spec.file_path.read_bytes()).hexdigest()
            if details['provider_hashes'].get(module)!=sha:raise ValueError('validated source hash differs')
            old=db.execute('SELECT * FROM atoms WHERE fqdn=%s FOR UPDATE',(spec.fqdn,)).fetchone()
            if old:
                atom_id=old['atom_id'];counts['existing_provider_identities']+=1
                if old['import_module']!=module or old['source_symbol']!=symbol:raise ValueError('existing import metadata differs')
                version=db.execute('SELECT * FROM atom_versions WHERE atom_id=%s AND is_latest FOR SHARE',(atom_id,)).fetchone()
                if not version or str(version['version_id'])!=str(spec.version_id) or version['content_hash']!=spec.content_hash:raise ValueError('existing served version differs; separate version review required')
            else:
                owner=db.execute("SELECT DISTINCT a.owner_id,a.source_repo_id FROM atoms a JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name=%s",(spec.repo_name,)).fetchall()
                if len(owner)!=1:raise ValueError('ambiguous provider ownership')
                atom_id=uuid5(NAMESPACE_URL,'sciona-provider-draft:'+spec.fqdn)
                fields={name:getattr(spec,name) for name in ['fqdn','namespace_root','namespace_path','repo_name','source_module_path','import_module','source_symbol','description','domain_tags','source_kind','is_ffi']}
                fields.update(source_package=spec.namespace_root,status='flagged',is_publishable=False)
                old=AtomSeedRow(**fields).as_dict(owner_id=str(owner[0]['owner_id']),source_repo_id=str(owner[0]['source_repo_id']));old['atom_id']=atom_id
                ensure_row(db,'atoms',{'atom_id':atom_id},old);counts['new_provider_identities']+=1
                ensure_row(db,'atom_versions',{'version_id':spec.version_id},{'version_id':spec.version_id,'atom_id':atom_id,'content_hash':spec.content_hash,'semver':spec.semver,'is_latest':True,'trust_tier':3,'s3_key':'','fingerprint':spec.fingerprint})
            canonical={k:v for k,v in old.items() if k in columns and k not in {'created_at','updated_at'}}
            canonical.update(artifact_id=atom_id,artifact_kind='atom',status='draft',is_publishable=False)
            counts['canonical_identities_created']+=ensure_row(db,'artifacts',{'artifact_id':atom_id},canonical)
            counts['canonical_versions_created']+=ensure_row(db,'artifact_versions',{'version_id':spec.version_id},{'version_id':spec.version_id,'artifact_id':atom_id,'content_hash':spec.content_hash,'semver':spec.semver,'is_latest':True,'trust_tier':3,'s3_key':'','fingerprint':spec.fingerprint})
            if set(p.name for p in node.inputs)-set(inspect.signature(fn).parameters):raise ValueError('graph inputs differ from callable')
            # Preserve legacy interfaces; record canonical ports of the tested graph contract.
            for direction,ports in [('input',node.inputs),('output',node.outputs)]:
                for ordinal,port in enumerate(ports):
                    row={'artifact_id':atom_id,'version_id':spec.version_id,'direction':direction,'name':port.name,'ordinal':ordinal,'type_desc':port.type_desc,'constraints':port.constraints,'required':port.required,'default_value_repr':port.default_value_repr}
                    counts['canonical_io_created']+=ensure_row(db,'artifact_io_specs',{k:row[k] for k in ['artifact_id','version_id','direction','name']},row)
                    if old['status']=='flagged':
                        legacy={('atom_id' if k=='artifact_id' else k):v for k,v in row.items()}
                        ensure_row(db,'atom_io_specs',{k:legacy[k] for k in ['atom_id','version_id','direction','name']},legacy)
            audit_id=uuid5(UUID(str(spec.version_id)),'causal-provider-intake.v1')
            ensure_row(db,'artifact_audit_evidence',{'evidence_id':audit_id},{'evidence_id':audit_id,'artifact_id':atom_id,'version_id':spec.version_id,'audit_type':'asset_integrity_check','passed':True,'status':'completed','source_kind':'automated','runner_version':'causal-provider-intake.v1','details':Jsonb({'source_sha256':sha,'runtime_fqdn':node.matched_primitive,'catalog_fqdn':spec.fqdn,'source_execution_evidence_id':str(executed['evidence_id']),'source_execution_evidence_sha256':_digest(details),'scope':'Exact installed provider identity and tested graph-port intake; independent publication approval pending.'})})
            binding={'version_id':intake['version_id'],'node_id':node.node_id,'bound_artifact_fqdn':spec.fqdn,'bound_version_content_hash':spec.content_hash,'binding_confidence':1.,'binding_source':'causal-provider-binding.v1','status':'active','alternatives':Jsonb([]),'evidence_summary':Jsonb({'runtime_fqdn':node.matched_primitive,'provider_version_id':str(spec.version_id),'provider_source_sha256':sha,'identity_basis':'parsed catalog import_module and source_symbol resolve to the identical runtime callable'})}
            counts['bindings_created']+=ensure_row(db,'artifact_cdg_bindings',{'version_id':intake['version_id'],'node_id':node.node_id},binding)
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'counts':dict(counts)},indent=2))


if __name__=='__main__':main()
