#!/usr/bin/env python3
"""Targeted, history-preserving provider draft import and exact execution binding."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid5,NAMESPACE_URL
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms,AtomSeedRow
from sciona.physics_ingest.series_execution import PRIMITIVE


def insert(db,table,row):
    db.execute(sql.SQL('INSERT INTO {} ({}) VALUES ({})').format(sql.Identifier(table),sql.SQL(',').join(map(sql.Identifier,row)),sql.SQL(',').join(sql.Placeholder() for _ in row)),tuple(row.values()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    provider=Path(__file__).resolve().parents[2]/'sciona-atoms'
    parsed=[p for p in _parse_registered_atoms(repo=ProviderRepo('sciona-atoms',provider),artifact_root=provider/'src/sciona/atoms') if p.fqdn==PRIMITIVE]
    if len(parsed)!=1:raise ValueError('provider identity must resolve exactly once')
    spec=parsed[0];atom_id=uuid5(NAMESPACE_URL,'sciona-provider-draft:'+spec.fqdn)
    source_sha=hashlib.sha256(spec.file_path.read_bytes()).hexdigest()
    counts={'created_atoms':0,'created_bindings':0}
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(spec.fqdn,))
        owners=db.execute("SELECT DISTINCT a.owner_id,a.source_repo_id FROM atoms a JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name='sciona-atoms'").fetchall()
        if len(owners)!=1:raise ValueError('provider owner/repository association is ambiguous')
        owner=owners[0]
        fields={name:getattr(spec,name) for name in ['fqdn','namespace_root','namespace_path','repo_name','source_module_path','import_module','source_symbol','description','domain_tags','source_kind','is_ffi']}
        fields.update(source_package=spec.namespace_root,status='flagged',is_publishable=False)
        atom=AtomSeedRow(**fields).as_dict(owner_id=str(owner['owner_id']),source_repo_id=str(owner['source_repo_id']))
        atom['atom_id']=atom_id
        existing=db.execute('SELECT atom_id FROM atoms WHERE fqdn=%s FOR UPDATE',(spec.fqdn,)).fetchone()
        if existing:
            if existing['atom_id']!=atom_id:raise ValueError('provider identity already exists under a different draft id')
        else:
            if db.execute('SELECT 1 FROM artifacts WHERE fqdn=%s',(spec.fqdn,)).fetchone():raise ValueError('canonical provider identity already exists')
            insert(db,'atoms',atom)
            canonical_columns={r['column_name'] for r in db.execute("SELECT column_name FROM information_schema.columns WHERE table_name='artifacts'").fetchall()}
            canonical={k:v for k,v in atom.items() if k in canonical_columns}
            canonical.update(artifact_id=atom_id,artifact_kind='atom',status='draft')
            insert(db,'artifacts',canonical)
            insert(db,'atom_versions',{'version_id':spec.version_id,'atom_id':atom_id,'content_hash':spec.content_hash,'semver':spec.semver,'s3_key':'','fingerprint':spec.fingerprint,'is_latest':True})
            insert(db,'artifact_versions',{'version_id':spec.version_id,'artifact_id':atom_id,'content_hash':spec.content_hash,'semver':spec.semver,'s3_key':'','fingerprint':spec.fingerprint,'is_latest':True})
            insert(db,'artifact_audit_evidence',{'artifact_id':atom_id,'version_id':spec.version_id,'audit_type':'asset_integrity_check','passed':True,'status':'completed','source_kind':'automated','runner_version':'provider-draft-intake.v1','details':Jsonb({'scope':'local provider identity and source integrity only; legacy flagged means pending review, canonical status is draft; not approval or distribution availability','source_file_sha256':source_sha,'provider_content_hash':spec.content_hash,'publication_blockers':['semantic_review_pending','provider_distribution_not_verified'],'review_bundle':'data/audit_reviews/series_resistance_pending.json'})})
            counts['created_atoms']+=1
        for table,id_key in [('atom_versions','atom_id'),('artifact_versions','artifact_id')]:
            version=db.execute(sql.SQL('SELECT content_hash,{} AS owner FROM {} WHERE version_id=%s').format(sql.Identifier(id_key),sql.Identifier(table)),(spec.version_id,)).fetchone()
            if not version or version!={'content_hash':spec.content_hash,'owner':atom_id}:raise ValueError('provider version identity drift')
        evidence=db.execute("SELECT details FROM artifact_audit_evidence WHERE artifact_id=%s AND version_id=%s AND runner_version='provider-draft-intake.v1'",(atom_id,spec.version_id)).fetchall()
        if len(evidence)!=1 or evidence[0]['details']['source_file_sha256']!=source_sha:raise ValueError('provider source integrity drift')
        nodes=db.execute("SELECT n.version_id,n.node_id FROM artifact_cdg_nodes n JOIN artifact_versions v USING(version_id) JOIN artifacts a USING(artifact_id) WHERE n.matched_primitive=%s AND v.semver LIKE '0.0.0+execution.%%' AND a.status='draft' FOR UPDATE OF n FOR SHARE OF v,a",(spec.fqdn,)).fetchall()
        for node in nodes:
            existing_binding=db.execute('SELECT bound_artifact_fqdn,bound_version_content_hash FROM artifact_cdg_bindings WHERE version_id=%s AND node_id=%s',(node['version_id'],node['node_id'])).fetchall()
            expected={'bound_artifact_fqdn':spec.fqdn,'bound_version_content_hash':spec.content_hash}
            if existing_binding:
                if existing_binding!=[expected]:raise ValueError('existing primitive binding conflicts')
                continue
            insert(db,'artifact_cdg_bindings',{'version_id':node['version_id'],'node_id':node['node_id'],**expected,'binding_confidence':1.0,'binding_source':'provider-draft-intake.v1','status':'active','evidence_summary':Jsonb({'scope':'exact local implementation identity only; target remains draft and unserved','provider_version_id':spec.version_id,'provider_source_sha256':source_sha,'source_derivation_review_required':True})})
            counts['created_bindings']+=1
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'counts':counts},indent=2))


if __name__=='__main__':main()
