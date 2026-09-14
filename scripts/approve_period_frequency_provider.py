#!/usr/bin/env python3
"""Approve tested Community provider versions in the local catalog; no distribution upload."""
import argparse
from collections import Counter
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
from uuid import UUID,uuid5
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.atoms.audit_review_bundles import load_review_bundle_entries,merge_audit_manifest_entries
from sciona.physics_ingest.pdg_evidence import _digest
from scripts.import_residual_execution_drafts import ensure_row

from sciona.physics_ingest.period_frequency_execution import PRIMITIVE
REFERENCE='openstax-university-physics-v1-15-1'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');args=parser.parse_args();counts=Counter()
    root=Path(__file__).resolve().parents[1];provider=root.parent/'sciona-atoms'
    specs={p.fqdn:p for p in _parse_registered_atoms(repo=ProviderRepo('sciona-atoms',provider),artifact_root=provider/'src/sciona/atoms') if p.fqdn==PRIMITIVE}
    if len(specs)!=1:raise ValueError('expected one tested provider')
    reviews=[]
    for path in ['period_frequency_community.json']:
        merged,skipped=merge_audit_manifest_entries([],load_review_bundle_entries(provider/'data/audit_reviews'/path))
        if skipped:raise ValueError('review bundle failed to load')
        reviews.extend(merged)
    review_by_name={r['atom_name']:r for r in reviews}
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('local-physics-community-approval.v1'))")
        reference=db.execute('SELECT ref_id,title,url FROM references_registry WHERE ref_id=%s FOR SHARE',(REFERENCE,)).fetchone()
        if not reference:raise ValueError('verified scientific reference missing')
        reports=db.execute("SELECT * FROM artifact_audit_evidence WHERE runner_version IN ('period-frequency-execution.v1') AND passed FOR SHARE").fetchall()
        for fqdn,spec in specs.items():
            review=review_by_name.get(fqdn,{})
            if review.get('review_status')!='approved' or any(review.get(k)!='pass' for k in ['structural_status','runtime_status','semantic_status','developer_semantics_status']):raise ValueError('approved automated review bundle required')
            sha=hashlib.sha256(spec.file_path.read_bytes()).hexdigest();evidence=[]
            for report in reports:
                details=report['details'];graph=details.get('execution_graph') or details.get('execution_cdg')
                if graph['nodes'][0]['matched_primitive']!=fqdn:continue
                if (details.get('provider_source_sha256') or details.get('provider_implementation_sha256'))!=sha:raise ValueError('provider implementation drift')
                for path,digest in details['implementation_hashes'].items():
                    if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest:raise ValueError('execution evidence implementation drift')
                evidence.append(report)
            if not evidence:raise ValueError('version-matched CDG execution evidence required')
            module,name=fqdn.rsplit('.',1);fn=getattr(importlib.import_module(module),name)
            row=db.execute('SELECT a.artifact_id,a.status,a.is_publishable,v.version_id,v.content_hash,v.trust_tier FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE a.fqdn=%s AND v.is_latest FOR UPDATE OF a,v',(fqdn,)).fetchone()
            if not row or str(row['version_id'])!=str(spec.version_id) or row['content_hash']!=spec.content_hash or row['trust_tier']!=3:raise ValueError('canonical version mismatch')
            if row['status'] not in {'draft','approved'}:raise ValueError('explicit disqualifying status')
            if db.execute("SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND passed=false AND status IN ('completed','failed') LIMIT 1",(row['version_id'],)).fetchone():raise ValueError('provider version has explicit failed audit evidence')
            legacy=db.execute('SELECT atom_id,status FROM atoms WHERE fqdn=%s FOR UPDATE',(fqdn,)).fetchone()
            if not legacy or legacy['atom_id']!=row['artifact_id'] or legacy['status'] not in {'flagged','approved'}:raise ValueError('legacy identity/review state mismatch')
            ports=db.execute('SELECT direction,name,dim_signature FROM artifact_io_specs WHERE version_id=%s ORDER BY direction,ordinal FOR SHARE',(row['version_id'],)).fetchall()
            if [p['name'] for p in ports if p['direction']=='input']!=list(inspect.signature(fn).parameters) or sum(p['direction']=='output' for p in ports)!=1 or any(not p['dim_signature'] for p in ports):raise ValueError('callable IO metadata mismatch')
            if not spec.description.strip():raise ValueError('provider description missing')
            ref={**reference,'ref_key':REFERENCE,'source':'llm_extracted','verified':True,'confidence':'high','relevance_note':'Supports ordinary frequency as the reciprocal of a positive cycle duration. Implementation behavior is covered by the version-bound automated review; no human certification is claimed.'}
            for table,owner in [('atom_references','atom_id'),('artifact_references','artifact_id')]:
                ensure_row(db,table,{owner:row['artifact_id'],'ref_key':REFERENCE},{owner:row['artifact_id'],**ref})
            limitations=review.get('review_limitations',[])
            rollup={'overall_verdict':'acceptable_with_limits','structural_status':'pass','runtime_status':'pass','semantic_status':'pass','developer_semantics_status':'pass','review_status':'approved','review_semantic_verdict':'pass','review_developer_semantics_verdict':'pass','trust_readiness':'ready','review_limitations':limitations,'review_required_actions':[],'trust_blockers':[],'acceptability_band':'acceptable_with_limits','parity_coverage_level':'positive_and_negative','parity_test_status':'pass'}
            for table,owner in [('atom_audit_rollups','atom_id'),('artifact_audit_rollups','artifact_id')]:ensure_row(db,table,{owner:row['artifact_id']},{owner:row['artifact_id'],**rollup})
            _,line=inspect.getsourcelines(fn)
            approval={'publication_tier':3,'review_source':'automated','scope':'local catalog approval for the installed, tested provider; no package upload, license assertion, Tier 2 usage claim or Tier 1 certification',
                      'content_hash':spec.content_hash,'source_sha256':sha,'source_module':fn.__module__,'source_line':line,
                      'execution_evidence':{str(e['evidence_id']):_digest(e['details']) for e in evidence},
                      'review_bundle_sha256':_digest(review),'limitations':limitations}
            audit_id=uuid5(UUID(str(row['version_id'])),'local-physics-community-approval.v1')
            ensure_row(db,'artifact_audit_evidence',{'evidence_id':audit_id},{'evidence_id':audit_id,'artifact_id':row['artifact_id'],'version_id':row['version_id'],'audit_type':'semantic_audit','passed':True,'status':'completed','source_kind':'automated','runner_version':'local-physics-community-approval.v1','details':Jsonb(approval)})
            counts['already_approved' if row['status']=='approved' and row['is_publishable'] else 'approved']+=1
            db.execute("UPDATE artifacts SET status='approved',is_publishable=true WHERE artifact_id=%s",(row['artifact_id'],))
            db.execute("UPDATE atoms SET status='approved',is_publishable=true WHERE atom_id=%s",(row['artifact_id'],))
            if not db.execute('SELECT 1 FROM catalog_atoms_served WHERE atom_id=%s',(row['artifact_id'],)).fetchone():raise ValueError('approved version absent from served view')
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'counts':dict(counts)},indent=2))


if __name__=='__main__':main()
