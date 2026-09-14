#!/usr/bin/env python3
"""Apply version-bound automated Community reviews to the causal graph providers."""
import argparse
from collections import Counter
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
from sciona.physics_ingest.pdg_evidence import _digest
from scripts.import_residual_execution_drafts import ensure_row


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];reviews={};counts=Counter()
    for repo in ['sciona-atoms','sciona-atoms-ml']:
        bundle=json.loads((root.parent/repo/'data/audit_reviews/causal_execution_community.json').read_text())
        for entry in bundle['atoms']:
            if entry['atom_name'] in reviews:raise ValueError('duplicate provider review')
            reviews[entry['atom_name']]=entry['audit']
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('causal-provider-community.v1'))")
        records=db.execute("SELECT e.*,a.fqdn,a.status AS artifact_status,v.content_hash,v.is_latest,v.trust_tier FROM artifact_audit_evidence e JOIN artifacts a USING(artifact_id) JOIN artifact_versions v USING(version_id) WHERE e.runner_version='causal-provider-intake.v1' ORDER BY e.evidence_id FOR UPDATE OF a,v FOR SHARE OF e").fetchall()
        if len(records)!=8 or set(reviews)!={r['fqdn'] for r in records}:raise ValueError('eight exact provider reviews required')
        for record in records:
            review=reviews[record['fqdn']]
            if review['publication_tier']!=3 or review['review_source']!='automated' or review['review_status']!='approved' or any(review[k]!='pass' for k in ['structural_status','runtime_status','semantic_status','developer_semantics_status']) or review['review_required_actions']:raise ValueError('review not ready')
            if str(record['version_id'])!=review['version_id'] or record['trust_tier']!=3 or not record['is_latest'] or record['artifact_status'] not in {'draft','approved'}:raise ValueError('review version/state mismatch')
            module,name=review['runtime_fqdn'].rsplit('.',1);fn=getattr(importlib.import_module(module),name)
            source_hash=hashlib.sha256(Path(inspect.getfile(inspect.unwrap(fn))).read_bytes()).hexdigest()
            if source_hash!=review['source_sha256'] or source_hash!=record['details']['source_sha256']:raise ValueError('provider source drift')
            executed=db.execute("SELECT * FROM artifact_audit_evidence WHERE evidence_id=%s AND runner_version='causal-full-execution.v1' AND passed FOR SHARE",(review['source_execution_evidence_id'],)).fetchone()
            if not executed or _digest(executed['details'])!=review['source_execution_evidence_sha256']:raise ValueError('execution evidence drift')
            for path,digest in executed['details']['implementation_hashes'].items():
                if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest:raise ValueError('execution implementation drift')
            for provider,digest in executed['details']['provider_hashes'].items():
                if hashlib.sha256(Path(inspect.getfile(importlib.import_module(provider))).read_bytes()).hexdigest()!=digest:raise ValueError('transitive implementation drift')
            if db.execute("SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND passed=false AND status IN ('failed','completed')",(record['version_id'],)).fetchone():raise ValueError('explicit failed provider audit')
            legacy=db.execute('SELECT a.status,v.content_hash FROM atoms a JOIN atom_versions v USING(atom_id) WHERE a.atom_id=%s AND v.version_id=%s AND v.is_latest FOR UPDATE OF a FOR SHARE OF v',(record['artifact_id'],record['version_id'])).fetchone()
            if not legacy or legacy['content_hash']!=record['content_hash'] or legacy['status'] not in {'flagged','approved'}:raise ValueError('legacy identity mismatch')
            interface=db.execute("SELECT * FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='causal-provider-interface.v1' AND passed FOR SHARE",(record['version_id'],)).fetchone()
            if not interface or interface['details']['signature']!=str(inspect.signature(fn)):raise ValueError('interface review missing')
            for table in ['artifact_io_specs','atom_io_specs']:
                ports=db.execute('SELECT name,required,default_value_repr FROM '+table+" WHERE version_id=%s AND direction='input' ORDER BY ordinal FOR SHARE",(record['version_id'],)).fetchall()
                if [p['name'] for p in ports]!=list(inspect.signature(fn).parameters):raise ValueError('incomplete callable interface')
                for p,param in zip(ports,inspect.signature(fn).parameters.values()):
                    required=param.default is inspect.Parameter.empty
                    if p['required']!=required or p['default_value_repr']!=('' if required else repr(param.default)):raise ValueError('default contract mismatch')
            ref_id='jarfo-cause-effect-f4d0f0d8'
            url='https://github.com/jarfo/cause-effect/tree/'+executed['details']['source_commit']
            ensure_row(db,'references_registry',{'ref_id':ref_id},{'ref_id':ref_id,'ref_type':'repository','title':'Pinned cause-effect source implementation','url':url})
            ref={'ref_id':ref_id,'ref_key':ref_id,'title':'Pinned cause-effect source implementation','url':url,'source':'llm_extracted','verified':True,'confidence':'high','relevance_note':'Pinned code parity for reconstructed numerical behavior under recorded libraries; no human certification or predictive-quality claim.'}
            for table,key in [('artifact_references','artifact_id'),('atom_references','atom_id')]:ensure_row(db,table,{key:record['artifact_id'],'ref_key':ref_id},{key:record['artifact_id'],**ref})
            rollup={'overall_verdict':'acceptable_with_limits','structural_status':'pass','runtime_status':'pass','semantic_status':'pass','developer_semantics_status':'pass','review_status':'approved','review_semantic_verdict':'pass','review_developer_semantics_verdict':'pass','trust_readiness':'ready','review_limitations':review['review_limitations'],'review_required_actions':[],'trust_blockers':[],'acceptability_band':'acceptable_with_limits','parity_coverage_level':'positive_and_negative','parity_test_status':'pass'}
            ensure_row(db,'artifact_audit_rollups',{'artifact_id':record['artifact_id']},{'artifact_id':record['artifact_id'],**rollup})
            # Preserve existing legacy reviews; new providers receive the same review.
            if not db.execute('SELECT 1 FROM atom_audit_rollups WHERE atom_id=%s',(record['artifact_id'],)).fetchone():ensure_row(db,'atom_audit_rollups',{'atom_id':record['artifact_id']},{'atom_id':record['artifact_id'],**rollup})
            details={'publication_tier':3,'review_source':'automated','content_hash':record['content_hash'],'source_sha256':source_hash,'review_bundle_sha256':_digest(review),'execution_evidence_id':str(executed['evidence_id']),'execution_evidence_sha256':_digest(executed['details']),'interface_evidence_id':str(interface['evidence_id']),'runtime_fqdn':review['runtime_fqdn'],'limitations':review['review_limitations']}
            evidence_id=uuid5(record['version_id'],'causal-provider-community.v1')
            counts['new_approvals']+=ensure_row(db,'artifact_audit_evidence',{'evidence_id':evidence_id},{'evidence_id':evidence_id,'artifact_id':record['artifact_id'],'version_id':record['version_id'],'audit_type':'semantic_audit','passed':True,'status':'completed','source_kind':'automated','runner_version':'causal-provider-community.v1','details':Jsonb(details)})
            db.execute("UPDATE artifacts SET status='approved',is_publishable=true WHERE artifact_id=%s",(record['artifact_id'],))
            db.execute("UPDATE atoms SET status='approved',is_publishable=true WHERE atom_id=%s",(record['artifact_id'],))
            if not db.execute('SELECT 1 FROM catalog_atoms_served WHERE atom_id=%s',(record['artifact_id'],)).fetchone():raise ValueError('approved provider not served')
            counts['providers_verified']+=1
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'counts':dict(counts)},indent=2))


if __name__=='__main__':main()
