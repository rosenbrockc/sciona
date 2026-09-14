#!/usr/bin/env python3
"""Approve the exact source-validated causal execution CDG at Community tier."""
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
from sciona.cdg_projection import build_published_cdg_projection
from sciona.physics_ingest.pdg_evidence import _digest
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.import_residual_execution_drafts import ensure_row


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('causal-execution-community.v1'))")
        rows=db.execute("SELECT e.*,a.fqdn,a.status AS artifact_status,v.content_hash,v.trust_tier FROM artifact_audit_evidence e JOIN artifacts a USING(artifact_id) JOIN artifact_versions v USING(version_id) WHERE e.runner_version='causal-execution-intake.v1' FOR UPDATE OF a,v FOR SHARE OF e").fetchall()
        if len(rows)!=1:raise ValueError('unique causal execution intake required')
        intake=rows[0]
        if not intake['passed'] or intake['artifact_status'] not in {'draft','approved'} or intake['trust_tier']!=3:raise ValueError('unexpected review state')
        execution=db.execute("SELECT * FROM artifact_audit_evidence WHERE evidence_id=%s AND runner_version='causal-full-execution.v1' AND passed FOR SHARE",(intake['details']['source_execution_evidence_id'],)).fetchone()
        if not execution or _digest(execution['details'])!=intake['details']['source_execution_evidence_sha256']:raise ValueError('execution evidence mismatch')
        details=execution['details']
        for path,digest in details['implementation_hashes'].items():
            if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest:raise ValueError('execution implementation drift')
        for module,digest in details['provider_hashes'].items():
            if hashlib.sha256(Path(inspect.getfile(importlib.import_module(module))).read_bytes()).hexdigest()!=digest:raise ValueError('transitive provider drift')
        source=db.execute("SELECT e.details,a.fqdn,v.content_hash FROM artifact_audit_evidence e JOIN artifacts a USING(artifact_id) JOIN artifact_versions v USING(version_id) WHERE e.version_id=%s AND e.runner_version='competition-intake.v1' FOR SHARE OF e,a,v",(execution['version_id'],)).fetchone()
        if not source or _digest(source['details']['snapshot'])!=details['source_intake_sha256']:raise ValueError('source provenance mismatch')
        graph=CDGExport.model_validate(details['execution_graph'])
        doc=db.execute('SELECT get_artifact_document(%s) AS d',(intake['fqdn'],)).fetchone()['d']
        if _artifact_document_to_cdg(doc,version_id=str(intake['version_id']),content_hash=intake['content_hash'],require_execution_envelope=True)!=graph:raise ValueError('catalog graph differs from validated graph')
        bindings=db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s FOR SHARE',(intake['version_id'],)).fetchall()
        if len(bindings)!=len(graph.nodes):raise ValueError('incomplete provider bindings')
        provider_reviews={}
        for node in graph.nodes:
            matches=[b for b in bindings if b['node_id']==node.node_id]
            if len(matches)!=1:raise ValueError('ambiguous node binding')
            b=matches[0]
            if b['status']!='active' or b['evidence_summary']['runtime_fqdn']!=node.matched_primitive:raise ValueError('runtime identity mismatch')
            provider=db.execute("SELECT e.* FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_audit_evidence e ON e.version_id=v.version_id JOIN atoms legacy ON legacy.atom_id=a.artifact_id WHERE a.fqdn=%s AND v.content_hash=%s AND v.is_latest AND v.trust_tier=3 AND a.status='approved' AND a.is_publishable AND legacy.status='approved' AND legacy.is_publishable AND e.runner_version='causal-provider-community.v1' AND e.passed FOR SHARE OF a,v,e,legacy",(b['bound_artifact_fqdn'],b['bound_version_content_hash'])).fetchone()
            if not provider or provider['details']['runtime_fqdn']!=node.matched_primitive:raise ValueError('exact provider approval missing')
            if db.execute("SELECT 1 FROM artifact_audit_evidence WHERE version_id=ANY(%s) AND passed=false AND status IN ('failed','completed')",([intake['version_id'],provider['version_id']],)).fetchone():raise ValueError('explicit failed execution/provider audit')
            provider_reviews[str(provider['evidence_id'])]=_digest(provider['details'])
        deps=db.execute('SELECT * FROM artifact_dependencies WHERE dependent_version_id=%s FOR SHARE',(intake['version_id'],)).fetchall()
        if len(deps)!=1 or deps[0]['dependency_artifact_fqdn']!=source['fqdn'] or deps[0]['dependency_content_hash']!=source['content_hash'] or deps[0]['optional']:raise ValueError('mandatory conceptual source reference missing')
        reference=db.execute("SELECT ref_id,title,url FROM references_registry WHERE ref_id='jarfo-cause-effect-f4d0f0d8' FOR SHARE").fetchone()
        if not reference:raise ValueError('source registry reference missing')
        ensure_row(db,'artifact_references',{'artifact_id':intake['artifact_id'],'ref_key':reference['ref_id']},{'artifact_id':intake['artifact_id'],**reference,'ref_key':reference['ref_id'],'source':'llm_extracted','verified':True,'confidence':'high','relevance_note':'Pinned source feature/training/prediction parity under recorded modern libraries, including the 500-estimator setting.'})
        limitations=['Tier 3 Community automated review; no human certification or Tier 2 usage claim.','Requires typed finite nonconstant paired observations and representative labelled training pairs with all three causal classes. Undefined feature regimes are rejected.','Synthetic same-library source parity is validated; no original historical-library reproduction, real competition score or causal accuracy guarantee.','Training runs locally and yields in-memory models. Default 500 estimators can be computationally expensive.','Returns forward/reverse paired scores; callers must retain pair ordering. Embedded draft metadata preserves the immutable validation snapshot.']
        rollup={'artifact_id':intake['artifact_id'],'overall_verdict':'acceptable_with_limits','structural_status':'pass','runtime_status':'pass','semantic_status':'pass','developer_semantics_status':'pass','review_status':'approved','review_semantic_verdict':'pass','review_developer_semantics_verdict':'pass','trust_readiness':'ready','review_limitations':limitations,'review_required_actions':[],'trust_blockers':[],'acceptability_band':'acceptable_with_limits','parity_coverage_level':'positive_and_negative','parity_test_status':'pass'}
        ensure_row(db,'artifact_audit_rollups',{'artifact_id':intake['artifact_id']},rollup)
        approval={'publication_tier':3,'review_source':'automated','content_hash':intake['content_hash'],'execution_evidence_id':str(execution['evidence_id']),'execution_evidence_sha256':_digest(details),'provider_approvals':provider_reviews,'source_intake_sha256':details['source_intake_sha256'],'source_dependency_review':'Conceptual provenance reviewed through pinned source code and full execution parity; not an invoked numerical dependency.','limitations':limitations}
        audit_id=uuid5(intake['version_id'],'causal-execution-community.v1')
        created=ensure_row(db,'artifact_audit_evidence',{'evidence_id':audit_id},{'evidence_id':audit_id,'artifact_id':intake['artifact_id'],'version_id':intake['version_id'],'audit_type':'semantic_audit','passed':True,'status':'completed','source_kind':'automated','runner_version':'causal-execution-community.v1','details':Jsonb(approval)})
        inputs=db.execute("SELECT count(*) AS n FROM artifact_io_specs WHERE version_id=%s AND direction='input'",(intake['version_id'],)).fetchone()['n']
        topo=build_published_cdg_projection(artifact={'artifact_id':str(intake['artifact_id']),'fqdn':intake['fqdn']},version={'version_id':str(intake['version_id']),'content_hash':intake['content_hash']},cdg=graph).topo_hash
        db.execute("UPDATE artifacts SET status='approved',is_publishable=true,verified_leaf_coverage=1,leaf_count=%s,top_level_input_arity=%s,top_level_output_arity=1,topo_hash=%s,source_symbol='causal_pair_training_prediction' WHERE artifact_id=%s",(len(graph.nodes),inputs,topo,intake['artifact_id']))
        db.execute('UPDATE artifact_versions SET is_latest=(version_id=%s) WHERE artifact_id=%s',(intake['version_id'],intake['artifact_id']))
        if not db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s',(intake['artifact_id'],)).fetchone():raise ValueError('approved CDG not served')
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'new_approval':created,'approved_provider_bindings':len(provider_reviews)},indent=2))


if __name__=='__main__':main()
