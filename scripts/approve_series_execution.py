#!/usr/bin/env python3
"""Approve the exact tested series-resistance execution CDG at Community tier."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
from uuid import uuid5
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sciona.atoms.electrical.series_resistance import series_resistance
from sciona.architect.handoff import CDGExport
from sciona.cdg_projection import build_published_cdg_projection
from sciona.physics_ingest.pdg_evidence import prepare_pdg_evidence,_digest
from sciona.physics_ingest.review import assess_publishability
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.import_residual_execution_drafts import ensure_row


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--symbol-file',required=True,type=Path);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    symbols=args.symbol_file.read_bytes();root=Path(__file__).resolve().parents[1]
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('series-execution-community-approval.v1'))")
        intakes=db.execute("SELECT e.*,a.fqdn,a.status AS artifact_status,a.is_publishable,v.content_hash,v.trust_tier FROM artifact_audit_evidence e JOIN artifacts a USING(artifact_id) JOIN artifact_versions v USING(version_id) WHERE e.runner_version='execution-draft-import.v1' FOR UPDATE OF a,v FOR SHARE OF e").fetchall()
        if len(intakes)!=1:raise ValueError('one series execution intake required')
        intake=intakes[0]
        if intake['artifact_status'] not in {'draft','approved'} or intake['trust_tier']!=3:raise ValueError('unexpected publication state')
        def evidence(identity,runner):
            row=db.execute('SELECT * FROM artifact_audit_evidence WHERE evidence_id=%s FOR SHARE',(identity,)).fetchone()
            if not row or row['runner_version']!=runner or not row['passed']:raise ValueError('required evidence unavailable')
            for path,digest in row['details'].get('implementation_hashes',{}).items():
                if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest:raise ValueError('evidence implementation drift')
            return row
        executed=evidence(intake['details']['source_execution_evidence_id'],'series-cdg-execution.v1')
        if _digest(executed['details'])!=intake['details']['source_execution_evidence_sha256']:raise ValueError('execution intake mismatch')
        physical=evidence(executed['details']['source_runtime_evidence_id'],'series-terminal-runtime.v1')
        if _digest(physical['details'])!=executed['details']['source_runtime_details_sha256']:raise ValueError('physical evidence mismatch')
        replay=evidence(physical['details']['replay_evidence_id'],'pdg-graph-replay.v1')
        if _digest(replay['details'])!=physical['details']['replay_details_sha256']:raise ValueError('replay evidence mismatch')
        if len({str(r['version_id']) for r in [executed,physical,replay]})!=1:raise ValueError('source version mismatch')
        if hashlib.sha256(Path(inspect.getfile(series_resistance)).read_bytes()).hexdigest()!=executed['details']['provider_implementation_sha256']:raise ValueError('provider source drift')
        source=db.execute('SELECT a.fqdn,v.content_hash FROM artifact_versions v JOIN artifacts a USING(artifact_id) WHERE version_id=%s FOR SHARE OF a,v',(replay['version_id'],)).fetchone()
        if source['content_hash']!=replay['details']['graph_content_hash']:raise ValueError('source graph version drift')
        nodes=db.execute('SELECT * FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id FOR SHARE',(replay['version_id'],)).fetchall()
        edges=db.execute('SELECT * FROM artifact_cdg_edges WHERE version_id=%s ORDER BY source_id,target_id FOR SHARE',(replay['version_id'],)).fetchall()
        projection={'nodes':[{k:str(v) if k=='version_id' else v for k,v in n.items()} for n in nodes],'edges':[{k:str(v) if k=='version_id' else v for k,v in e.items()} for e in edges]}
        if _digest(projection)!=replay['details']['graph_projection_sha256']:raise ValueError('source proof projection changed')
        assessments={}
        for key,digest in replay['details']['expression_evidence_sha256'].items():
            row=db.execute('SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload FROM artifact_symbolic_expressions e JOIN physics_equation_candidates q USING(candidate_id) JOIN physics_ingest_snapshots s USING(snapshot_id) WHERE expression_id=%s FOR SHARE OF e,q,s',(key,)).fetchone()
            fresh=prepare_pdg_evidence(row,symbols)
            if fresh!=row['evidence_json']['pdg_source_comparison'] or _digest(fresh)!=digest:raise ValueError('source expression evidence drift')
            candidate=db.execute('SELECT * FROM physics_equation_candidates WHERE candidate_id=%s',(row['candidate_id'],)).fetchone()
            variables=db.execute('SELECT * FROM artifact_symbolic_variables WHERE expression_id=%s FOR SHARE',(key,)).fetchall()
            bounds=db.execute('SELECT * FROM artifact_validity_bounds WHERE expression_id=%s FOR SHARE',(key,)).fetchall()
            refs=db.execute('SELECT * FROM artifact_references WHERE artifact_id=%s FOR SHARE',(row['artifact_id'],)).fetchall()
            assessed=assess_publishability(candidate=candidate,expression=row,variables=variables,validity_bounds=bounds,references=refs)
            if not assessed.publishable:raise ValueError('source expression Community review failed')
            assessments[key]=assessed.to_report().to_dict()
        graph=CDGExport.model_validate(executed['details']['execution_cdg'])
        document=db.execute('SELECT get_artifact_document(%s) AS document',(intake['fqdn'],)).fetchone()['document']
        if _artifact_document_to_cdg(document,version_id=str(intake['version_id']),content_hash=intake['content_hash'],require_execution_envelope=True)!=graph:raise ValueError('execution graph differs from validated snapshot')
        bindings=db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s FOR SHARE',(intake['version_id'],)).fetchall()
        if len(bindings)!=1 or bindings[0]['status']!='active' or bindings[0]['bound_artifact_fqdn']!=graph.nodes[0].matched_primitive:raise ValueError('execution binding mismatch')
        provider=db.execute("SELECT a.artifact_id,v.version_id FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN atoms legacy ON legacy.atom_id=a.artifact_id WHERE a.fqdn=%s AND v.content_hash=%s AND v.is_latest AND v.trust_tier=3 AND a.status='approved' AND a.is_publishable AND legacy.status='approved' AND legacy.is_publishable FOR SHARE OF a,v,legacy",(bindings[0]['bound_artifact_fqdn'],bindings[0]['bound_version_content_hash'])).fetchone()
        if not provider:raise ValueError('exact provider version is not approved and served')
        provider_review=db.execute("SELECT * FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='local-physics-community-approval.v1' AND passed FOR SHARE",(provider['version_id'],)).fetchone()
        if not provider_review:raise ValueError('version-bound provider approval missing')
        dependencies=db.execute('SELECT dependency_artifact_fqdn,dependency_content_hash,optional,binding_metadata FROM artifact_dependencies WHERE dependent_version_id=%s FOR SHARE',(intake['version_id'],)).fetchall()
        if len(dependencies)!=1 or dependencies[0]['dependency_artifact_fqdn']!=source['fqdn'] or dependencies[0]['dependency_content_hash']!=source['content_hash'] or dependencies[0]['optional']:raise ValueError('mandatory proof provenance missing')
        if db.execute("SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND passed=false AND status IN ('failed','completed') LIMIT 1",(intake['version_id'],)).fetchone():raise ValueError('execution version has failed audit evidence')
        reference=db.execute("SELECT ref_id,ref_key,title,url,verified FROM artifact_references WHERE artifact_id=%s AND ref_key='openstax-university-physics-v2-10-2' FOR SHARE",(replay['artifact_id'],)).fetchone()
        if not reference or not reference['verified']:raise ValueError('scientific reference missing')
        ensure_row(db,'artifact_references',{'artifact_id':intake['artifact_id'],'ref_key':reference['ref_key']},{'artifact_id':intake['artifact_id'],**reference,'source':'llm_extracted','confidence':'high','relevance_note':'Supports the reviewed ohmic series model; exact implementation and domain checked by automated evidence.'})
        limitations=['Community tier: automated review, no human certification.','Two nonnegative ohmic components in one series branch; common nonzero current and additive voltage drops.','Finite real NumPy-broadcastable inputs; output is equivalent resistance in ohms.','Embedded draft metadata records the original validation snapshot; catalog publication state is authoritative.']
        rollup={'artifact_id':intake['artifact_id'],'overall_verdict':'acceptable_with_limits','structural_status':'pass','runtime_status':'pass','semantic_status':'pass','developer_semantics_status':'pass','review_status':'approved','review_semantic_verdict':'pass','review_developer_semantics_verdict':'pass','trust_readiness':'ready','review_limitations':limitations,'review_required_actions':[],'trust_blockers':[],'acceptability_band':'acceptable_with_limits','parity_coverage_level':'positive_and_negative','parity_test_status':'pass'}
        ensure_row(db,'artifact_audit_rollups',{'artifact_id':intake['artifact_id']},rollup)
        approval={'publication_tier':3,'review_source':'automated','content_hash':intake['content_hash'],'source_dependency_review':'passed as mandatory proof provenance, not an invoked numerical dependency','source_expression_assessments':assessments,'source_replay_evidence_id':str(replay['evidence_id']),'source_replay_sha256':_digest(replay['details']),'physical_evidence_id':str(physical['evidence_id']),'physical_evidence_sha256':_digest(physical['details']),'execution_evidence_id':str(executed['evidence_id']),'execution_evidence_sha256':_digest(executed['details']),'provider_approval_evidence_id':str(provider_review['evidence_id']),'provider_approval_sha256':_digest(provider_review['details']),'limitations':limitations}
        audit_id=uuid5(intake['version_id'],'series-execution-community-approval.v1')
        created=ensure_row(db,'artifact_audit_evidence',{'evidence_id':audit_id},{'evidence_id':audit_id,'artifact_id':intake['artifact_id'],'version_id':intake['version_id'],'audit_type':'semantic_audit','passed':True,'status':'completed','source_kind':'automated','runner_version':'series-execution-community-approval.v1','details':Jsonb(approval)})
        topo=build_published_cdg_projection(artifact={'artifact_id':str(intake['artifact_id']),'fqdn':intake['fqdn']},version={'version_id':str(intake['version_id']),'content_hash':intake['content_hash']},cdg=graph).topo_hash
        db.execute("UPDATE artifacts SET status='approved',is_publishable=true,verified_leaf_coverage=1,leaf_count=1,top_level_input_arity=3,top_level_output_arity=1,topo_hash=%s,source_symbol='series_equivalent_resistance',description=%s WHERE artifact_id=%s",(topo,'Compute equivalent resistance of two nonnegative ohmic resistors in series, with a nonzero common-current guard. Returns resistance in ohms; finite scalar or broadcastable array inputs.',intake['artifact_id']))
        db.execute('UPDATE artifact_versions SET is_latest=(version_id=%s) WHERE artifact_id=%s',(intake['version_id'],intake['artifact_id']))
        if not db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s',(intake['artifact_id'],)).fetchone():raise ValueError('approved graph not served')
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'new_approval':created,'source_expressions_reviewed':len(assessments)},indent=2))


if __name__=='__main__':main()
