#!/usr/bin/env python3
"""Approve source-bound mathematical residual CDGs at Community tier."""
import argparse
from collections import Counter
import hashlib
import inspect
import json
import os
from pathlib import Path
from uuid import uuid5
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sciona.atoms.electrical import residuals
from sciona.cdg_projection import build_published_cdg_projection
from sciona.ghost.symbolic import deserialize_expr
from sciona.physics_ingest.pdg_evidence import prepare_pdg_evidence, _digest
from sciona.physics_ingest.residual_execution import build_residual_execution
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.import_residual_execution_drafts import ensure_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbol-file', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    symbols = args.symbol_file.read_bytes()
    root = Path(__file__).resolve().parents[1]
    counts = Counter()
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row) as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('residual-community-approval.v1'))")
        intakes = db.execute("SELECT e.*,a.fqdn,a.status AS artifact_status,v.content_hash,v.trust_tier FROM artifact_audit_evidence e JOIN artifacts a USING(artifact_id) JOIN artifact_versions v USING(version_id) WHERE e.runner_version='residual-execution-import.v1' ORDER BY e.evidence_id FOR UPDATE OF a,v FOR SHARE OF e").fetchall()
        if not intakes:
            raise ValueError('no residual execution intakes')
        for intake in intakes:
            if not intake['passed'] or intake['artifact_status'] not in {'draft', 'approved'} or intake['trust_tier'] != 3:
                raise ValueError('unexpected intake state')
            executed = db.execute("SELECT * FROM artifact_audit_evidence WHERE evidence_id=%s AND runner_version='residual-cdg-execution.v1' AND passed FOR SHARE", (intake['details']['source_execution_evidence_id'],)).fetchone()
            if not executed or _digest(executed['details']) != intake['details']['source_execution_evidence_sha256']:
                raise ValueError('execution evidence mismatch')
            details = executed['details']
            for path, digest in details['implementation_hashes'].items():
                if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest:
                    raise ValueError('execution implementation drift')
            if hashlib.sha256(Path(inspect.getfile(residuals)).read_bytes()).hexdigest() != details['provider_source_sha256']:
                raise ValueError('provider implementation drift')
            proof = db.execute("SELECT * FROM artifact_audit_evidence WHERE evidence_id=%s AND runner_version='residual-reuse.v1' AND passed FOR SHARE", (details['reuse_proof_evidence_id'],)).fetchone()
            if not proof or proof['version_id'] != executed['version_id'] or _digest(proof['details']) != details['reuse_proof_sha256']:
                raise ValueError('reuse proof mismatch')
            source = db.execute('SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload,v.content_hash,a.fqdn FROM artifact_symbolic_expressions e JOIN physics_equation_candidates q USING(candidate_id) JOIN physics_ingest_snapshots s USING(snapshot_id) JOIN artifact_versions v ON v.version_id=e.version_id JOIN artifacts a ON a.artifact_id=e.artifact_id WHERE e.expression_id=%s FOR SHARE OF e,q,s,v,a', (proof['details']['source_expression_id'],)).fetchone()
            fresh = prepare_pdg_evidence(source, symbols)
            if fresh != source['evidence_json']['pdg_source_comparison'] or _digest(fresh) != proof['details']['source_comparison_sha256'] or source['content_hash'] != details['source_content_hash'] or source['version_id'] != executed['version_id']:
                raise ValueError('source provenance drift')
            dims = {r['symbol_name']: r['dim_signature'] for r in db.execute('SELECT symbol_name,dim_signature FROM artifact_symbolic_variables WHERE expression_id=%s FOR SHARE', (source['expression_id'],)).fetchall()}
            graph, mapping = build_residual_execution(deserialize_expr(source['sympy_srepr']), dims, source_version_id=source['version_id'], source_content_hash=source['content_hash'])
            if graph.model_dump(mode='json') != details['execution_graph'] or intake['content_hash'] != details['execution_graph_sha256']:
                raise ValueError('fresh source-to-implementation proof differs')
            document = db.execute('SELECT get_artifact_document(%s) AS d', (intake['fqdn'],)).fetchone()['d']
            if _artifact_document_to_cdg(document, version_id=str(intake['version_id']), content_hash=intake['content_hash'], require_execution_envelope=True) != graph:
                raise ValueError('stored execution graph differs')
            bindings = db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s FOR SHARE', (intake['version_id'],)).fetchall()
            if len(bindings) != 1 or bindings[0]['status'] != 'active' or bindings[0]['bound_artifact_fqdn'] != graph.nodes[0].matched_primitive or bindings[0]['bound_version_content_hash'] != intake['details']['provider_content_hash']:
                raise ValueError('provider binding mismatch')
            provider = db.execute("SELECT e.* FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN atoms legacy ON legacy.atom_id=a.artifact_id JOIN artifact_audit_evidence e ON e.version_id=v.version_id WHERE a.fqdn=%s AND v.content_hash=%s AND v.is_latest AND v.trust_tier=3 AND a.status='approved' AND a.is_publishable AND legacy.status='approved' AND legacy.is_publishable AND e.runner_version='local-physics-community-approval.v1' AND e.passed FOR SHARE OF a,v,legacy,e", (bindings[0]['bound_artifact_fqdn'], bindings[0]['bound_version_content_hash'])).fetchone()
            if not provider:
                raise ValueError('exact provider approval missing')
            if db.execute("SELECT 1 FROM artifact_audit_evidence WHERE version_id=ANY(%s) AND passed=false AND status IN ('failed','completed') LIMIT 1", ([intake['version_id'], provider['version_id']],)).fetchone():
                raise ValueError('failed version audit')
            deps = db.execute('SELECT * FROM artifact_dependencies WHERE dependent_version_id=%s FOR SHARE', (intake['version_id'],)).fetchall()
            if len(deps) != 1 or deps[0]['dependency_artifact_fqdn'] != source['fqdn'] or deps[0]['dependency_content_hash'] != source['content_hash'] or deps[0]['optional']:
                raise ValueError('mandatory source relation provenance missing')
            refs = db.execute('SELECT ref_id,ref_key,title,url,verified FROM artifact_references WHERE artifact_id=%s AND verified FOR SHARE', (provider['artifact_id'],)).fetchall()
            if not refs:
                raise ValueError('scientific reference missing')
            for ref in refs:
                ensure_row(db, 'artifact_references', {'artifact_id': intake['artifact_id'], 'ref_key': ref['ref_key']}, {'artifact_id': intake['artifact_id'], **ref, 'source': 'llm_extracted', 'confidence': 'high', 'relevance_note': 'Context for the shared electrical residual; source-specific equality and SI mapping checked separately. Does not establish physical validity.'})
            limitations = ['Tier 3 Community, automated review only.', 'Evaluates the signed LHS-minus-RHS residual; does not solve unknowns or establish physical validity.', 'Finite real broadcastable inputs in the declared SI units; floating overflow and invalid values are rejected.', 'Zero residual alone does not establish physical validity, including degenerate zero-current cases.', 'Shared providers avoid duplicate implementations; source-specific graphs preserve distinct provenance and variable mappings.']
            rollup = {'artifact_id': intake['artifact_id'], 'overall_verdict': 'acceptable_with_limits', 'structural_status': 'pass', 'runtime_status': 'pass', 'semantic_status': 'pass', 'developer_semantics_status': 'pass', 'review_status': 'approved', 'review_semantic_verdict': 'pass', 'review_developer_semantics_verdict': 'pass', 'trust_readiness': 'ready', 'review_limitations': limitations, 'review_required_actions': [], 'trust_blockers': [], 'acceptability_band': 'acceptable_with_limits', 'parity_coverage_level': 'positive_and_negative', 'parity_test_status': 'pass'}
            ensure_row(db, 'artifact_audit_rollups', {'artifact_id': intake['artifact_id']}, rollup)
            approval = {'publication_tier': 3, 'review_source': 'automated', 'content_hash': intake['content_hash'], 'execution_evidence_id': str(executed['evidence_id']), 'execution_evidence_sha256': _digest(details), 'provider_approval_id': str(provider['evidence_id']), 'provider_approval_sha256': _digest(provider['details']), 'source_comparison_sha256': _digest(fresh), 'source_symbol_bindings': mapping, 'source_dependency_review': 'Exact source relation and dimension-preserving implementation mapping reviewed as mandatory provenance; no physical validity transfer.', 'limitations': limitations}
            audit_id = uuid5(intake['version_id'], 'residual-community-approval.v1')
            counts['new_approvals'] += ensure_row(db, 'artifact_audit_evidence', {'evidence_id': audit_id}, {'evidence_id': audit_id, 'artifact_id': intake['artifact_id'], 'version_id': intake['version_id'], 'audit_type': 'semantic_audit', 'passed': True, 'status': 'completed', 'source_kind': 'automated', 'runner_version': 'residual-community-approval.v1', 'details': Jsonb(approval)})
            topo = build_published_cdg_projection(artifact={'artifact_id': str(intake['artifact_id']), 'fqdn': intake['fqdn']}, version={'version_id': str(intake['version_id']), 'content_hash': intake['content_hash']}, cdg=graph).topo_hash
            db.execute("UPDATE artifacts SET status='approved',is_publishable=true,verified_leaf_coverage=1,leaf_count=1,top_level_input_arity=%s,top_level_output_arity=1,topo_hash=%s,source_symbol='equation_residual_validator' WHERE artifact_id=%s", (len(graph.nodes[0].inputs), topo, intake['artifact_id']))
            db.execute('UPDATE artifact_versions SET is_latest=(version_id=%s) WHERE artifact_id=%s', (intake['version_id'], intake['artifact_id']))
            if not db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s', (intake['artifact_id'],)).fetchone():
                raise ValueError('approved residual graph not served')
            counts['verified'] += 1
        if not args.apply:
            db.rollback()
    print(json.dumps({'applied': args.apply, 'counts': dict(counts)}, indent=2))


if __name__ == '__main__':
    main()
