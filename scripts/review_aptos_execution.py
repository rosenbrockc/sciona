"""Validate APTOS conditional Tier 3 evidence without catalog mutations."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_VERSION = '94df8d52-0272-5d0b-b84f-1f150d8e11f0'
SOURCE_HASH = '02129e9c5bef75cf05490e52f2291f5d336d70a92e114d022a1b635ad35b3f58'
STAGE_MAPPING = {'image_standardization': 'Plain full-image resizing replaces unsupported retinal cropping/local-color subtraction. Reference augmentation and publisher normalization are explicit choices.', 'cnn_backbone': 'Eight models: two each of Inception-ResNet-v2, Inception-v4, SE-ResNeXt-50 and SE-ResNeXt-101, with trainable GeM and scalar regression head. EfficientNet substitution excluded.', 'quadratic_weighted_kappa_qwk_loss': 'Smooth-L1 regression replaces unsupported differentiable-QWK training loss. QWK motivates final fixed ordinal thresholds, not the training objective.', 'pre_training_fine_tuning': 'All pretrained backbone parameters train in stage one; an ensemble teacher builds three added target roles, and each own model continues ten epochs. Four-level grouping differs from five-level averaging labels.', 'threshold_calibration': 'Fixed manually selected cuts 0.7,1.5,2.5,3.5 replace unsupported Nelder-Mead optimization. Equality convention is explicit.'}
LIMITATIONS = ['Automated Community Tier 3 reference; no human-reviewed Tier 1 or usage-qualified Tier 2 designation.', 'Corrected eight-model CPU reconstruction; original winner runtime, artifact identity and leaderboard equivalence remain unproven.', 'First-stage budgets in five-epoch increments, Adam settings, ensemble teacher and fresh second-stage optimizer are explicit reference choices.', 'Augmentation units/distributions/order, bilinear resize, publisher normalization, scalar head and single-view inference are explicit choices.', 'Symmetric group bounds and threshold equality convention are explicit; the source establishes only the demonstrated lower-bound example.', 'Caller establishes population-role provenance and physical-image identity; supplied-label queries reject, while pseudo-label queries are deliberately transductive.', 'Synthetic full-resolution execution uses five first-stage and ten second-stage epochs per model; original population-scale performance is unproven.', 'Local external model artifacts require caller use rights. Each call retrains; temporary checkpoints are verified and removed.']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(hashes):
    if not hashes:
        raise ValueError('Required evidence hashes missing')
    for name, digest in hashes.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha(path) != digest:
            raise ValueError('Evidence drift: ' + name)


def audit():
    def read(suffix):
        return json.loads((ROOT / 'docs/reviews' / ('competition_aptos_' + suffix + '.json')).read_text())
    suffixes = ['source_review', 'provider_structure', 'provider_contract_validation',
        'duplication_review', 'reference_scope_review', 'publisher_payload_validation',
        'environment_audit', 'package_file_audit', 'pretrainedmodels_source_origin']
    reports = {name: read(name) for name in suffixes}
    for report in reports.values():
        if 'sha256' in report:
            check_hashes(report['sha256'])
    source = reports['source_review']
    if (source['source_version_id'], source['source_content_hash'], source['stage_mapping']) != (SOURCE_VERSION, SOURCE_HASH, STAGE_MAPPING):
        raise ValueError('Source identity or semantic stage mapping differs')
    structure = reports['provider_structure']
    contracts = reports['provider_contract_validation']
    provider = ROOT.parent / 'sciona-atoms-ml/src/sciona/atoms/ml/aptos_execution.py'
    if not structure['passed'] or contracts['tests_passed'] != 6 or not contracts['actual_runner_routing']:
        raise ValueError('Provider registration or routing proof missing')
    if sha(provider) != structure['provider_sha256'] or sha(provider) != contracts['provider_sha256']:
        raise ValueError('Provider source differs')
    duplication = reports['duplication_review']
    if not duplication['duplication_check_passed'] or duplication['exact_content_or_fingerprint_duplicates']:
        raise ValueError('Duplication review failed')
    environment = reports['environment_audit']
    if not environment['dependency_constraints_satisfied'] or environment['failures']:
        raise ValueError('Dependency constraints failed')
    for name, package in environment['packages'].items():
        if metadata.version(name) != package['version']:
            raise ValueError('Installed dependency version changed: ' + name)
    if not reports['package_file_audit']['all_recorded_hashes_match']:
        raise ValueError('Installed distribution RECORD mismatch')
    payload = reports['publisher_payload_validation']
    if not payload['all_compared_payloads_match'] or not payload['all_install_scheme_files_compared']:
        raise ValueError('Publisher payload comparison incomplete')
    origin = reports['pretrainedmodels_source_origin']
    if origin['different_or_missing'] or len(origin['matched_installed_python_files']) != 27:
        raise ValueError('Source-only dependency correspondence incomplete')
    references = reports['reference_scope_review']
    if references['source_attribution_status'] != 'conditional_pass' or references['model_redistribution_approved'] is not False or not references['conditions']:
        raise ValueError('Reference scope conditions missing')
    evidence = {}
    for suffix in ['pipeline_validation', 'graph_pipeline_validation']:
        report = read(suffix)
        check_hashes(report['sha256'])
        required = {'passed': True, 'synthetic_only': True, 'model_count': 8,
            'training_stage_fits': 16, 'first_stage_epochs_per_model': 5,
            'second_stage_epochs_per_model': 10, 'optimizer_updates': 280,
            'reference_augmentation_used': True, 'checkpoint_replays_exact': True,
            'temporary_checkpoints_removed': True, 'target_role_oracle_passed': True,
            'final_ensemble_and_threshold_oracle_passed': True}
        if any(report.get(key) != value for key, value in required.items()):
            raise ValueError('Incomplete native training evidence: ' + suffix)
        if suffix == 'graph_pipeline_validation':
            graph = report['graph_execution']
            if (graph['actual_runner_nodes'] != 2 or not graph['codec_roundtrip']
                    or graph['serialized_graph_sha256'] != structure['serialized_graph_sha256']
                    or graph['provider_sha256'] != sha(provider)):
                raise ValueError('Trained graph/provider identity mismatch')
        evidence[suffix] = sha(ROOT / 'docs/reviews' / ('competition_aptos_' + suffix + '.json'))
    return {'approved': False, 'catalog_mutations': 0, 'target_tier': 3,
        'source_version_id': SOURCE_VERSION, 'source_content_hash': SOURCE_HASH,
        'structural_status': 'pass', 'runtime_status': 'pass',
        'semantic_status': 'pending_final_review', 'developer_semantics_status': 'pending_final_review',
        'stage_mapping': STAGE_MAPPING, 'limitations': LIMITATIONS,
        'reference_conditions': references['conditions'],
        'required_actions': ['Bind conditions into a final semantic review and deterministic publication plan',
            'Pass transactional publication gates and verify served execution'],
        'trained_execution_evidence': evidence, 'provider_sha256': sha(provider),
        'serialized_graph_sha256': structure['serialized_graph_sha256'],
        'sha256': {str(p.relative_to(ROOT)): sha(p) for p in [
            ROOT / 'scripts/review_aptos_execution.py',
            *[ROOT / 'docs/reviews' / ('competition_aptos_' + suffix + '.json') for suffix in suffixes]]}}

def final_review():
    """Qualify the reviewed scope for a transaction; never publish by itself."""
    readiness = audit()
    if readiness['runtime_status'] != 'pass':
        raise ValueError('Both standalone and actual trained graph evidence are required')
    from scripts.audit_aptos_provider_contract import audit as structure_audit
    from scripts.audit_aptos_duplication import audit as duplication_audit
    from scripts.prepare_aptos_publication import prepare
    structure = structure_audit()
    stored = json.loads((ROOT / 'docs/reviews/competition_aptos_provider_structure.json').read_text())
    if structure != stored:
        raise ValueError('Provider structural review differs from current registration')
    duplication = duplication_audit()
    stored = json.loads((ROOT / 'docs/reviews/competition_aptos_duplication_review.json').read_text())
    for key in ['proposed', 'reviewed_candidates', 'exact_content_or_fingerprint_duplicates']:
        if duplication[key] != stored[key]:
            raise ValueError('Catalog duplication evidence changed')
    plan = prepare()
    conditions = LIMITATIONS + readiness['reference_conditions']
    if (plan['review_conditions'] != conditions
            or plan['execution']['content_hash'] != readiness['serialized_graph_sha256']
            or any(row['target_trust_tier'] != 3 for row in plan['providers'])
            or plan['execution']['target_trust_tier'] != 3):
        raise ValueError('Publication scope or tier differs from reviewed candidate')
    # Transaction and post-publication serving checks are subsequent gates.
    # They are not silently converted into completed review actions here.
    return {'approved': False, 'catalog_mutations': 0, 'proposed_tier': 3,
        'review_source': 'automated', 'verdict': 'acceptable_with_limits',
        'structural_status': 'pass', 'runtime_status': 'pass',
        'semantic_status': 'conditional_pass', 'developer_semantics_status': 'conditional_pass',
        'ready_for_transaction': True, 'conditions': conditions,
        'source_version_id': SOURCE_VERSION, 'source_content_hash': SOURCE_HASH,
        'stage_mapping': STAGE_MAPPING,
        'serialized_graph_sha256': readiness['serialized_graph_sha256'],
        'provider_sha256': readiness['provider_sha256'],
        'evidence_sha256': readiness['sha256'],
        'trained_execution_evidence': readiness['trained_execution_evidence'],
        'remaining_publication_actions': ['Pass transactional publication gates',
            'Verify served graph execution']}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final', action='store_true', help='Require complete trained graph evidence')
    args = parser.parse_args()
    if args.final:
        report = final_review()
        (ROOT / 'docs/reviews/competition_aptos_semantic_review.json').write_text(json.dumps(report, indent=2) + '\n')
        print('PASS automated Tier 3 conditional semantic review; catalog unchanged')
        raise SystemExit(0)
    report = audit()
    (ROOT / 'docs/reviews/competition_aptos_execution_readiness.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'structural_status': report['structural_status'],
                      'runtime_status': report['runtime_status'], 'required_actions': report['required_actions']}))
