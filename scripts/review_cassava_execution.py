"""Assemble current Tier 3 evidence without authorizing catalog publication.

Missing trained execution or publication evidence remains an explicit pending
action. A structural pass alone never turns this report into an approval.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_VERSION = '735c71d1-6f8a-565b-a5c3-f1bcf3a89285'
SOURCE_HASH = 'c74ce39f2cea94fe168690a36ac59be792dad72a02f705424dea7a8f24cc2218'
STAGE_MAPPING = {
    'image_augmentation': 'Separate source-comparison evidence for ResNeXt, ViT, B4 training and B4/CropNet inference; caller supplies aligned prepared B4 images.',
    'model_backbone': 'Three trainable pretrained families (ResNeXt, ViT, B4) plus frozen official CropNet; intake omits two families.',
    'robust_loss': 'Family-specific objectives replace the unsupported universal robust-loss claim: ResNeXt CE, ViT bi-tempered loss and source B4 focal behavior.',
    'pseudo_labeling': 'Excluded: not established by inspected winning writeup and pinned notebook sources.',
    'ensembling': 'Fivefold means for ResNeXt/ViT, separate B4 full refit with ten-view inference, CropNet unknown-mass distribution, then (ViT+ResNeXt)/2+B4+CropNet.',
}
LIMITATIONS = [
    'Automated Tier 3 Community candidate; no Tier 1 human certification or Tier 2 usage qualification.',
    'Corrected CPU reconstruction; historical TPU execution, exact historical timm runtime and leaderboard equivalence are unproven.',
    'Caller establishes physical-image identity and prepared-view provenance. Exact decoded duplicates and key mismatch reject; near-duplicate identities are not inferred.',
    'ViT validation is kept disjoint from training; B4 final normalization explicitly adapts the full supplied training population.',
    'Explicit historical-reference timm weights and official CropNet are used; exact winner cached artifacts are not universally proven.',
    'Synthetic full-resolution execution with batch one and zero torch workers does not establish competitive accuracy or original population-scale performance.',
    'Caller supplies local reference artifacts and manages private checkpoint outputs; no model artifact is bundled by these provider wrappers.',
    'Final five-class scores sum to three and are not probabilities. Each invocation retrains the complete workflow.',
]


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
        return json.loads((ROOT / 'docs/reviews' / ('competition_cassava_' + suffix + '.json')).read_text())
    source = read('source_review')
    if (source['source_version_id'], source['source_content_hash']) != (SOURCE_VERSION, SOURCE_HASH):
        raise ValueError('Source intake identity changed')
    check_hashes(source['sha256'])
    structural = read('provider_structure')
    contracts = read('provider_contract_validation')
    duplicates = read('duplication_review')
    for report in [structural, contracts, duplicates]:
        check_hashes(report['sha256'])
    if not structural['passed'] or contracts['tests_passed'] < 7 or not contracts['actual_runner_routing']:
        raise ValueError('Provider structure or routing evidence missing')
    provider = ROOT.parent / 'sciona-atoms-ml/src/sciona/atoms/ml/cassava_execution.py'
    if sha(provider) != structural['provider_sha256'] or sha(provider) != contracts['provider_sha256']:
        raise ValueError('Provider implementation changed')
    if not duplicates['duplication_check_passed'] or duplicates['exact_content_or_fingerprint_duplicates']:
        raise ValueError('Duplication review failed')
    environment = read('publisher_payload_validation')
    check_hashes(environment['sha256'])
    if not environment['all_compared_payloads_match'] or not environment['all_install_scheme_files_compared']:
        raise ValueError('Publisher payload verification failed')
    pending, evidence = [], {}
    for suffix in ['pipeline_validation', 'graph_execution_validation']:
        path = ROOT / 'docs/reviews' / ('competition_cassava_' + suffix + '.json')
        if not path.exists():
            pending.append('Complete ' + suffix)
            continue
        report = json.loads(path.read_text())
        check_hashes(report['sha256'])
        required = {'passed': True, 'synthetic_only': True, 'trainable_model_fits': 16,
            'frozen_models': 1, 'torch_folds_per_family': 5, 'resnext_epochs_per_fold': 15,
            'vit_epochs_per_fold': 10, 'efficientnet_refit_epochs': 14,
            'torch_checkpoint_prediction_replays_exact': True, 'final_scalar_assembly_exact': True}
        if any(report.get(k) != v for k, v in required.items()):
            raise ValueError('Incomplete trained execution evidence: ' + suffix)
        if len(report['efficientnet_cv_epochs']) != 5 or any(not 1 <= n <= 20 for n in report['efficientnet_cv_epochs']):
            raise ValueError('Five EfficientNet CV lifecycles required')
        if suffix == 'graph_execution_validation':
            if (report.get('actual_runner_nodes') != 2 or not report.get('graph_codec_roundtrip')
                    or report['serialized_graph_sha256'] != structural['serialized_graph_sha256']
                    or report['provider_sha256'] != sha(provider)):
                raise ValueError('Trained graph differs from reviewed providers')
        evidence[suffix] = sha(path)
    references = read('reference_scope_review')
    check_hashes(references['sha256'])
    if (references['source_attribution_status'] != 'conditional_pass'
            or references['model_redistribution_approved'] is not False
            or not references['conditions']):
        raise ValueError('Source attribution or model-reference conditions missing')
    pending += ['Incorporate reference-scope conditions into final semantic approval',
                'Pass transactional publication gates and verify served graph execution']
    return {'approved': False, 'catalog_mutations': 0, 'target_tier': 3,
        'source_version_id': SOURCE_VERSION, 'source_content_hash': SOURCE_HASH,
        'structural_status': 'pass', 'runtime_status': 'pass' if len(evidence) == 2 else 'pending',
        'semantic_status': 'pending_final_review', 'developer_semantics_status': 'pending_final_review',
        'stage_mapping': STAGE_MAPPING, 'limitations': LIMITATIONS,
        'reference_conditions': references['conditions'],
        'required_actions': pending, 'trained_execution_evidence': evidence,
        'provider_sha256': sha(provider), 'serialized_graph_sha256': structural['serialized_graph_sha256'],
        'sha256': {str(p.relative_to(ROOT)): sha(p) for p in [
            ROOT / 'scripts/review_cassava_execution.py',
            ROOT / 'docs/reviews/competition_cassava_source_review.json',
            ROOT / 'docs/reviews/competition_cassava_provider_structure.json',
            ROOT / 'docs/reviews/competition_cassava_provider_contract_validation.json',
            ROOT / 'docs/reviews/competition_cassava_duplication_review.json',
            ROOT / 'docs/reviews/competition_cassava_reference_scope_review.json',
            ROOT / 'docs/reviews/competition_cassava_publisher_payload_validation.json']}}


def final_review():
    """Qualify the reviewed scope for a transaction; never publish by itself."""
    readiness = audit()
    if readiness['runtime_status'] != 'pass':
        raise ValueError('Both standalone and actual trained graph evidence are required')
    from scripts.audit_cassava_provider_contract import audit as structure_audit
    from scripts.audit_cassava_duplication import audit as duplication_audit
    from scripts.prepare_cassava_publication import prepare
    structure = structure_audit()
    stored = json.loads((ROOT / 'docs/reviews/competition_cassava_provider_structure.json').read_text())
    if structure != stored:
        raise ValueError('Provider structural review differs from current registration')
    duplication = duplication_audit()
    stored = json.loads((ROOT / 'docs/reviews/competition_cassava_duplication_review.json').read_text())
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
        (ROOT / 'docs/reviews/competition_cassava_semantic_review.json').write_text(json.dumps(report, indent=2) + '\n')
        print('PASS automated Tier 3 conditional semantic review; catalog unchanged')
        raise SystemExit(0)
    report = audit()
    (ROOT / 'docs/reviews/competition_cassava_execution_readiness.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'structural_status': report['structural_status'],
                      'runtime_status': report['runtime_status'], 'required_actions': report['required_actions']}))
