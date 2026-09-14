"""Check current Flavours evidence and expose outstanding publication gates."""
import hashlib
import base64
import importlib.metadata as metadata
import json
from pathlib import Path

from scripts.build_flavours_execution_graph import build_graph
from sciona.services.execution_graph_codec import encode_execution_graph

ROOT = Path(__file__).resolve().parents[1]
SOURCE_VERSION = '6256d5b1-d742-53b1-9ff4-c2c7265ce981'
SOURCE_HASH = '1f7003883cd8067a1fcaa170601336d99e2911e2024a0590eb41425757dc0539'
STAGE_MAPPING = {
    'feature_selection_safe': 'Replace unsupported three-safe-variable selection with caller-ordered46base inputs,one source exclusion and physical mass-proxy construction.',
    'flatness_constrained_gradient_boosting': 'Replace unsupported flatness boosting with35direct-mass regressors,30classifier fits and50neural fits under the recovered notebook recipe.',
    'noise_injection_decorrelation': 'No source noise injection; use the source nonlinear six-component blend and preserve failed diagnostics.',
    'compute_ks_agreement': 'Reuse existing weighted KS atom on separately predicted agreement populations.',
    'compute_cvm_mass_decorrelation': 'Reuse200-neighbour/50-step CvM on separately predicted correlation population.',
    'roc_auc_truncated_weighted': 'Reuse source truncated weighting after strict quality>0.4; OOF discrimination remains diagnostic.',
}
LIMITATIONS = [
    'Automated Community Tier3 only; no human-reviewed Tier1 or usage-qualified Tier2 claim.',
    'Independent mathematical implementation; no winning notebook code or task data bundled.',
    'Notebook direct-mass repeated cross-fitting and classifier full refits resolve conflicting PDF prose.',
    'Modern exact-tree CPU XGBoost/base_score0.5 and NumPy float64 neural runtime; historical numerical parity unproven.',
    'Caller establishes population identity/disjointness,physical units and source base-column ordering/exclusion.',
    'Mass cross-fitting is not nested within classifier folds; OOF discrimination is diagnostic only.',
    'Synthetic correlation diagnostic failed and was returned faithfully; no competition accuracy or general constraint-satisfaction claim.',
    'Qualified runtime requires exact reviewed OpenCV overlay and native OpenMP runtime; shared conflicting cv2 installation alone is unqualified.',
    'Source weights are not downloaded or redistributed; each call retrains all115models.',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    def read(name):
        return json.loads((ROOT / 'docs/reviews' / ('competition_flavours_'+name+'.json')).read_text())
    runtime = read('graph_validation')
    expected = dict(passed=True,synthetic_only=True,catalog_mutations=0,model_fits=115,
                    tree_boosting_rounds=111458,neural_epochs=150000,
                    neural_optimizer_updates=600000,neural_checkpoint_replays_exact=True)
    if any(runtime.get(k) != v for k,v in expected.items()):
        raise ValueError('Full graph training evidence incomplete')
    for name,digest in runtime['source_sha256'].items():
        if sha(ROOT/name) != digest: raise ValueError('Runtime source drift: '+name)
    provider = ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/flavours_execution.py'
    graph_hash = encode_execution_graph(build_graph())[0]
    graph = runtime['graph_execution']
    if (graph['serialized_graph_sha256'] != graph_hash or graph['provider_sha256'] != sha(provider)
            or graph['actual_runner_nodes'] != 1 or not graph['codec_roundtrip']):
        raise ValueError('Provider/graph identity mismatch')
    qualified = read('overlay_qualification')
    for name,digest in qualified['sha256'].items():
        if sha(ROOT/name) != digest: raise ValueError('Qualification evidence drift: '+name)
    environment = read('environment_audit')
    if environment['failures'] or not environment['dependency_constraints_satisfied']:
        raise ValueError('Dependency constraint failure')
    for name,item in environment['packages'].items():
        if metadata.version(name) != item['version']: raise ValueError('Dependency version drift: '+name)
    # Metadata versions alone cannot detect the shared cv2 package collision.
    opencv = metadata.distribution('opencv-python')
    for entry in opencv.files or []:
        if not entry.hash:
            continue
        path = Path(opencv.locate_file(entry))
        if not path.is_file(): raise ValueError('Qualified OpenCV file missing')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, entry.hash.mode).digest()
        if base64.urlsafe_b64encode(digest).decode().rstrip('=') != entry.hash.value:
            raise ValueError('Qualified OpenCV overlay is not active or has changed')
    if not read('package_file_audit')['all_recorded_hashes_match']:
        raise ValueError('Dependency integrity evidence failed')
    duplicate = read('duplication_review')
    if duplicate['exact_content_or_fingerprint_duplicates']:
        raise ValueError('Unresolved exact provider duplication')
    publisher = read('publisher_payload_validation')
    if not publisher['all_compared_payloads_match'] or not publisher['all_install_scheme_files_compared']:
        raise ValueError('Publisher payload verification incomplete')
    native = read('native_publisher')
    if not native.get('relocation_verified') or native.get('codesign_strict_verification') != 'passed':
        raise ValueError('Native publisher relocation evidence incomplete')
    if sha(Path('/opt/homebrew/opt/libomp/lib/libomp.dylib')) != native['installed_library_sha256']:
        raise ValueError('Native runtime drift')
    reference = read('reference_scope_review')
    if reference['source_attribution_status'] != 'conditional_pass' or not reference['conditions']:
        raise ValueError('Conditional attribution review required')
    for name,digest in reference['sha256'].items():
        if sha(ROOT/name) != digest: raise ValueError('Attribution evidence drift: '+name)
    return dict(approved=False,catalog_mutations=0,target_tier=3,
                structural_status='pass',runtime_status='pass',publication_ready=False,
                serialized_graph_sha256=graph_hash,provider_sha256=sha(provider),
                stage_mapping=STAGE_MAPPING,limitations=LIMITATIONS,reference_conditions=reference['conditions'],
                required_actions=['Finalize semantic approval with explicit conditions',
                                  'Pass negative publication gates,rollback,atomic apply,idempotency and served verification'])


def final_review():
    readiness = audit()
    from scripts.prepare_flavours_publication import prepare
    plan = prepare()
    conditions = readiness['limitations'] + readiness['reference_conditions']
    if plan['review_conditions'] != conditions or len(plan['providers']) != 1:
        raise ValueError('Plan scope differs from reviewed implementation')
    paths = [ROOT/'scripts/review_flavours_execution.py', ROOT/'scripts/prepare_flavours_publication.py',
             ROOT/'scripts/build_flavours_execution_graph.py',ROOT/'scripts/flavours_graph_execution.py',
             ROOT/'scripts/validate_flavours_graph.py', ROOT/'requirements/flavours-reference.txt',
             ROOT/'requirements/flavours-reference-macos-arm64.lock']
    suffixes = ['graph_validation','overlay_qualification','environment_audit','package_file_audit',
                'duplication_review','reference_scope_review','publisher_payload_validation',
                'native_publisher','dependency_notices','source_triage','neural_reference']
    paths += [ROOT/'docs/reviews'/('competition_flavours_'+s+'.json') for s in suffixes]
    paths += list((ROOT/'sciona').glob('flavours_*.py'))
    return dict(approved=False,catalog_mutations=0,proposed_tier=3,review_source='automated',
                verdict='acceptable_with_limits',semantic_status='conditional_pass',
                developer_semantics_status='conditional_pass',ready_for_transaction=True,
                conditions=conditions,source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,
                stage_mapping=STAGE_MAPPING,serialized_graph_sha256=readiness['serialized_graph_sha256'],
                provider_sha256=readiness['provider_sha256'],
                evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths})


if __name__ == '__main__':
    result=audit()
    (ROOT/'docs/reviews/competition_flavours_readiness.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
