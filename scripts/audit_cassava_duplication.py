"""Read-only catalog duplication review for the two proposed Cassava atoms."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms

ROOT = Path(__file__).resolve().parents[1]
PATTERN = (r'cassava|cropnet|resnext|resnet|efficientnet|fold.plan|fold.contract|five.fold|'
           r'fold.ensemble|stratified.kfold|image.classif|vision.transformer|'
           r'cross.validation|ensemble_execute|fold_members|fold_assignment')
DECISIONS = {
    'efficientnet_backbone': 'Zero-valued feature-shape contract; no pretrained fitting, folds, refit or four-family inference.',
    'resnet_family_backbone': 'Zero-valued pooled feature contract; no classification head, pretrained fitting or four-family lifecycle.',
    'fold_ensemble_average': 'Reduces supplied matrices using arithmetic/geometric/rank averaging; no identity plan or model fitting. Float64 coercion differs from retained per-family precision.',
    'compute_stratified_kfold_indices': 'Generates splits from targets; does not validate caller-supplied identities, classes and five held-out assignments.',
    'partition_groups_to_folds': 'Generates group-balanced assignments; does not preserve and validate the supplied common fold plan.',
    'fit_cv_candidate': 'Fits and scores one estimator candidate; no family-specific losses, checkpoints, final B4 refit or frozen CropNet.',
    'generate_cartesian_grid': 'Enumerates parameter combinations; no fold-plan validation or training orchestration.',
    'sample_parameter_distributions': 'Samples search parameters; no fold-plan validation or training orchestration.',
    'tabular_ensemble_execute': 'Tabular fold-local models, OOF stacker and sigmoid calibration; different inputs, models and output arithmetic.',
    'cross_validation': 'BioSPPy stratified randomized split iterator; does not validate the prescribed five-fold image population.',
}


def audit():
    directory = ROOT.parent / 'sciona-atoms-ml'
    parsed = _parse_registered_atoms(repo=ProviderRepo('sciona-atoms-ml', directory),
                                    artifact_root=directory / 'src/sciona/atoms')
    proposed = [row for row in parsed if row.import_module == 'sciona.atoms.ml.cassava_execution']
    if len(proposed) != 2:
        raise ValueError('Expected two proposed provider atoms')
    with psycopg.connect(dotenv_values(ROOT / '.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        count = db.execute('SELECT count(*) FROM atoms').fetchone()[0]
        matches = db.execute("""SELECT DISTINCT a.fqdn FROM atoms a JOIN atom_versions v USING(atom_id)
            WHERE (v.content_hash = ANY(%s) OR v.fingerprint::text = ANY(%s))
            AND NOT a.fqdn = ANY(%s)""", ([r.content_hash for r in proposed],
                [r.fingerprint for r in proposed], [r.fqdn for r in proposed])).fetchall()
        if matches:
            raise ValueError('Existing content/fingerprint identity requires reuse review')
        rows = db.execute("""SELECT fqdn,import_module,source_symbol FROM atoms
            WHERE concat_ws(' ',fqdn,description) ~* %s AND NOT fqdn = ANY(%s)
            ORDER BY fqdn""", (PATTERN, [r.fqdn for r in proposed])).fetchall()
    reviewed = []
    for fqdn, module, symbol in rows:
        if symbol not in DECISIONS:
            raise ValueError('Unreviewed catalog candidate: ' + fqdn)
        function = inspect.unwrap(getattr(importlib.import_module(module), symbol))
        reviewed.append({'fqdn': fqdn, 'signature': str(inspect.signature(function)),
            'function_source_sha256': hashlib.sha256(inspect.getsource(function).encode()).hexdigest(),
            'decision': 'different_behavior', 'reason': DECISIONS[symbol]})
    return {'approved': False, 'catalog_mutations': 0, 'duplication_check_passed': True,
        'catalog_atoms_scanned': count, 'candidate_pattern': PATTERN,
        'exact_content_or_fingerprint_duplicates': 0, 'reviewed_candidates': reviewed,
        'scope': 'All catalog atom versions checked for exact identity; named/description candidates reviewed for semantic overlap. This is a catalog snapshot, not a universal semantic-equivalence proof.',
        'proposed': {r.fqdn: {'content_hash': r.content_hash, 'fingerprint': r.fingerprint} for r in proposed},
        'sha256': {'scripts/audit_cassava_duplication.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}


if __name__ == '__main__':
    report = audit()
    (ROOT / 'docs/reviews/competition_cassava_duplication_review.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS catalog duplication check:', len(report['reviewed_candidates']), 'semantic candidates reviewed')
