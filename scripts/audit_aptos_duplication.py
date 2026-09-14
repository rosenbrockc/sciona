"""Read-only catalog duplication review for the two proposed APTOS atoms."""
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
PATTERN = 'aptos|inception|seresnext|se.resnext|generalized.mean|gem.pool|smooth.l1|pseudo.label|label.refin|group.mean|fold.ensemble|population|cassava'
DECISIONS = {'contrails_prepare': 'Prepares thermal segmentation tensors and scoring masks, not the four ordinal/soft-target roles.', 'fold_ensemble_average': 'Arithmetic reduction is a compatible suboperation, but this atom accepts an anonymous rank-three matrix. It does not enforce eight architecture/replica identities, align private keys, train models or refine targets; no new generic averaging atom is proposed.', 'hubmap_pseudo_labels': 'Produces stitched binary segmentation masks with four-view inference, not unbounded scalar regression targets.', 'amex_prepare': 'Sequence categories and fold populations differ from four image label roles.', 'biosignal_sequence_prepare': 'Subject-separated signal/spectral inputs differ from RGB regression populations.', 'cassava_fold_plan': 'Validates five shared held-out folds and class complements, not stage-one/stage-two label-role separation.', 'cassava_train_ensemble': 'Trains CV classifiers and a B4 refit with frozen CropNet; no eight-model scalar GeM regression, ensemble teacher or supplemental refinement.', 'cornell_prepare': 'Audio and fold-lifecycle contract differs from image and ordinal refinement inputs.', 'future_sales_prepare': 'Entity/period contract differs from RGB populations and supplemental labels.', 'image_tabular_prepare': 'Image-plus-metadata grouped-fold contract differs from the four staged label roles.', 'march_prepare': 'Game-population contract differs from RGB regression and soft targets.', 'openvaccine_prepare': 'Member-split molecular regression recipe differs from the four image label roles.', 'otto_prepare': 'Nine-class tabular stacking, embedding and meta-fold contract differs from scalar image regression.', 'physical_operator_prepare': 'Periodic SI state contract differs from image labels and pseudo targets.', 'porto_prepare': 'Binary tabular populations and five denoising/neural controls differ from four image roles and eight GeM models.', 'santa_prepare': 'Replay and visible-history contract differs from staged image training.', 'santander_execute': 'Tabular binary pseudo-label feature regeneration and retraining differ from continuous image targets and native vision backbones.', 'santander_prepare': 'Binary tabular matrices, ten-fold branches and retained pseudo-query controls differ from four image label roles.', 'self_training_select_pseudo_labels': 'Selects confidence-threshold or top-k indices, not continuous teacher targets over every supplied added role.', 'extract_pseudo_labels': 'Selects confident positive/negative probability indices; no scalar teacher averaging or four-level group bounding.', 'tabular_ensemble_prepare': 'Tabular fold and training controls differ from image populations and source label roles.', 'text_pair_prepare': 'Text-pair population contract differs from RGB scalar regression.', 'video_multilabel_prepare': 'Video-feature multilabel contract differs from RGB scalar regression.', 'weighted_population_variance': 'Numerical mean/variance reduction has no ordinal groups, target clipping, identity validation or training lifecycle.', 'andriy_distribution_features': 'Window distribution features are not image label-role preparation or model training.', 'andriy_time_features': 'Signal derivative/time features are not image label-role preparation or model training.', 'channel_basic_statistics': 'Window/channel descriptive statistics do not implement any proposed provider boundary.', 'collect_population_scores': 'Concatenates three score/ID groups without training, averaging or supplemental target refinement.', 'ensemble_population_indices': 'Returns three fixed positions; no population validation or staged target construction.', 'ordered_prediction_ids': 'Binds clip counts and integer identities; no staged ordinal label roles.', 'select_ensemble_population': 'Selects one of three binary clip populations; no four-role soft-target lifecycle.', 'ensemble_family_inputs': 'Forwards distinct signal-family population inputs; no image-role validation or regression fitting.', 'pack_eleven_predictions': 'Packs eleven supplied scores and identity groups; no eight-model training or target refinement.', 'feng_fft_features': 'Signal FFT feature extraction does not implement an image population or training provider.', 'andriy_r_glm_segment_probabilities': 'Fits source binary signal-feature models for three populations and aggregates segment scores; different inputs, models and arithmetic from eight image regressors.', 'andriy_r_svm_segment_probabilities': 'Fits source binary signal-feature models for three populations and aggregates segment scores; different inputs, models and arithmetic from eight image regressors.', 'andriy_r_xgb_segment_probabilities': 'Fits source binary signal-feature models for three populations and aggregates segment scores; different inputs, models and arithmetic from eight image regressors.', 'andriy_documented_population': 'Builds signal clip/window features and binary normalized populations; does not implement RGB first-stage base plus averaging/group-bounded/pseudo roles.', 'andriy_population_inputs': 'Builds signal clip/window features and binary normalized populations; does not implement RGB first-stage base plus averaging/group-bounded/pseudo roles.', 'andriy_documented_populations': 'Builds signal clip/window features and binary normalized populations; does not implement RGB first-stage base plus averaging/group-bounded/pseudo roles.'}


def audit():
    directory = ROOT.parent / 'sciona-atoms-ml'
    parsed = _parse_registered_atoms(repo=ProviderRepo('sciona-atoms-ml', directory),
                                    artifact_root=directory / 'src/sciona/atoms')
    proposed = [row for row in parsed if row.import_module == 'sciona.atoms.ml.aptos_execution']
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
        'sha256': {'scripts/audit_aptos_duplication.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}


if __name__ == '__main__':
    report = audit()
    (ROOT / 'docs/reviews/competition_aptos_duplication_review.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS catalog duplication check:', len(report['reviewed_candidates']), 'semantic candidates reviewed')
