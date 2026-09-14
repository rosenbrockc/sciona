#!/usr/bin/env python3
"""Synthetic parity for source RelativeLogPower and its full bagged classifier."""
import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import numpy as np
import yaml
from scipy.signal import welch
from sklearn.base import BaseEstimator, TransformerMixin
from sciona.atoms.riemannian_bci.covariance_features import relative_power
from sciona.atoms.ml.xgboost.competition_bagging import bagged_window_probabilities
from sciona.atoms.riemannian_bci.signal_processing import feature_vectorization
from validate_frequency_coherence import SOURCE_HASH
from validate_bci_predictions import reference_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source = (args.reference_dir / 'preproc.py').read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_HASH:
        raise ValueError('pinned source mismatch')
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in {'relative_log_power', 'RelativeLogPower'}]
    namespace = dict(np=np, welch=welch, BaseEstimator=BaseEstimator, TransformerMixin=TransformerMixin)
    exec(compile(tree, 'pinned_relative_power.py', 'exec'), namespace)
    vectorizer_path = args.reference_dir / 'mne_transformer_013.py'
    vectorizer_hash = '469718ad7f8e26c0bb31ec20ab71d8d223a4a84f091bfdde01b4f0acee54604a'
    if hashlib.sha256(vectorizer_path.read_bytes()).hexdigest() != vectorizer_hash:
        raise ValueError('pinned historical vectorizer mismatch')
    vectorizer_tree = ast.parse(vectorizer_path.read_bytes())
    vectorizer_tree.body = [n for n in vectorizer_tree.body if isinstance(n, ast.ClassDef) and n.name == 'Vectorizer']
    exec(compile(vectorizer_tree, 'pinned_mne_vectorizer.py', 'exec'), namespace)
    bands = [[.1, 4], [4, 8], [8, 15], [15, 30], [30, 90], [90, 170]]
    cases, classifier_cases, maximum = 0, 0, 0.
    config_path = args.reference_dir / 'Alex_Gilberto_models_relative_log_power_XGB.yml'
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != '74b55594ae46d9d756ecaedf8a9e7524be70a37e21a3bddf968e02673d0014ce':
        raise ValueError('pinned source model configuration mismatch')
    config = yaml.safe_load(config_path.read_text())
    for seed in range(5):
        rng = np.random.default_rng(1200 + seed)
        windows = rng.normal(size=(28, 3, 1800))
        for overlap in [0., .25, .5]:
            expected = namespace['RelativeLogPower'](window=512, overlap=overlap, fs=400, frequencies=bands).fit_transform(windows)
            actual = relative_power.relative_log_band_power(windows, bands, 400., 512, overlap)
            np.testing.assert_array_equal(actual, expected)
            maximum = max(maximum, float(np.max(abs(actual - expected))))
            cases += 1
            if overlap == .25 and seed < 3:
                labels = np.tile([0, 1], 6)
                indices = np.repeat(np.arange(12), 2)
                vectorizer = namespace['Vectorizer']()
                reference_train = vectorizer.fit_transform(expected[:24])
                reference_prediction = vectorizer.transform(expected[24:])
                model = reference_model(config)
                model.fit(reference_train, labels.repeat(2))
                expected_scores = model.predict_proba(reference_prediction)[:, 1]
                training_features, prediction_features = feature_vectorization.vectorize_feature_partitions(actual[:24], actual[24:])
                np.testing.assert_array_equal(training_features, reference_train)
                np.testing.assert_array_equal(prediction_features, reference_prediction)
                scores = bagged_window_probabilities(training_features, indices, labels, prediction_features, 10)
                np.testing.assert_array_equal(scores, expected_scores)
                classifier_cases += 1
    report = dict(synthetic_only=True, feature_cases_passed=cases, full_classifier_cases_passed=classifier_cases,
        maximum_feature_absolute_error=maximum, source_commit='00f937cc7710977dc812d9fc675864e2b8288658',
        source_hashes={'preproc.py': SOURCE_HASH, config_path.name: hashlib.sha256(config_path.read_bytes()).hexdigest(), vectorizer_path.name: vectorizer_hash},
        implementation_hash=hashlib.sha256(Path(inspect.getfile(relative_power)).read_bytes()).hexdigest(),
        vectorizer_implementation_hash=hashlib.sha256(Path(inspect.getfile(feature_vectorization)).read_bytes()).hexdigest(),
        validation_script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        library_versions={name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'scikit-learn', 'xgboost']},
        limitations='Same-library source numerical parity and modern model configuration only; no historical-library or predictive-performance claim. Vectorizer reference is contemporaneous MNE v0.13, not a proven original environment lock. No full CDG or final eleven-model ensemble executed.')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
