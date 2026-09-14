#!/usr/bin/env python3
"""Synthetic comparison with pinned competition and historical pyRiemann classes."""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import scipy
from joblib import Parallel, delayed, parallel_backend
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import shrunk_covariance
from sklearn.pipeline import make_pipeline
from sciona.atoms.riemannian_bci.covariance_features import source_tangent
from sciona.atoms.riemannian_bci.covariance_features.delay_correlation import channel_delay_correlations
from sciona.atoms.riemannian_bci.covariance_features.frequency_coherence import frequency_band_coherence

HASHES = {
    'models.py': '38905d4d1bd61560b6c273876f060884c47156eedc500d9085e9ca5fe7c1889b',
    'pyriemann_estimation.py': '6048dd6b1ad5f59b866088aa0d781ecf5d0dd62d917a12dcbd1e61c740600a22',
    'pyriemann_utils_base.py': 'cb97d9806133344b19e48ad23d54050173c72620a0607de372a40a9f3a391c12',
    'pyriemann_utils_mean.py': '1545419ff4dfc1ac2a0d60ed5edb264a1ec096cde7f6640a809e3c308e9534b2',
    'pyriemann_utils_tangentspace.py': 'f18370271f9387da2f527bd34e24b5cfc5b1e238b88bd51d55c73ce029376edb',
    'pyriemann_tangentspace.py': '720a810479e01d7a2e2e3e1c9a0f3817f17bb784db1b306090804a7c80ca4b8b',
}


def load_reference(directory):
    namespace = dict(np=np, numpy=np, scipy=scipy, sp=scipy, BaseEstimator=BaseEstimator,
                     TransformerMixin=TransformerMixin, make_pipeline=make_pipeline,
                     shrunk_covariance=shrunk_covariance, Parallel=Parallel, delayed=delayed)
    selections = {
        'pyriemann_utils_base.py': {'_matrix_operator', 'logm', 'expm', 'invsqrtm'},
        'pyriemann_utils_mean.py': {'_get_sample_weight', 'mean_logeuclid', 'mean_identity', 'mean_covariance'},
        'pyriemann_utils_tangentspace.py': {'tangent_space'},
        'pyriemann_tangentspace.py': {'TangentSpace'},
        'pyriemann_estimation.py': {'Shrinkage'},
        'models.py': {'fit_one_ts', 'apply_one_ts', 'CoherenceToTangent'},
    }
    for name, selected in selections.items():
        source = (directory / name).read_bytes()
        if hashlib.sha256(source).hexdigest() != HASHES[name]:
            raise ValueError('pinned source hash mismatch: ' + name)
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in selected]
        if len(tree.body) != len(selected):
            raise ValueError('missing reference definitions')
        exec(compile(tree, name, 'exec'), namespace)
    namespace['mean_methods'] = {name: namespace['mean_' + name] for name in ['identity', 'logeuclid']}
    return namespace['CoherenceToTangent']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    reference = load_reference(args.reference_dir)
    maximum = 0.
    cases = 0
    for seed in range(5):
        rng = np.random.default_rng(700 + seed)
        windows = rng.normal(size=(9, 3, 1800))
        branches = [
            (channel_delay_correlations(windows, [1, 2, 4, 8, 16, 32, 64], 2), 'identity', False),
            (frequency_band_coherence(windows, [[.1, 4], [4, 8], [8, 15], [15, 30], [30, 90], [90, 170]], 400., 512, .5), 'logeuclid', True),
        ]
        for matrices, metric, update in branches:
            train, prediction = matrices[:5], matrices[5:]
            model = reference(metric=metric, tsupdate=update, n_jobs=8)
            with parallel_backend('threading'):
                expected = model.fit_transform(train), model.transform(prediction)
            actual = source_tangent.channel_tangent_features(train, prediction, metric, update)
            for got, want in zip(actual, expected):
                np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-11)
                maximum = max(maximum, float(np.max(abs(got - want))))
            cases += 1
    report = {
        'synthetic_only': True, 'cases_passed': cases, 'maximum_absolute_error': maximum,
        'competition_revision': '00f937cc7710977dc812d9fc675864e2b8288658',
        'pyriemann_revision': '38c3ec1fae7b2d414ebdc669938b4d00a92d50ee',
        'source_hashes': HASHES,
        'implementation_hash': hashlib.sha256(Path(inspect.getfile(source_tangent)).read_bytes()).hexdigest(),
        'validation_script_hash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reference_loading': 'Unmodified selected AST definitions; imports injected; metric dispatch restricted to identity/logeuclid; source-configured n_jobs=8 using threading backend. The unused source n_jobs=1 transform path reverses arguments and fails.',
        'limitations': 'Same installed NumPy/SciPy/sklearn libraries; contemporaneous pyRiemann revision, not proven original environment. No classifier or predictive-performance validation.'
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
