#!/usr/bin/env python3
"""Synthetic parity with historical conditional AR fit/bse and competition ARError."""
import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.tsatools import lagmat, add_trend
from statsmodels.tsa.vector_ar import util
from sciona.atoms.riemannian_bci.signal_processing import ar_standard_errors

HASHES = {'preproc.py': '1cfbc71faa230e1e4ffb72f27604be557ea08cb31186457363d37eee71d44bb2',
          'statsmodels_ar_model_080.py': '6d4730507514f7b4605e9740e8fab9a647db2e358ef74f4ede504a02c39e2d80'}


def load_reference(directory):
    for name, digest in HASHES.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError('pinned source mismatch')
    tree = ast.parse((directory / 'statsmodels_ar_model_080.py').read_bytes())
    methods = []
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name in {'AR', 'ARResults'}:
            for method in cls.body:
                wanted = {'fit', '_stackX'} if cls.name == 'AR' else {'bse'}
                if isinstance(method, ast.FunctionDef) and method.name in wanted:
                    method.decorator_list = []
                    methods.append(method)
    namespace = dict(np=np, OLS=OLS, lagmat=lagmat, add_trend=add_trend, util=util)
    exec(compile(ast.Module(body=methods, type_ignores=[]), 'historical_ar_methods.py', 'exec'), namespace)
    class Results:
        def __init__(self, model, params, normalized_cov_params):
            self.model, self.params = model, params
            self.nobs, self.k_ar, self.k_trend = model.nobs, model.k_ar, model.k_trend
            self.resid = model.Y.ravel() - model.X @ params
            self.normalized_cov_params = normalized_cov_params
        def cov_params(self, scale):
            return self.normalized_cov_params * scale
        bse = property(namespace['bse'])
    class ReferenceAR:
        def __init__(self, values):
            self.endog = np.asarray(values)[:, None]
            self.endog_names = 'synthetic'
        fit = namespace['fit']
        _stackX = namespace['_stackX']
    namespace.update(ARResults=Results, ARResultsWrapper=lambda result: result)
    source = ast.parse((directory / 'preproc.py').read_bytes())
    source.body = [n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'ARError']
    outer = dict(np=np, AR=ReferenceAR, BaseEstimator=BaseEstimator, TransformerMixin=TransformerMixin)
    exec(compile(source, 'pinned_ar_error.py', 'exec'), outer)
    return outer['ARError']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    reference = load_reference(args.reference_dir)
    count, maximum = 0, 0.
    for seed in range(5):
        for order, subsample, length in [(1, 1, 121), (3, 2, 121), (5, 4, 8000)]:
            a = np.random.default_rng(seed + 1610).normal(size=(2, 3, length))
            expected = reference(order=order, subsample=subsample).fit_transform(a)
            actual = ar_standard_errors.channel_ar_standard_errors(a, order, subsample)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
            maximum = max(maximum, float(np.max(abs(actual - expected))))
            count += 1
    report = dict(synthetic_only=True, cases_passed=count, maximum_absolute_error=maximum,
        source_revision='00f937cc7710977dc812d9fc675864e2b8288658', statsmodels_reference='v0.8.0', source_hashes=HASHES,
        implementation_hash=hashlib.sha256(Path(inspect.getfile(ar_standard_errors)).read_bytes()).hexdigest(),
        validation_script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        library_versions={n: importlib.metadata.version(n) for n in ['numpy', 'scipy', 'statsmodels']},
        reference_adaptation='Unmodified historical AR.fit, AR._stackX and ARResults.bse method bodies; property decorators stripped. Current OLS/lag/trend utilities injected. Minimal model/result containers expose original conditional-OLS residual and covariance semantics. Original ARError class unmodified.',
        limitations='Conditional OLS with explicit order and constant trend only; no historical runtime reproduction, maximum-likelihood branch, model selection or predictive-performance validation.')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
