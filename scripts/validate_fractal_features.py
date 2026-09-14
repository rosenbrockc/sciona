#!/usr/bin/env python3
"""Compare the source VariousFeatures block after explicit Python 2 compatibility."""
import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sciona.atoms.riemannian_bci.signal_processing import fractal_features

HASHES = {'preproc.py': '1cfbc71faa230e1e4ffb72f27604be557ea08cb31186457363d37eee71d44bb2',
          'transformer.py': '701697177ebae94aece0018130241b50a4f75f7212a258e31c09bd8b3bc230f5'}


class IntegerEndpoints(ast.NodeTransformer):
    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if any(isinstance(t, ast.Name) and t.id == 'i_end' for t in node.targets):
            if not isinstance(node.value, ast.BinOp) or not isinstance(node.value.op, ast.Div):
                raise ValueError('unexpected source endpoint expression')
            node.value.op = ast.FloorDiv()
        return node


def load_reference(directory):
    for name, digest in HASHES.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError('pinned public source mismatch')
    namespace = dict(np=np, pd=SimpleNamespace(expanding_std=lambda x: pd.Series(x).expanding().std(ddof=1).to_numpy()),
                     xrange=range, BaseEstimator=BaseEstimator, TransformerMixin=TransformerMixin)
    for name, selected in [('transformer.py', {'Hurst', 'PFD', 'HFD'}), ('preproc.py', {'VariousFeatures'})]:
        tree = ast.parse((directory / name).read_bytes())
        tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in selected]
        tree = ast.fix_missing_locations(IntegerEndpoints().visit(tree))
        exec(compile(tree, 'ported_' + name, 'exec'), namespace)
    return namespace['VariousFeatures']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    reference = load_reference(args.reference_dir)
    count, maximum = 0, 0.
    for seed in range(5):
        for length in [17, 121, 8000]:
            a = np.random.default_rng(1720 + seed).normal(size=(2, 3, length))
            before = a.copy()
            expected = reference().fit_transform(a.copy())
            actual = fractal_features.channel_fractal_features(a)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
            np.testing.assert_array_equal(a, before)
            maximum = max(maximum, float(np.max(abs(actual - expected))))
            count += 1
    report = dict(synthetic_only=True, cases_passed=count, maximum_absolute_error=maximum,
        source_revision='00f937cc7710977dc812d9fc675864e2b8288658', source_hashes=HASHES,
        implementation_hash=hashlib.sha256(Path(inspect.getfile(fractal_features)).read_bytes()).hexdigest(),
        validation_script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        library_versions={n: importlib.metadata.version(n) for n in ['numpy', 'pandas']},
        reference_adaptation='Python 2 integer i_end division and xrange restored; removed pandas expanding_std maps to expanding sample standard deviation on ndarray. Current NumPy least-squares defaults retained. Original classes and VariousFeatures ordering otherwise unchanged.',
        limitations='Source-specific feature formulas, not interchangeable standard fractal estimators. Source Hurst mutation intentionally removed while preserving outputs. No full branch, historical-library or physiological-validity claim.')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
