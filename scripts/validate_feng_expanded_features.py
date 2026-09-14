#!/usr/bin/env python3
"""Compare expanded Feng features with pinned public numerical source bodies."""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sciona.atoms.riemannian_bci.signal_processing.feng_expanded_features import feng_expanded_features

SOURCE_SHA = '41e9a15823c6ee386b2c35e9676a8d9daeb57701637a8319a7825430aa82273a'


def load_reference(directory):
    path = directory/'Feng_preprocessors_fft.py'
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA:
        raise ValueError('source differs')
    selected = {'CorrelationMatrix', 'Eigenvalues', 'upper_right_triangle', 'group_into_bands',
                'compute_fft_more', 'compute_timedomaincoef', 'compute_frequencydomaincoef'}
    tree = ast.parse(data)
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in selected]
    assert len(tree.body) == len(selected)
    class Compatibility(ast.NodeTransformer):
        def visit_BinOp(self, node):
            self.generic_visit(node)
            if isinstance(node.op, ast.Div) and isinstance(node.right, ast.Name) and node.right.id in {'stride_sec', 'n_timesteps'}:
                node.op = ast.FloorDiv()
            return node
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == 'np' and node.attr == 'alltrue':
                node.attr = 'all'
            return node
    tree = ast.fix_missing_locations(Compatibility().visit(tree))
    ns = {'np': np, 'DataFrame': pd.DataFrame, 'preprocessing': preprocessing}
    exec(compile(tree, str(path), 'exec'), ns)
    return ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    ns = load_reference(args.reference_dir)
    cases = []
    for seed in [9521, 9522, 9523]:
        for duration in [50, 100, 600]:
            x = np.random.default_rng(seed).normal(size=(duration*400, 16)).astype(np.float32)
            spectral = ns['compute_fft_more'](x, duration, 400, 6, 50, 50, 'meanlog_std')
            time = ns['compute_timedomaincoef'](x, duration, 400, 50, 50)
            frequency = ns['compute_frequencydomaincoef'](x, duration, 400, 6, 50, 50)
            expected = np.c_[spectral, time, frequency]
            actual = feng_expanded_features(x)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
            cases.append(dict(seed=seed, duration_seconds=duration, max_abs_error=float(np.max(abs(actual-expected)))))
    root = Path(__file__).resolve().parents[1]
    report = dict(source_commit='00f937cc7710977dc812d9fc675864e2b8288658', source_sha256=SOURCE_SHA,
        provider_sha256=hashlib.sha256(Path(inspect.getsourcefile(feng_expanded_features)).read_bytes()).hexdigest(),
        validation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256((root/'tests/test_feng_expanded_features.py').read_bytes()).hexdigest(),
        cases_passed=len(cases), cases=cases, libraries={'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
        limitations=['Python2 integer shape division and removed np.alltrue adapted; numerical source bodies otherwise unchanged.',
                     'Current-library synthetic parity, no historical environment or predictive-performance claim.',
                     'Source fixed 16-channel feature contract; complete model execution and approval remain outstanding.'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases_passed': len(cases), 'maximum_absolute_error': max(c['max_abs_error'] for c in cases)}))


if __name__ == '__main__':
    main()
