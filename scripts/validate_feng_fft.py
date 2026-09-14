#!/usr/bin/env python3
"""Validate Feng FFT features against pinned numerical source definitions."""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sciona.atoms.riemannian_bci.signal_processing.feng_fft import feng_fft_features

SOURCE_SHA = '41e9a15823c6ee386b2c35e9676a8d9daeb57701637a8319a7825430aa82273a'


def load_reference(reference_dir):
    source = reference_dir / 'Feng_preprocessors_fft.py'
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA:
        raise ValueError('pinned FFT source differs')
    tree = ast.parse(data)
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'group_into_bands', 'compute_fft'}]
    assert len(tree.body) == 2
    # Exactly one Python 2 integer shape division; all spectral arithmetic stays
    # unchanged. Only these numerical functions execute, with no file access.
    divisions = [n for n in ast.walk(tree) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)]
    shape_division = [n for n in divisions if isinstance(n.right, ast.Name) and n.right.id == 'stride_sec']
    assert len(shape_division) == 1
    shape_division[0].op = ast.FloorDiv()
    namespace = {'np': np, 'DataFrame': pd.DataFrame}
    exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), namespace)
    return namespace['compute_fft']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    reference = load_reference(args.reference_dir)
    cases = []
    for seed in [9101, 9102, 9103, 9104, 9105]:
        for duration in [30, 61, 600]:
            x = np.random.default_rng(seed).normal(size=(duration*400, 3)).astype(np.float32)
            expected = reference(x, duration, 400, 6, 30, 30, 'meanlog_std')
            actual = feng_fft_features(x)
            np.testing.assert_array_equal(actual, expected)
            cases.append(dict(seed=seed, duration_seconds=duration, exact=True, max_abs_error=0.))
    root = Path(__file__).resolve().parents[1]
    report = dict(source_commit='00f937cc7710977dc812d9fc675864e2b8288658',
        source_sha256=SOURCE_SHA,
        provider_sha256=hashlib.sha256(Path(inspect.getsourcefile(feng_fft_features)).read_bytes()).hexdigest(),
        validation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256((root/'tests/test_feng_fft_features.py').read_bytes()).hexdigest(),
        libraries={'numpy': np.__version__, 'pandas': pd.__version__},
        cases_passed=len(cases), cases=cases, limitations=[
            'Python 2 shape integer division adapted to floor division; numerical function bodies otherwise unchanged.',
            'Current NumPy/pandas synthetic numerical parity; no historical environment or predictive-performance claim.',
            'Source undefined log features are preserved for later classifier cleanup.',
            'Complete model execution and catalog promotion remain outstanding.'])
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'cases_passed': len(cases), 'all_exact': True}))


if __name__ == '__main__':
    main()
