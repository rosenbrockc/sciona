#!/usr/bin/env python3
"""Compare synthetic per-channel statistics to pinned competition BasicStats."""
import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sciona.atoms.riemannian_bci.signal_processing import basic_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source_hash = '1cfbc71faa230e1e4ffb72f27604be557ea08cb31186457363d37eee71d44bb2'
    if hashlib.sha256(args.reference.read_bytes()).hexdigest() != source_hash:
        raise ValueError('pinned public source mismatch')
    tree = ast.parse(args.reference.read_bytes())
    tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BasicStats']
    namespace = dict(np=np, sp=sp, BaseEstimator=BaseEstimator, TransformerMixin=TransformerMixin)
    exec(compile(tree, 'pinned_basic_stats.py', 'exec'), namespace)
    count = 0
    for seed in range(5):
        for length in [5, 121, 8000]:
            a = np.random.default_rng(9200 + seed).normal(size=(2, 3, length))
            expected = namespace['BasicStats']().fit_transform(a)
            actual = basic_statistics.channel_basic_statistics(a)
            np.testing.assert_array_equal(actual, expected)
            count += 1
    report = dict(synthetic_only=True, source_revision='00f937cc7710977dc812d9fc675864e2b8288658',
                  source_hash=source_hash, cases_passed=count, maximum_absolute_error=0.,
                  implementation_hash=hashlib.sha256(Path(inspect.getfile(basic_statistics)).read_bytes()).hexdigest(),
                  validation_script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  library_versions={n: importlib.metadata.version(n) for n in ['numpy', 'scipy']},
                  limitations='Same-library source feature parity only; no full combined-feature branch or historical environment/predictive-performance validation.')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
