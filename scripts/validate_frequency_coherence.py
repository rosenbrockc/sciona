#!/usr/bin/env python3
"""Compare synthetic frequency coherence against pinned public competition code."""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sciona.atoms.riemannian_bci.covariance_features import frequency_coherence

SOURCE_HASH = '1cfbc71faa230e1e4ffb72f27604be557ea08cb31186457363d37eee71d44bb2'
COMMIT = '00f937cc7710977dc812d9fc675864e2b8288658'


class Python2Slices(ast.NodeTransformer):
    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Div) and isinstance(node.left, ast.Name) and node.left.id == 'window' and isinstance(node.right, ast.Constant) and node.right.value == 2:
            node.op = ast.FloorDiv()
        return node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source = args.reference.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_HASH:
        raise ValueError('pinned public source hash mismatch')
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'coherences']
    if len(tree.body) != 1:
        raise ValueError('expected one reference function')
    tree = ast.fix_missing_locations(Python2Slices().visit(tree))
    namespace = {'np': SimpleNamespace(**vars(np), complex_=np.complex128),
                 'hanning': np.hanning, 'xrange': range}
    exec(compile(tree, 'ported_public_coherences.py', 'exec'), namespace)
    reference = namespace['coherences']
    cases = 0
    maximum = 0.
    bands = [[.1, 4], [4, 8], [8, 15], [15, 30], [30, 90], [90, 170]]
    for seed in range(5):
        for overlap in [0., .5, .75]:
            # The public reference hardcodes 16 channels in transpose mode.
            windows = np.random.default_rng(seed + 900).normal(size=(2, 16, 1031))
            expected = np.stack([reference(window, window=256, fs=400, overlap=overlap,
                                           frequencies=bands, transpose=True,
                                           aggregate=False, normalize=True) for window in windows])
            actual = frequency_coherence.frequency_band_coherence(windows, bands, 400., 256, overlap)
            np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
            maximum = max(maximum, float(np.max(abs(actual - expected))))
            cases += 1
    report = {
        'synthetic_only': True, 'source_commit': COMMIT, 'source_hash': SOURCE_HASH,
        'implementation_hash': hashlib.sha256(Path(inspect.getfile(frequency_coherence)).read_bytes()).hexdigest(),
        'validation_script_hash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'numpy_version': np.__version__, 'cases_passed': cases,
        'maximum_absolute_error': maximum,
        'reference_port': 'Python 2 window/2 slices -> integer division; xrange -> range; np.complex_ -> complex128; symmetric hanning -> np.hanning',
        'limitations': [
            'Same-library synthetic numerical parity, not historical environment reproduction or predictive performance validation.',
            'Reference comparison uses its hardcoded 16 channels; generalized channel counts have independent scalar DFT tests.',
            'Only transpose=True, aggregate=False, normalize=True behavior covered.',
            'Positive semidefinite output is not guaranteed strictly positive definite.'
        ]
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
