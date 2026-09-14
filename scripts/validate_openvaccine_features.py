"""Independent synthetic comparisons for pinned OpenVaccine feature primitives.

Does not import TensorFlow, access RNA datasets, or establish gradient parity.
"""
import ast
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    cache = Path('/private/tmp/sciona_openvaccine_source')
    manifest = json.loads((cache / 'manifest.json').read_text())
    name = 'scripts/nullrecurrent_inference.py'
    raw = (cache / name).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == next(p['sha256'] for p in manifest['pins'] if p['software_path'] == name)
    wanted = {'return_ohe', 'get_input', 'get_pair_idx', 'calc_dist_to_pair',
              'calc_dist_to_single', 'mean_squared_error1', 'get_structure_adj'}
    definitions = [n for n in ast.parse(raw).body
                   if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert len(definitions) == len(wanted)
    ns = dict(np=np, pd=pd)
    exec(compile(ast.Module(body=definitions, type_ignores=[]),
                 '<pinned-openvaccine-features>', 'exec'), ns)

    bases, loops = 'AGCU', 'SMIBHEX'
    combinations = list(itertools.product(bases, loops))
    # Enumerate vocabulary combinations, independent of source padding strings.
    frame = pd.DataFrame([dict(sequence=''.join(x[0] for x in combinations),
                               bpRNA_string=''.join(x[1] for x in combinations),
                               structure='.' * len(combinations))])
    original = frame.copy(deep=True)
    actual = ns['get_input'](frame)
    codes = sorted([2**4, 2**5] + [2**i + 2**(6+j)
                                  for i in range(4) for j in range(7)])
    expected = np.zeros((1, 30, 43), dtype=np.float32)
    expected[0, 0, 4] = expected[0, -1, 5] = 1
    expected[0, 0, 13 + codes.index(16)] = 1
    expected[0, -1, 13 + codes.index(32)] = 1
    for pos, (base, loop) in enumerate(combinations, start=1):
        i, j = bases.index(base), loops.index(loop)
        expected[0, pos, i] = expected[0, pos, 6+j] = 1
        expected[0, pos, 13 + codes.index(2**i + 2**(6+j))] = 1
    np.testing.assert_array_equal(actual, expected)
    pd.testing.assert_frame_equal(frame, original)
    for base, loop in combinations:
        single = pd.DataFrame([dict(sequence=base, bpRNA_string=loop, structure='.')])
        value = ns['get_input'](single)
        pos = combinations.index((base, loop)) + 1
        np.testing.assert_array_equal(value[0], expected[0, [0, pos, -1]])
    # Source calculates structure one-hot but excludes it from the returned node features.
    altered = frame.copy()
    altered['structure'] = '(' * 14 + ')' * 14
    np.testing.assert_array_equal(ns['get_input'](altered), expected)

    structures = ['........', '((....))', '.(..)...', '()()()()', '(((( ))))'.replace(' ', '')]
    for structure in structures:
        positions = np.arange(len(structure))
        for name, selected in [('calc_dist_to_pair', set('()')),
                               ('calc_dist_to_single', {'.'})]:
            seeds = [i for i, token in enumerate(structure) if token in selected]
            reference = (np.min(abs(positions[:, None] - np.array(seeds)[None, :]), axis=1)
                         if seeds else np.full(len(structure), 10000))
            np.testing.assert_array_equal(ns[name](structure), reference)
    pairs = ns['get_pair_idx']('((....))').astype(int)
    np.testing.assert_array_equal(pairs, [7, 6, 2, 3, 4, 5, 1, 0])
    # Entirely synthetic complementary symbols; no folding engine is invoked.
    adjacency = ns['get_structure_adj'](pd.DataFrame([
        dict(sequence='ACAAAAGU', structure='((....))', seq_length=8)]))
    reference = np.zeros((1, 10, 10, 1))
    for a, b in [(1, 8), (2, 7), (8, 1), (7, 2)]:
        reference[0, a, b, 0] = 1
    np.testing.assert_array_equal(adjacency, reference)
    # The parser silently accepts unmatched opening parentheses.
    np.testing.assert_array_equal(ns['get_pair_idx']('(...'), [0, 1, 2, 3])
    try:
        ns['get_pair_idx']('...)')
    except IndexError:
        pass
    else:
        raise AssertionError('Expected unmatched closing parenthesis failure')

    rng = np.random.default_rng(418)
    for batch, length in itertools.product([1, 2, 5], [2, 7]):
        truth, pred = rng.normal(size=(2, batch, length))
        weights = rng.uniform(0.1, 3, size=batch)
        reference = sum(float(w) * (sum(float(x-y)**2 for x, y in zip(t, p)) / length)**0.5
                        for t, p, w in zip(truth, pred, weights)) / sum(weights)
        np.testing.assert_allclose(ns['mean_squared_error1'](truth, pred, weights),
                                   reference, rtol=1e-14, atol=1e-14)
    report = dict(status='passed', source_commit=manifest['commit'], source_sha256=digest,
                  synthetic_only=True, node_channels=43, vocabulary_combinations=28,
                  vocabulary_batch_invariance=True, node_structure_features_excluded=True,
                  distance_cases=10, structure_adjacency_exact=True,
                  unmatched_opening_silently_accepted=True, unmatched_closing_raises=True,
                  weighted_per_sample_rmse_cases=6, tensorflow_executed=False,
                  scope='Node features, structure primitives and NumPy RMSE helper only; no folding, full preprocessing, model, gradient or training claim.',
                  validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT / 'docs/reviews/competition_openvaccine_features.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
