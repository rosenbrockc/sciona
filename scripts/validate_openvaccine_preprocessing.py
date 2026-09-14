"""Complete pinned preprocessing versus independent synthetic tensor reference."""
import ast
import hashlib
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def reference(sequence, loops, structure, probability):
    n = len(sequence)
    pairs = list(range(n))
    stack = []
    for i, token in enumerate(structure):
        if token == '(':
            stack.append(i)
        elif token == ')':
            j = stack.pop()
            pairs[i], pairs[j] = j, i
    assert not stack
    bases, loop_vocab = 'AGCU', 'SMIBHEX'
    codes = sorted([16, 32] + [2**i + 2**(6+j)
                              for i, j in itertools.product(range(4), range(7))])
    nodes = np.zeros((1, n+2, 55), dtype=np.float32)
    nodes[0, 0, 4] = nodes[0, -1, 5] = 1
    nodes[0, 0, 13+codes.index(16)] = nodes[0, -1, 13+codes.index(32)] = 1
    max_col = [max(probability[:, i]) for i in range(n)]
    sum_col = [sum(probability[:, i]) for i in range(n)]
    for pos in range(n):
        i, j = bases.index(sequence[pos]), loop_vocab.index(loops[pos])
        nodes[0, pos+1, i] = nodes[0, pos+1, 6+j] = 1
        nodes[0, pos+1, 13+codes.index(2**i + 2**(6+j))] = 1
        row = sorted(probability[pos])
        pair_prob = probability[pos, pairs[pos]]
        extra = [row[-1], 1-sum(row), row[-1]-row[-2],
                 max_col[int(np.argmax(probability[:, pos]))]-max_col[pos],
                 pair_prob, max_col[pos]-pair_prob,
                 max_col[pairs[pos]]-pair_prob, sum_col[pairs[pos]]-pair_prob]
        for selected in [set('()'), {'.'}]:
            seeds = [k for k, token in enumerate(structure) if token in selected]
            distance = min([abs(pos-k) for k in seeds], default=10000)
            extra.extend([1/(1+distance), (1/(1+distance))**0.5])
        nodes[0, pos+1, 43:] = extra
    adjacency = np.zeros((1, n+2, n+2, 8), dtype=np.float32)
    adjacency[0, 1:-1, 1:-1, 0] = probability
    for pos, paired in enumerate(pairs):
        if paired != pos:
            adjacency[0, pos+1, paired+1, 1] = 1
    pair_seeds = [(i+1, j+1) for i, j in enumerate(pairs) if i != j]
    for i, j in itertools.product(range(n+2), repeat=2):
        linear = 1/(1+abs(i-j))
        distance = min([10] + [abs(i-k)+abs(j-k) for k in range(n+2)]
                       + [1+abs(i-a)+abs(j-b) for a, b in pair_seeds])
        grid = 1/(1+distance)
        adjacency[0, i, j, 2:] = [linear, linear**2, linear**4, grid, grid**2, grid**4]
    return nodes, adjacency


def main():
    cache = Path('/private/tmp/sciona_openvaccine_source')
    manifest = json.loads((cache/'manifest.json').read_text())
    name = 'scripts/nullrecurrent_inference.py'
    raw = (cache/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == next(p['sha256'] for p in manifest['pins'] if p['software_path'] == name)
    wanted = {'return_ohe', 'get_input', 'get_pair_idx', 'calc_dist_to_pair',
              'calc_dist_to_single', 'get_structure_adj', 'pandas_list_to_array',
              'preprocess_inputs1', 'preprocess_inputs', 'get_inputs', 'padding_2D',
              'get_distance_matrix', 'get_distance_matrix_2d', 'calc_neighbor'}
    definitions = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in definitions} == wanted
    ns = dict(np=np, pd=pd, DIST_NEW=True, DIST_NEW2=True, BBP=True, BBP1=True,
              BBP2=True, BBP3=True, BBP4=True, BBP_TOTAL=8,
              token2int={x:i for i, x in enumerate('().ACGUBEHIMSXse')})
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<pinned-preprocessing>', 'exec'), ns)
    rng = np.random.default_rng(729)
    cases = [('ACAAAAGU', '((....))'), ('AUAUAUAU', '()()()()'),
             ('AGCUAGCU', '........'), ('AU', '()')]
    for sequence, structure in cases:
        n = len(sequence)
        loops = ''.join('SMIBHEX'[i % 7] for i in range(n))
        matrix = rng.uniform(0, 0.1/n, size=(n, n))
        matrix += matrix.T
        np.fill_diagonal(matrix, 0)
        frame = pd.DataFrame([dict(id='synthetic', sequence=sequence, structure=structure,
                                   bpRNA_string=loops, seq_length=n)])
        frame_before, matrix_before = frame.copy(deep=True), matrix.copy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            actual = ns['get_inputs'](frame, matrix)
        expected = reference(sequence, loops, structure, matrix)
        for observed, target in zip(actual, expected):
            assert observed.dtype == np.float32
            np.testing.assert_allclose(observed, target, rtol=2e-7, atol=1e-8)
        pd.testing.assert_frame_equal(frame, frame_before)
        np.testing.assert_array_equal(matrix, matrix_before)
    failures = {}
    for label, frame, matrix in [
        ('length_one', pd.DataFrame([dict(id='synthetic', sequence='A', structure='.', bpRNA_string='S', seq_length=1)]), np.zeros((1,1))),
        ('multiple_rows', pd.concat([frame, frame], ignore_index=True), matrix),
    ]:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            try:
                ns['get_inputs'](frame, matrix)
            except (IndexError, ValueError) as error:
                failures[label] = type(error).__name__
            else:
                raise AssertionError('Expected source boundary failure: '+label)
    report = dict(status='passed', source_commit=manifest['commit'], synthetic_only=True,
                  complete_preprocessing_cases=len(cases), node_channels=55, adjacency_channels=8,
                  source_inputs_unchanged=True, source_boundary_failures=failures,
                  source_modified=False, tensorflow_executed=False,
                  scope='Full inference preprocessing given synthetic structure/probabilities; no folding or model/training execution.',
                  validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'docs/reviews/competition_openvaccine_preprocessing.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
