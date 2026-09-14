#!/usr/bin/env python3
"""Compare explicit identity alignment with pinned source final blending."""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from sciona.atoms.riemannian_bci.signal_processing.ensemble_alignment import aligned_eleven_model_blend


def validate(reference_dir, output):
    root = Path(__file__).resolve().parents[1]
    source_bytes = (reference_dir/'make_blend.py').read_bytes()
    source_hash = '6a64446123cfa6d3a867837057cda256cf13442c674317bc239117d00520374b'
    if hashlib.sha256(source_bytes).hexdigest() != source_hash:
        raise ValueError('Pinned source changed')
    source = source_bytes.decode()
    assignments = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'weights' for target in node.targets)]
    assert len(assignments) == 1
    weights = ast.literal_eval(assignments[0].value)
    assert len(weights) == 11 and set(weights.values()) == {1.}
    replacements = {
        "default = pd.read_csv('./csv_files/sample_submission.csv', index_col=0)": 'default = fixture_default.copy(deep=True)',
        "res = pd.read_csv(base + fn, index_col='File')": 'res = fixture_predictions[fn].copy(deep=True)',
        'np.sum(weights.values())': 'np.sum(list(weights.values()))',
        "default.to_csv('./submissions/Winning_submission.csv')": '',
    }
    for before, after in replacements.items():
        assert source.count(before) == 1
        source = source.replace(before, after)
    cases = []
    for case in range(3):
        rng = np.random.default_rng(821 + case)
        output_ids = np.array([51, 29, 17, 84])
        baseline = np.zeros(4) if case == 0 else np.array([.7, -.5, .2, 1.1])
        predictions, identities, frames = [], [], {}
        for index, name in enumerate(weights):
            extra = np.arange(100, 100 + index % 3) if case else np.array([], dtype=int)
            ids = rng.permutation(np.r_[output_ids, extra])
            scores = rng.integers(0, 6, len(ids)).astype(float) / 5 if case < 2 else np.zeros(len(ids))
            predictions.append(scores)
            identities.append(ids)
            frames[name] = pd.DataFrame({'Class': scores}, index=pd.Index(ids, name='File'))
        namespace = dict(fixture_default=pd.DataFrame({'Class': baseline}, index=output_ids), fixture_predictions=frames)
        exec(compile(source, '<pinned-source-blend-with-synthetic-io>', 'exec'), namespace)
        expected = namespace['default']['Class'].to_numpy()
        actual = aligned_eleven_model_blend(predictions, identities, output_ids, baseline)
        np.testing.assert_array_equal(actual, expected)
        if case:
            assert np.max(actual) > 1  # Neither clipping nor final re-ranking.
        cases.append(dict(case=case, exact=True, model_rows=[len(value) for value in predictions],
                          output_rows=len(actual), nonzero_baseline=bool(case), all_tied=case == 2))
    provider = Path(inspect.getsourcefile(aligned_eleven_model_blend))
    report = dict(source_revision='00f937cc7710977dc812d9fc675864e2b8288658', source_sha256=source_hash,
        provider_sha256=hashlib.sha256(provider.read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256((root/'tests/test_ensemble_alignment.py').read_bytes()).hexdigest(),
        numpy_version=np.__version__, pandas_version=pd.__version__, scipy_version=scipy.__version__,
        cases=cases, limitations=['Synthetic opaque integer identities replace source file identities; no real prediction files are read.',
            'Source CSV I/O replaced by synthetic DataFrames; Python 2 dictionary-values summation adapted to a list.',
            'Source dictionary insertion order used explicitly; historical Python 2 iteration order may differ in last-bit summation.',
            'Final blending only; does not execute the eleven upstream models.'])
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(exact_source_cases=len(cases))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    validate(args.reference_dir, args.output)
