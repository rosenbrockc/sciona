#!/usr/bin/env python3
"""Synthetic numerical parity with the pinned source GLM train/predict bodies."""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sciona.atoms.ml.sklearn.linear_model.feng_glm import feng_glm_segment_probabilities

SOURCE_SHA = '66ae627eb632b4d734ba846dfa167db0e01e354d0cbf96e1d576e632a0077d85'


def load_reference(reference_dir):
    path = reference_dir / 'Feng_models_GLM.py'
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA:
        raise ValueError('pinned source KNN differs')
    lines = data.decode().splitlines()
    assert sum(line.lstrip().startswith('print ') for line in lines) == 2
    tree = ast.parse('\n'.join(line for line in lines if not line.lstrip().startswith('print ')))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'reshape_data', 'train', 'predict'}]
    assert len(tree.body) == 3
    ns = {'np': np, 'StandardScaler': StandardScaler, 'linear_model': linear_model}
    exec(compile(tree, str(path), 'exec'), ns)
    return ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    ns = load_reference(args.reference_dir)
    cases = []
    for seed in [9211, 9212, 9213, 9214, 9215]:
        for windows in [10, 12, 31]:
            rng = np.random.default_rng(seed)
            labels = np.array([0, 0, 1, 1, 0, 1])
            train = rng.normal(size=(6, windows, 384)) + labels[:, None, None]*3
            prediction = rng.normal(size=(4, 12, 384)) + np.array([0, 3, 0, 3])[:, None, None]
            train[0, 0, 0] = -np.inf
            prediction[1, 1, 1] = -np.inf
            # Source loaders allocate float32 tensors. Substitute only in-memory
            # synthetic loading; numerical train/predict/reshape stay unchanged.
            ns['load_train_data_lasso'] = lambda *_: {'x': train.astype(np.float32), 'y': labels.copy()}
            ns['load_test_data'] = lambda *_: {'x': prediction.astype(np.float32), 'id': list(range(len(prediction)))}
            model, scaler = ns['train'](None, None)
            expected = np.array([score for _, score in ns['predict'](None, model, scaler, None)])
            actual = feng_glm_segment_probabilities(train, labels, prediction)
            np.testing.assert_array_equal(actual, expected)
            spread = float(np.ptp(expected))
            assert spread > .1
            cases.append(dict(seed=seed, training_windows_per_segment=windows, exact=True, prediction_spread=spread))
    root = Path(__file__).resolve().parents[1]
    report = dict(source_commit='00f937cc7710977dc812d9fc675864e2b8288658', source_sha256=SOURCE_SHA,
        provider_sha256=hashlib.sha256(Path(inspect.getsourcefile(feng_glm_segment_probabilities)).read_bytes()).hexdigest(),
        validation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256((root/'tests/test_feng_glm.py').read_bytes()).hexdigest(),
        libraries={'numpy': np.__version__, 'sklearn': sklearn.__version__}, cases_passed=len(cases), cases=cases,
        limitations=['Removed two Python2 diagnostic print statements; numerical bodies unchanged, with synthetic source-float32 loaders.',
                     'Source leaves solver unspecified; current sklearn LBFGS defaults used, not historical 2016 solver reproduction.',
                     'Training scaler is reused at prediction; only negative infinity is cleaned, NaNs rejected.',
                     'Complete serialized graph validation and catalog approval remain outstanding.'])
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'cases_passed': len(cases), 'all_exact': True, 'minimum_prediction_spread': min(c['prediction_spread'] for c in cases)}))


if __name__ == '__main__':
    main()
