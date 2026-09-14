#!/usr/bin/env python3
"""Validate source-configured bagging and postprocessing on synthetic features."""
import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from sciona.atoms.ml.xgboost import competition_bagging
from sciona.atoms.riemannian_bci.signal_processing import segment_scores

HASHES = {
    'Alex_Gilberto_models_autocorrmat_TS_XGB.yml': '53626a0381a5ed4c3f0e5dbb2c877272cacadec5005b783a34911feeda617494',
    'Alex_Gilberto_models_coherence_transposed_TS_XGB.yml': 'd562f251c13b0de848898f0cb1250a2fd7e4c65696ef9178e4ee3c77d80351da',
    'make_blend.py': '6a64446123cfa6d3a867837057cda256cf13442c674317bc239117d00520374b',
}


def reference_model(config):
    params = dict(config['model'][1]['BaggingClassifier'])
    expression = ast.parse(params.pop('base_estimator'), mode='eval').body
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name) or expression.func.id != 'XGBClassifier' or expression.args:
        raise ValueError('unexpected source constructor')
    kwargs = {entry.arg: ast.literal_eval(entry.value) for entry in expression.keywords}
    if None in kwargs:
        raise ValueError('unexpected expanded arguments')
    # Serial execution and the modern constructor name are explicit compatibility choices.
    return BaggingClassifier(estimator=XGBClassifier(**kwargs, n_jobs=1), **params, n_jobs=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    sources = {}
    for name, digest in HASHES.items():
        content = (args.reference_dir / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError('pinned reference hash mismatch')
        sources[name] = content.decode()
    counts = {'classifier_cases': 0, 'segment_cases': 0, 'eleven_model_blend_cases': 0}
    maximum = 0.
    for seed in range(3):
        rng = np.random.default_rng(2900 + seed)
        labels = np.tile([0, 1], 12)
        indices = np.repeat(np.arange(len(labels)), 5)
        train = rng.normal(size=(len(indices), 24)) + .7 * labels[indices, None]
        prediction = rng.normal(size=(28, 24))
        for name in HASHES:
            if not name.endswith('.yml'):
                continue
            config = yaml.safe_load(sources[name])
            model = reference_model(config)
            model.fit(train, labels.repeat(5))
            expected = model.predict_proba(prediction)[:, 1]
            actual = competition_bagging.bagged_window_probabilities(train, indices, labels, prediction,
                                                                     config['model'][1]['BaggingClassifier']['n_estimators'])
            np.testing.assert_array_equal(actual, expected)
            counts['classifier_cases'] += 1
            maximum = max(maximum, float(np.max(abs(actual - expected))))
            ids = np.repeat(np.arange(7), 4)
            grouped = pd.DataFrame({'probability': expected, 'segment': ids}).groupby('segment').max()['probability'].to_numpy()
            np.testing.assert_array_equal(segment_scores.segment_probability_max(actual, ids, 7), grouped)
            counts['segment_cases'] += 1
        # Execute the actual public blend script against in-memory synthetic frames.
        tree = ast.parse(sources['make_blend.py'])
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        class Python2Values(ast.NodeTransformer):
            def visit_Call(self, node):
                node = self.generic_visit(node)
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'values':
                    return ast.copy_location(ast.Call(func=ast.Name(id='list', ctx=ast.Load()), args=[node], keywords=[]), node)
                return node
        tree = ast.fix_missing_locations(Python2Values().visit(tree))
        weights = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'weights' for t in node.targets))
        predictions = rng.integers(0, 7, size=(len(weights), 13)).astype(float)
        frames = {name: pd.DataFrame({'Class': row}, index=np.arange(13)) for name, row in zip(weights, predictions)}
        outputs = []
        class Frame(pd.DataFrame):
            def to_csv(self, *args, **kwargs):
                outputs.append(self.copy())
        class PandasMemory:
            @staticmethod
            def read_csv(path, **kwargs):
                name = path.rsplit('/', 1)[-1]
                return Frame(frames[name].copy()) if name in frames else Frame({'Class': np.zeros(13)})
        exec(compile(tree, 'pinned_make_blend.py', 'exec'), {'np': np, 'pd': PandasMemory, 'rankdata': rankdata})
        actual = segment_scores.normalized_rank_blend(predictions, np.ones(len(weights)))
        np.testing.assert_array_equal(actual, outputs[0]['Class'].to_numpy())
        counts['eleven_model_blend_cases'] += 1
    report = {
        'synthetic_only': True, 'counts': counts, 'maximum_classifier_absolute_error': maximum,
        'source_revision': '00f937cc7710977dc812d9fc675864e2b8288658', 'source_hashes': HASHES,
        'implementation_hashes': {m.__name__: hashlib.sha256(Path(inspect.getfile(m)).read_bytes()).hexdigest()
                                  for m in [competition_bagging, segment_scores]},
        'validation_script_hash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'library_versions': {name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'scikit-learn', 'xgboost']},
        'limitations': 'Modern-library source-configuration parity only. No historical XGBoost/sklearn environment or competition performance claim. Eleven-model blend inputs are synthetic; only two model branches are reconstructed so far.'
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
