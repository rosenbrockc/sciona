"""Wiring test; full trained execution is a separate validation requirement."""
import cv2
import numpy as np

from sciona.cassava_fold_contract import build_plan
from sciona.cassava_pipeline import execute


def test_all_training_stages_feed_correct_family_outputs(tmp_path, monkeypatch):
    import sciona.cassava_torch_folds as torch_folds
    import sciona.cassava_efficientnet_folds as cv
    import sciona.cassava_efficientnet_refit as refit
    import sciona.cassava_cropnet as cropnet
    keys = [f'synthetic-{i}' for i in range(10)]
    plan = build_plan(keys, [i % 5 for i in range(10)], [i // 2 for i in range(10)])
    def image(value, shape):
        ok, encoded = cv2.imencode('.jpg', np.full(shape, value, np.uint8))
        assert ok
        return encoded.tobytes()
    source = [(key, image(40 + i, (600, 800, 3))) for i, key in enumerate(keys)]
    prepared = [(key, image(40 + i, (512, 512, 3))) for i, key in enumerate(keys)]
    query = [('synthetic-query', image(99, (600, 800, 3)))]
    calls = []
    def probabilities(column):
        result = np.zeros((1, 5))
        result[0, column] = 1.
        return result
    def torch_train(family, actual_plan, training, query_keys, queries, **settings):
        assert actual_plan is plan and training == tuple(row[1] for row in source)
        assert tuple(query_keys) == ('synthetic-query',) and queries == (query[0][1],)
        assert settings['expected_sha256'] == family
        calls.append(family)
        return probabilities(0 if family == 'vit' else 1), {}, {}, {}
    def cross_validate(actual_plan, training, **settings):
        assert actual_plan is plan and training == tuple(row[1] for row in prepared)
        assert settings['expected_sha256'] == 'efficientnet'
        calls.append('cv')
        return {}, {}, {}
    final_model, frozen_model = object(), object()
    def final_train(training, labels, **settings):
        assert training == tuple(row[1] for row in prepared) and labels == plan.labels
        assert settings['weight_format'] == 'upstream'
        calls.append('refit')
        return final_model, {}, tmp_path / 'synthetic-final'
    def final_predict(model, images):
        assert model is final_model and images[0].shape == (600, 800, 3)
        return probabilities(2)
    def frozen_predict(model, images):
        assert model is frozen_model and images[0].shape == (600, 800, 3)
        calls.append('cropnet')
        return probabilities(3)
    monkeypatch.setattr(torch_folds, 'train_five_folds', torch_train)
    monkeypatch.setattr(cv, 'train_five_folds', cross_validate)
    monkeypatch.setattr(refit, 'train_final', final_train)
    monkeypatch.setattr(refit, 'predict_images', final_predict)
    monkeypatch.setattr(cropnet, 'load_model', lambda *a, **k: frozen_model)
    monkeypatch.setattr(cropnet, 'predict_images', frozen_predict)
    references = {family: dict(path='synthetic', sha256=family) for family in ('vit', 'resnext', 'efficientnet')}
    references['cropnet'] = dict(directory='synthetic', files={})
    result, outputs, _, _, _ = execute(plan, source_rows=source[::-1], efficientnet_rows=prepared[::-1],
        query_rows=query, references=references, torch_batch_size=1, torch_workers=0,
        efficientnet_batch_size=1, output_directory=tmp_path / 'execution')
    assert calls == ['resnext', 'vit', 'cv', 'refit', 'cropnet']
    assert set(outputs) == {'resnext', 'vit', 'efficientnet', 'cropnet'}
    np.testing.assert_array_equal(result['scores'], [[.5, .5, 1., 1., 0.]])
    np.testing.assert_array_equal(result['labels'], [2])
