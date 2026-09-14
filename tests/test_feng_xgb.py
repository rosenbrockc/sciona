import numpy as np
import pytest
from sciona.atoms.ml.xgboost.feng_xgb import feng_xgb_segment_probabilities


def test_learned_class_direction_missing_values_and_input_nonmutation():
    rng = np.random.default_rng(540)
    labels = np.array([0, 1]*4)
    train = rng.normal(scale=.1, size=(8, 2, 3, 20)) + labels[:, None, None, None]*3
    prediction = np.stack([np.zeros((2, 3, 20)), np.full((2, 3, 20), 3.)])
    train[0, 0, 0, 0] = np.nan
    before = train.copy()
    actual = feng_xgb_segment_probabilities(train, labels, prediction)
    assert actual.shape == (2,) and actual.dtype == np.float32
    assert actual[0] < .1 and actual[1] > .9
    np.testing.assert_array_equal(train, before)


@pytest.mark.parametrize('bad', ['negative_inf', 'positive_inf', 'labels', 'shape', 'complex', 'empty'])
def test_invalid_contract(bad):
    train, prediction, labels = np.zeros((2, 2, 3, 20)), np.zeros((1, 2, 3, 20)), np.array([0, 1])
    if bad == 'negative_inf': train[0, 0, 0, 0] = -np.inf
    if bad == 'positive_inf': prediction[0, 0, 0, 0] = np.inf
    if bad == 'labels': labels = np.array([0, 0])
    if bad == 'shape': prediction = np.zeros((1, 3, 2, 20))
    if bad == 'complex': train = train.astype(complex)
    if bad == 'empty': train = train[:0]
    with pytest.raises(ValueError):
        feng_xgb_segment_probabilities(train, labels, prediction)
