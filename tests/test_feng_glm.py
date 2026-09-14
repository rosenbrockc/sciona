import numpy as np
import pytest
from sciona.atoms.ml.sklearn.linear_model.feng_glm import feng_glm_segment_probabilities


def test_training_scaler_reuse_predicts_positive_constant_batch():
    train = np.r_[np.full((3, 12, 2), -2.), np.full((3, 12, 2), 2.)]
    labels = np.array([0]*3+[1]*3)
    prediction = np.full((2, 12, 2), 2.)
    before = train.copy()
    result = feng_glm_segment_probabilities(train, labels, prediction)
    assert result.shape == (2,) and np.all(result > .95)
    np.testing.assert_array_equal(train, before)


def test_prediction_batch_composition_does_not_change_existing_scores():
    rng = np.random.default_rng(561)
    train = rng.normal(size=(6, 12, 4))
    labels = np.array([0, 1]*3)
    prediction = rng.normal(size=(2, 12, 4))
    expected = feng_glm_segment_probabilities(train, labels, prediction)
    actual = feng_glm_segment_probabilities(train, labels, np.r_[prediction, prediction+11])
    np.testing.assert_allclose(actual[:2], expected, atol=1e-14)


def test_negative_infinity_cleanup_preserves_input():
    rng = np.random.default_rng(562)
    train, prediction = rng.normal(size=(6, 12, 4)), rng.normal(size=(2, 12, 4))
    labels = np.array([0, 1]*3)
    train[0, 0, 0] = -np.inf
    expected = feng_glm_segment_probabilities(np.where(np.isneginf(train), 0, train), labels, prediction)
    np.testing.assert_array_equal(feng_glm_segment_probabilities(train, labels, prediction), expected)
    assert np.isneginf(train[0, 0, 0])


@pytest.mark.parametrize('bad', ['nan', 'inf', 'labels', 'width', 'empty', 'complex'])
def test_invalid_contract(bad):
    train, prediction, labels = np.zeros((4, 12, 3)), np.zeros((2, 12, 3)), np.array([0, 1, 0, 1])
    if bad == 'nan': train[0, 0, 0] = np.nan
    if bad == 'inf': prediction[0, 0, 0] = np.inf
    if bad == 'labels': labels = np.zeros(4)
    if bad == 'width': prediction = np.zeros((2, 12, 4))
    if bad == 'empty': prediction = prediction[:0]
    if bad == 'complex': train = train.astype(complex)
    with pytest.raises(ValueError):
        feng_glm_segment_probabilities(train, labels, prediction)
