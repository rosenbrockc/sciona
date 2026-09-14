import numpy as np
import pytest
from sciona.atoms.ml.sklearn.neighbors.feng_expanded_knn import feng_expanded_knn_segment_probabilities


def test_exact_duplicates_average_windows_and_preserve_inputs():
    train = np.r_[np.zeros((2, 12, 3)), np.full((2, 12, 3), 2.)]
    prediction = np.array([[[0.]*3, [2.]*3], [[2.]*3, [0.]*3]])
    before = train.copy()
    np.testing.assert_array_equal(feng_expanded_knn_segment_probabilities(train, np.array([0, 0, 1, 1]), prediction), [.5, .5])
    np.testing.assert_array_equal(train, before)


def test_source_cleanup_matches_zero_features():
    rng = np.random.default_rng(751)
    train, prediction = rng.normal(size=(4, 12, 5)), rng.normal(size=(2, 12, 5))
    labels = np.array([0, 1, 0, 1])
    train[0, 0, 0] = np.nan
    prediction[0, 0, 0] = -np.inf
    expected = feng_expanded_knn_segment_probabilities(np.nan_to_num(train, nan=0), labels, np.where(np.isneginf(prediction), 0, prediction))
    np.testing.assert_array_equal(feng_expanded_knn_segment_probabilities(train, labels, prediction), expected)
    assert np.isnan(train[0, 0, 0]) and np.isneginf(prediction[0, 0, 0])


@pytest.mark.parametrize('bad', ['few_windows', 'width', 'labels', 'inf', 'complex', 'empty'])
def test_invalid_contract(bad):
    train, prediction, labels = np.zeros((4, 12, 3)), np.zeros((2, 12, 3)), np.array([0, 1, 0, 1])
    if bad == 'few_windows': train = train[:, :9]
    if bad == 'width': prediction = np.zeros((2, 12, 4))
    if bad == 'labels': labels = np.zeros(4)
    if bad == 'inf': prediction[0, 0, 0] = np.inf
    if bad == 'complex': train = train.astype(complex)
    if bad == 'empty': prediction = prediction[:0]
    with pytest.raises(ValueError):
        feng_expanded_knn_segment_probabilities(train, labels, prediction)
