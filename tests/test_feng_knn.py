import numpy as np
import pytest
from sciona.atoms.ml.sklearn.neighbors.feng_knn import feng_knn_segment_probabilities


def test_exact_duplicate_neighbors_and_mean_segment_aggregation():
    train = np.r_[np.zeros((2, 1, 1, 20)), np.full((2, 1, 1, 20), 2.)]
    prediction = np.array([[[[0., 0., 2., 2.]]], [[[0., 2., 2., 0.]]]])
    result = feng_knn_segment_probabilities(train, np.array([0, 0, 1, 1]), prediction)
    np.testing.assert_array_equal(result, [.5, .5])


def test_matches_independent_manhattan_distance_weighting():
    rng = np.random.default_rng(413)
    train = rng.normal(size=(4, 2, 3, 12))
    prediction = rng.normal(size=(3, 2, 3, 5)) + 17
    labels = np.array([0, 1, 1, 0])
    before = train.copy()
    def scalar_rows(tensor):
        a = tensor.astype(np.float32)
        rows = np.array([[a[s, c, f, w] for c in range(a.shape[1]) for f in range(a.shape[2])]
                         for s in range(len(a)) for w in range(a.shape[3])], dtype=float)
        return (rows-rows.mean(axis=0))/rows.std(axis=0)
    x, xp = scalar_rows(train), scalar_rows(prediction)
    y = np.repeat(labels, 12)
    expected = []
    for row in xp:
        distances = np.abs(x-row).sum(axis=1)
        ids = np.argsort(distances)[:40]
        weights = 1/distances[ids]
        expected.append(np.sum(weights*y[ids])/weights.sum())
    expected = np.array(expected).reshape(3, 5).mean(axis=1)
    np.testing.assert_allclose(feng_knn_segment_probabilities(train, labels, prediction), expected, rtol=1e-14)
    np.testing.assert_array_equal(train, before)


def test_source_cleanup_and_no_input_mutation():
    rng = np.random.default_rng(414)
    train, prediction = rng.normal(size=(4, 1, 2, 20)), rng.normal(size=(2, 1, 2, 20))
    train[0, 0, 0, 0] = np.nan
    prediction[0, 0, 0, 1] = -np.inf
    labels = np.array([0, 1, 0, 1])
    expected = feng_knn_segment_probabilities(np.nan_to_num(train, nan=0), labels, np.where(np.isneginf(prediction), 0, prediction))
    np.testing.assert_array_equal(feng_knn_segment_probabilities(train, labels, prediction), expected)
    assert np.isnan(train[0, 0, 0, 0]) and np.isneginf(prediction[0, 0, 0, 1])


@pytest.mark.parametrize('bad', ['too_few', 'axis_mismatch', 'labels', 'positive_inf', 'complex', 'empty'])
def test_invalid_contract(bad):
    train, prediction, labels = np.zeros((2, 2, 3, 20)), np.zeros((1, 2, 3, 20)), np.array([0, 1])
    if bad == 'too_few': train = train[..., :19]
    if bad == 'axis_mismatch': prediction = np.zeros((1, 3, 2, 20))
    if bad == 'labels': labels = np.array([0, 0])
    if bad == 'positive_inf': prediction[0, 0, 0, 0] = np.inf
    if bad == 'complex': train = train.astype(complex)
    if bad == 'empty': prediction = prediction[:0]
    with pytest.raises(ValueError):
        feng_knn_segment_probabilities(train, labels, prediction)
