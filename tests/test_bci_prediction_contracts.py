import numpy as np
import pandas as pd
import pytest
from sciona.atoms.ml.xgboost.competition_bagging import bagged_window_probabilities
from sciona.atoms.riemannian_bci.signal_processing.segment_scores import segment_probability_max, normalized_rank_blend


def test_segment_max_matches_groupby_with_unsorted_variable_window_counts():
    p = np.array([.1, .6, .4, .8, .2, .3])
    ids = np.array([2, 0, 1, 2, 0, 2])
    expected = pd.Series(p).groupby(ids).max().to_numpy()
    np.testing.assert_array_equal(segment_probability_max(p, ids, 3), expected)


def test_rank_blend_ties_weights_and_no_final_reranking():
    p = np.array([[.1, .1, .8, .4], [.7, .2, .3, .1]])
    # Per-row average ranks: [1.5,1.5,4,3], [4,2,3,1].
    expected = (np.array([1.5, 1.5, 4., 3.]) + 3 * np.array([4., 2., 3., 1.])) / 16.
    np.testing.assert_array_equal(normalized_rank_blend(p, np.array([1., 3.])), expected)


@pytest.mark.parametrize('p,ids,n', [([.1], [1], 2), ([.1], [0.], 1), ([1.2], [0], 1), ([np.nan], [0], 1)])
def test_invalid_segment_contract_rejected(p, ids, n):
    with pytest.raises(ValueError):
        segment_probability_max(np.array(p), np.array(ids), n)


@pytest.mark.parametrize('weights', [[0., 0.], [-1., 2.], [np.nan, 1.]])
def test_invalid_blend_weights_rejected(weights):
    with pytest.raises(ValueError):
        normalized_rank_blend(np.ones((2, 3)), np.array(weights))


def test_window_classifier_uses_segment_labels_and_positive_class():
    rng = np.random.default_rng(62)
    labels = np.array([1, 0, 1, 0])
    ids = np.tile(np.arange(4), 30)
    x = rng.normal(scale=.1, size=(len(ids), 8)) + labels[ids, None] * 4
    prediction = np.vstack([np.zeros((4, 8)), np.full((4, 8), 4.)])
    result = bagged_window_probabilities(x, ids, labels, prediction, 2, 32)
    assert result.shape == (8,)
    assert np.all((result >= 0) & (result <= 1))
    assert result[:4].mean() < result[4:].mean()


def test_classifier_rejects_unrepresented_segments():
    with pytest.raises(ValueError, match='indices'):
        bagged_window_probabilities(np.ones((10, 3)), np.zeros(10, dtype=int),
                                    np.array([0, 1]), np.ones((2, 3)), 2, 3)
