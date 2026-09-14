import numpy as np
import pandas as pd
import pytest

from sciona.m5_encoding import encode


def test_cutoff_inclusive_and_future_targets_cannot_change_features():
    values = [2., 6., 999., 10., np.nan]
    times = [1, 2, 3, 1, 4]
    keys = [[0], [0], [0], [1], [2]]
    actual = encode(values, times, keys, 2, [[0]])
    expected = np.array([[4, np.sqrt(8)]] * 3 + [[10, np.nan], [np.nan, np.nan]], dtype=np.float16)
    np.testing.assert_equal(actual, expected)
    values[2] = -1e20
    np.testing.assert_equal(encode(values, times, keys, 2, [[0]]), expected)


def test_requested_role_combinations_against_pandas():
    rng = np.random.default_rng(455)
    keys = rng.integers(0, 4, (800, 5))
    values = rng.normal(50, 10, 800)
    values[::19] = np.nan
    times = rng.integers(0, 90, 800)
    groups = [[0], [1], [2], [3], [0,2], [0,3], [1,2], [1,3], [4], [4,0], [4,1]]
    actual = encode(values, times, keys, 61, groups)
    frame = pd.DataFrame(keys)
    frame['target'] = np.where(times <= 61, values, np.nan)
    expected = np.column_stack([frame.groupby(group).target.transform(stat).to_numpy()
                                for group in groups for stat in ('mean', 'std')]).astype(np.float16)
    np.testing.assert_equal(actual, expected)
    assert actual.dtype == np.float16 and actual.shape == (800,22)


def test_alignment_and_permutation_invariance():
    x = np.array([1., 3., 8., 12.])
    t = np.array([0, 1, 0, 1])
    k = np.array([[5], [5], [6], [6]])
    order = [3, 0, 2, 1]
    np.testing.assert_equal(encode(x[order], t[order], k[order], 1, [[0]]), encode(x,t,k,1,[[0]])[order])


@pytest.mark.parametrize('groupings', [[], [[0,0]], [[1]], [[True]], [[0],[0]], [0]])
def test_invalid_grouping_rejects(groupings):
    with pytest.raises(ValueError):
        encode([1.], [0], [[0]], 0, groupings)


def test_invalid_types_and_overflow_reject():
    with pytest.raises(ValueError):
        encode([np.inf], [0], [[0]], 0, [[0]])
    with pytest.raises(ValueError):
        encode([1.], [0], [[np.nan]], 0, [[0]])
    with pytest.raises(ValueError):
        encode([1.], [0], [[0]], True, [[0]])
    with pytest.raises(ValueError, match='float16'):
        encode([1e6], [0], [[0]], 0, [[0]])
