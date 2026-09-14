import numpy as np
import pandas as pd
import pytest

from sciona.m5_lags import features, rolling, shifted


def test_interleaved_groups_preserve_order_and_causality():
    groups = np.array([5, 7, 5, 7, 5, 7, 5, 7])
    values = np.array([1., 20., 3., 40., 5., 60., 7., 80.])
    np.testing.assert_equal(shifted(values, groups, 2), [np.nan]*4 + [1., 20., 3., 40.])
    np.testing.assert_equal(rolling(values, groups, 1, 2), [np.nan]*4 + [2., 30., 4., 50.])
    changed = values.copy()
    changed[-2:] = 999
    np.testing.assert_equal(rolling(values, groups, 1, 2), rolling(changed, groups, 1, 2))


def test_full_windows_and_sample_deviation():
    x = [1., 3., np.nan, 9., 11., 13.]
    g = [0]*6
    np.testing.assert_equal(rolling(x, g, 1, 2), [np.nan, np.nan, 2., np.nan, np.nan, 10.])
    np.testing.assert_allclose(rolling(x, g, 1, 2, 'std'), [np.nan, np.nan, np.sqrt(2), np.nan, np.nan, np.sqrt(2)], equal_nan=True)
    assert np.isnan(rolling(x, g, 1, 1, 'std')).all()


def test_all_features_against_grouped_pandas_oracle():
    rng = np.random.default_rng(717)
    groups = np.tile([3, 8], 240)
    values = rng.uniform(0, 20, len(groups))
    values[11] = np.nan
    frame = pd.DataFrame({'group': groups, 'value': values})
    stored = features(values, groups)
    live = features(values, groups, recursive=True)
    assert len(stored) == len(live) == 37
    for name, result in stored.items():
        parts = name.split('_')
        if parts[0] == 'lag':
            expected = frame.groupby('group').value.transform(lambda s: s.shift(int(parts[1])))
        else:
            shift = int(parts[1]) if parts[0] == 'temporary' else 28
            window = int(parts[-1])
            statistic = 'std' if parts[0] == 'std' else 'mean'
            expected = frame.groupby('group').value.transform(lambda s: getattr(s.shift(shift).rolling(window), statistic)())
        np.testing.assert_equal(result, expected.to_numpy().astype(np.float16))
        assert result.dtype == np.float16
        if name.startswith('temporary'):
            np.testing.assert_allclose(live[name], expected, rtol=1e-12, atol=1e-12, equal_nan=True)
            assert live[name].dtype == np.float64
        else:
            np.testing.assert_equal(live[name], result)


@pytest.mark.parametrize('values,groups', [([], []), ([True], [0]), ([np.inf], [0]), ([1.], ['a']), ([1.,2.], [0]), ([[1.]], [0])])
def test_invalid_population_rejects(values, groups):
    with pytest.raises(ValueError):
        features(values, groups)


@pytest.mark.parametrize('shift', [0, -1, 1.5, True])
def test_invalid_shift_rejects(shift):
    with pytest.raises(ValueError):
        shifted([1.], [0], shift)


def test_storage_overflow_rejects():
    with pytest.raises(ValueError, match='float16'):
        features(np.full(50, 1e6), np.zeros(50, dtype=int))
