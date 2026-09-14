import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.andriy_time_features import andriy_time_features


def test_strict_crossings_duplicate_minmax_and_variance_conventions():
    x = np.array([[-1., 0., 1., -2., 2.]])
    a = andriy_time_features(x)[0]
    assert a[0] == 2 and a[1] == a[6] == 2
    assert a[3] == np.var(x)
    assert a[8] == np.var(np.diff(x[0]), ddof=1)
    assert a[9] == np.var(np.diff(x[0], n=2), ddof=1)
    assert a[2] == np.sqrt(2.)


def test_constant_source_undefined_ratios_and_no_mutation():
    x = np.ones((1, 10));a = andriy_time_features(x)[0]
    assert a[2] == 1 and a[3] == 0 and a[4] == 0 and np.isnan(a[5])
    np.testing.assert_array_equal(x, np.ones((1, 10)))


@pytest.mark.parametrize('x', [np.ones(10), np.ones((1, 3)), np.ones((0, 5)), np.full((1, 5), np.nan)])
def test_invalid_windows(x):
    with pytest.raises(ValueError):
        andriy_time_features(x)
