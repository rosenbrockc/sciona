import numpy as np
import pytest
from scipy.signal import butter
from sciona.atoms.riemannian_bci.signal_processing.feng_filter import feng_resample_filter


def test_impulse_matches_scalar_causal_difference_equation():
    x = np.zeros((400, 2)); x[0] = [1, -2]
    before = x.copy()
    b, a = butter(5, np.array([.1, 180]) / 200, btype='band')
    expected = np.zeros_like(x)
    for n in range(len(x)):
        expected[n] = sum(b[k] * x[n-k] for k in range(min(n+1, len(b)))) - sum(a[k] * expected[n-k] for k in range(1, min(n+1, len(a))))
    actual = feng_resample_filter(x, 1)
    np.testing.assert_allclose(actual, expected.astype(np.float32), rtol=2e-3, atol=2e-6)
    assert actual.dtype == np.float32 and actual.shape == (400, 2)
    np.testing.assert_array_equal(x, before)
    assert not np.shares_memory(actual, x)


def test_default_source_duration_and_zero_input():
    actual = feng_resample_filter(np.zeros((803, 2)))
    assert actual.shape == (240000, 2)
    assert not np.any(actual)


@pytest.mark.parametrize('duration', [0, -1, 1.5, True])
def test_invalid_duration(duration):
    with pytest.raises(ValueError):
        feng_resample_filter(np.ones((10, 2)), duration)


@pytest.mark.parametrize('x', [np.ones(10), np.ones((1, 2)), np.ones((10, 0)), np.full((10, 2), np.nan), np.full((10, 2), np.inf), np.ones((10, 2), dtype=complex)])
def test_invalid_signal(x):
    with pytest.raises(ValueError):
        feng_resample_filter(x, 1)
