import numpy as np
import pytest
from scipy.signal import lfilter
from sciona.atoms.riemannian_bci.signal_processing.andriy_filter import andriy_filter_trim


def test_inclusive_matlab_trim_and_scalar_filter_oracle():
    fs = 200
    x = np.random.default_rng(201).normal(size=(2, 9*fs))
    before = x.copy()
    angle = 2*np.pi*60/fs
    b = [1, -2*np.cos(angle), 1]
    a = [1, -1.9*np.cos(angle), .95**2]
    reference = []
    for channel in x:
        centered = channel-channel.mean()
        notch = np.zeros_like(channel)
        for t in range(len(channel)):
            notch[t] = sum(b[k]*centered[t-k] for k in range(min(t+1, 3))) - sum(a[k]*notch[t-k] for k in range(1, min(t+1, 3)))
        high = lfilter([1, -1], [1, -(1-np.pi/fs)], notch)
        reference.append(high[4*fs-1:len(channel)-4*fs])
    actual = andriy_filter_trim(x, fs)
    assert actual.shape == (2, fs+1)
    np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-13)
    np.testing.assert_array_equal(x, before)
    assert not np.shares_memory(actual, x)


def test_constant_signal_centers_to_zero_at_minimum_length():
    result = andriy_filter_trim(np.full((2, 3200), 7.), 400)
    np.testing.assert_array_equal(result, np.zeros((2, 1)))


@pytest.mark.parametrize('frequency', [120, 0, 200.5, True])
def test_invalid_sampling_frequency(frequency):
    with pytest.raises(ValueError):
        andriy_filter_trim(np.zeros((1, 3200)), frequency)


@pytest.mark.parametrize('x', [np.zeros((1, 3199)), np.zeros((0, 3200)), np.ones(3200),
                             np.full((1, 3200), np.nan), np.ones((1, 3200), dtype=complex)])
def test_invalid_signal(x):
    with pytest.raises(ValueError):
        andriy_filter_trim(x, 400)
