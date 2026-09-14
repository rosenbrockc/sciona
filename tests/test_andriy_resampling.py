import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.andriy_resampling import andriy_documented_resample


@pytest.mark.parametrize('target', [128, 256])
def test_passband_amplitude_delay_and_antialiasing(target):
    t = np.arange(400*10+1)/400
    x = np.stack([np.sin(2*np.pi*7*t), np.sin(2*np.pi*180*t)])
    before = x.copy()
    actual = andriy_documented_resample(x, 400, target)
    assert actual.shape == (2, int(np.ceil(len(t)*target/400)))
    expected = np.sin(2*np.pi*7*np.arange(actual.shape[1])/target)
    np.testing.assert_allclose(actual[0, target:-target], expected[target:-target], atol=.003)
    assert np.max(abs(actual[1, target:-target])) < .003
    np.testing.assert_array_equal(x, before)


def test_identity_copy_and_impulse_alignment():
    x = np.zeros((2, 4001));x[:, 2000] = [1, -2]
    y = andriy_documented_resample(x, 400, 128)
    assert np.argmax(abs(y[0])) == 640
    np.testing.assert_array_equal(y[1], -2*y[0])
    copy = andriy_documented_resample(x, 128, 128)
    np.testing.assert_array_equal(copy, x)
    assert not np.shares_memory(copy, x)


@pytest.mark.parametrize('rates', [(0, 128), (400.5, 128), (True, 128), (400, 200)])
def test_invalid_rates(rates):
    with pytest.raises(ValueError):
        andriy_documented_resample(np.zeros((1, 20)), *rates)


@pytest.mark.parametrize('x', [np.ones(10), np.ones((0, 10)), np.full((1, 10), np.nan), np.ones((1, 10), dtype=complex)])
def test_invalid_signal(x):
    with pytest.raises(ValueError):
        andriy_documented_resample(x, 400, 128)
