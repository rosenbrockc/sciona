import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.andriy_wavelet_features import andriy_wavelet_features


def test_zero_and_constant_windows_have_zero_detail():
    np.testing.assert_allclose(andriy_wavelet_features(np.array([np.zeros(7680), np.ones(7680)])), 0, atol=1e-14)


def test_mean_absolute_scaling_and_no_mutation():
    x = np.random.default_rng(551).normal(size=(2, 7680))
    original = x.copy()
    np.testing.assert_allclose(andriy_wavelet_features(-3*x), 3*andriy_wavelet_features(x), rtol=1e-13)
    np.testing.assert_array_equal(x, original)


def test_periodic_shift_by_scale_preserves_mean_for_divisible_length():
    x = np.random.default_rng(552).normal(size=(1, 7680))
    np.testing.assert_allclose(andriy_wavelet_features(np.roll(x, 32, axis=1)), andriy_wavelet_features(x), rtol=1e-13)


@pytest.mark.parametrize('x', [np.ones(128), np.ones((1, 127)), np.ones((0, 128)), np.full((1, 128), np.nan), np.full((1, 128), np.inf), np.ones((1, 128), dtype=complex)])
def test_invalid_windows(x):
    with pytest.raises(ValueError):
        andriy_wavelet_features(x)
