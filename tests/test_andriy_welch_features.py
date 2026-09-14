import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.andriy_welch_features import andriy_documented_welch_features


def test_tone_location_and_amplitude_invariance():
    x = np.sin(2*np.pi*17*np.arange(7680)/256)[None, :]
    original = x.copy()
    actual = andriy_documented_welch_features(x)
    assert actual[0, 0] == pytest.approx(17, abs=.01)
    assert 0 < actual[0, 1] < 1
    np.testing.assert_allclose(andriy_documented_welch_features(-3*x), actual, rtol=1e-12)
    np.testing.assert_array_equal(x, original)


def test_dc_is_not_detrended_and_zero_input_is_undefined():
    assert np.isfinite(andriy_documented_welch_features(np.ones((1, 7680)))).all()
    assert np.isnan(andriy_documented_welch_features(np.zeros((1, 7680)))).all()


def test_truncated_tail_does_not_change_moments():
    x = np.random.default_rng(561).normal(size=(1, 7680))
    changed = x.copy()
    # 1706-sample windows, step 853: last included sample is index 7676.
    changed[0, 7677:] = 1e6
    np.testing.assert_array_equal(andriy_documented_welch_features(x), andriy_documented_welch_features(changed))


@pytest.mark.parametrize('x', [np.ones(32), np.ones((1, 31)), np.ones((0, 32)), np.full((1, 32), np.nan), np.full((1, 32), np.inf), np.ones((1, 32), dtype=complex)])
def test_invalid_windows(x):
    with pytest.raises(ValueError):
        andriy_documented_welch_features(x)
