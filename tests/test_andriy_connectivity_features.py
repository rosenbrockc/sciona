import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.andriy_connectivity_features import andriy_connectivity_features


def test_identical_channels_have_zero_lag_and_perfect_symmetry():
    row = np.random.default_rng(571).normal(size=768)
    x = np.broadcast_to(row, (1,16,768)).copy()
    original = x.copy()
    result = andriy_connectivity_features(x).reshape(6,6,5)
    np.testing.assert_array_equal(result[:,0], 0)
    np.testing.assert_allclose(result[:,1], 0, atol=1e-13)
    np.testing.assert_allclose(result[:,2], 1, atol=1e-13)
    np.testing.assert_allclose(result[:,4:], 1, atol=1e-13)
    np.testing.assert_array_equal(x, original)


def test_final_montage_reuses_fifth_montage_coherence():
    x = np.random.default_rng(572).normal(size=(1,16,768))
    result = andriy_connectivity_features(x).reshape(6,6,5)
    np.testing.assert_array_equal(result[5,4], result[4,4])
    assert not np.array_equal(result[5,2], result[4,2])


def test_zero_window_is_masked_without_affecting_other_windows():
    x = np.random.default_rng(573).normal(size=(1,16,768))
    actual = andriy_connectivity_features(np.concatenate([np.zeros_like(x), x]))
    assert np.isnan(actual[0]).all()
    np.testing.assert_array_equal(actual[1:], andriy_connectivity_features(x))


@pytest.mark.parametrize('x', [np.ones((16,768)), np.ones((1,15,768)), np.ones((1,16,254)), np.ones((0,16,768)), np.full((1,16,768),np.nan), np.full((1,16,768),np.inf), np.ones((1,16,768),dtype=complex)])
def test_invalid_windows(x):
    with pytest.raises(ValueError):
        andriy_connectivity_features(x)
