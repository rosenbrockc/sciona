import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.andriy_spectral_features import andriy_spectral_features


def test_source_peak_offset_power_scale_and_duplicate_bands():
    n = 7680;t = np.arange(n)/256
    x = np.sin(2*np.pi*7*t)[None,:]
    before = x.copy();a = andriy_spectral_features(x)[0]
    assert a.shape == (82,)
    assert a[80] == 7+256/n
    np.testing.assert_allclose(a[0], n/4, rtol=1e-13)
    assert a[68] == a[71] and a[75] == a[78]
    np.testing.assert_array_equal(x,before)


def test_zero_spectrum_keeps_source_undefined_relative_powers():
    a = andriy_spectral_features(np.zeros((1,256)))[0]
    assert np.isnan(a[32:63]).all() and np.isnan(a[73:80]).all()
    np.testing.assert_array_equal(a[63:66],[2,2,2])
    assert a[80] == 2 and a[81] == 0


@pytest.mark.parametrize('x',[np.ones(256),np.ones((1,255)),np.ones((1,257)),np.full((1,256),np.nan)])
def test_invalid_spectral_domain(x):
    with pytest.raises(ValueError):
        andriy_spectral_features(x)
