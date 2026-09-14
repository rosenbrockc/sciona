import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.andriy_mainstream_features import andriy_documented_mainstream_features


def test_strict_low_range_mask_preserves_other_rows():
    ramp = np.linspace(0, 1, 256)
    x = np.array([np.zeros(256), ramp*.0099, ramp*.01, ramp])
    original = x.copy()
    actual = andriy_documented_mainstream_features(x)
    assert actual.shape == (4, 102)
    assert np.isnan(actual[:2]).all()
    assert np.isfinite(actual[2:]).all()
    np.testing.assert_array_equal(x, original)


def test_source_feature_positions_on_sinusoid():
    x = np.sin(2*np.pi*7*np.arange(7680)/256)[None, :]
    actual = andriy_documented_mainstream_features(x)[0]
    assert actual[3] == pytest.approx(np.sqrt(.5))
    assert actual[4] == pytest.approx(7+256/7680)
    assert actual[5] == pytest.approx(.5)
    assert actual[12] == pytest.approx(1.5)
    assert actual[20] == pytest.approx(7680/4)
    assert actual[86] == pytest.approx(7, abs=.01)
    assert actual[90] == actual[93]
    assert actual[97] == actual[100]


@pytest.mark.parametrize('x', [np.ones(256), np.ones((1, 255)), np.ones((1, 257)), np.ones((0, 256)), np.full((1, 256), np.nan)])
def test_invalid_windows(x):
    with pytest.raises(ValueError):
        andriy_documented_mainstream_features(x)
