import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.andriy_clip_features import andriy_documented_clip_features


def test_high_range_gate_imputation_csp_exception_and_tail_exclusion():
    rng=np.random.default_rng(641)
    signal=rng.normal(size=(16,15360))
    csp=rng.normal(size=(1,7680))
    signal[0,:7680]=np.linspace(-500,500,7680)  # equality must be rejected
    csp[0,:3840]*=1000  # CSP has no upper-range gate
    original=signal.copy(),csp.copy()
    features,valid=andriy_documented_clip_features(signal,csp)
    assert features.shape==(2,1965) and valid.all()
    np.testing.assert_array_equal(features[0,:1632:16],features[1,:1632:16])
    np.testing.assert_array_equal(features[0,1632:1776:16],features[1,1632:1776:16])
    assert not np.array_equal(features[0,1776:1785],features[1,1776:1785])
    extended=np.pad(signal,((0,0),(0,17)),constant_values=1e6)
    extended_csp=np.pad(csp,((0,0),(0,7)),constant_values=1e6)
    actual,mask=andriy_documented_clip_features(extended,extended_csp)
    np.testing.assert_array_equal(actual,features)
    np.testing.assert_array_equal(mask,valid)
    np.testing.assert_array_equal(signal,original[0])
    np.testing.assert_array_equal(csp,original[1])


def test_all_missing_clip_is_retained_as_invalid():
    features,valid=andriy_documented_clip_features(np.zeros((16,7680)),np.zeros((1,3840)))
    assert np.isnan(features).all()
    np.testing.assert_array_equal(valid,[False])


@pytest.mark.parametrize('signal,csp',[
    (np.ones((15,7680)),np.ones((1,3840))),
    (np.ones((16,7679)),np.ones((1,3840))),
    (np.ones((16,7680)),np.ones((2,3840))),
    (np.ones((16,7680)),np.ones((1,7680))),
    (np.full((16,7680),np.nan),np.ones((1,3840))),
])
def test_invalid_clips(signal,csp):
    with pytest.raises(ValueError):
        andriy_documented_clip_features(signal,csp)
