import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.andriy_ar_features import andriy_documented_ar_features, _estimated_prediction


def test_ar1_estimates_initial_state_instead_of_zero_initialization():
    x=np.array([10.,6.,2.,-1.])
    np.testing.assert_allclose(_estimated_prediction(x,np.array([1.,-.5])),[10.,5.,3.,1.])


def test_zero_model_cannot_fit_an_unobservable_initial_state():
    x=np.arange(20.)
    np.testing.assert_array_equal(_estimated_prediction(x,np.array([1.,0.,0.])),0)


def test_effective_order_limits_initial_prefix_correction():
    x=np.arange(1.,21.)
    short=_estimated_prediction(x,np.array([1.,-.5]))
    long=_estimated_prediction(x,np.array([1.,-.5,0.,0.]))
    np.testing.assert_allclose(short,long,atol=1e-14)
    assert long[1]!=x[1]


def test_mask_and_sentinel_precedence():
    ramp=np.linspace(-1,1,20)
    x=np.array([np.zeros(40),np.r_[np.zeros(20),ramp],np.r_[ramp,np.zeros(20)],np.r_[np.zeros(20),np.ones(20)]])
    actual=andriy_documented_ar_features(x)
    assert np.isnan(actual[0]).all()
    np.testing.assert_array_equal(actual[1:],np.array([[50.]*9,[100.]*9,[50.]*9]))


def test_odd_middle_sample_is_omitted_and_inputs_unchanged():
    x=np.random.default_rng(621).normal(size=(1,101))
    original=x.copy()
    changed=x.copy();changed[0,50]=1e6
    np.testing.assert_array_equal(andriy_documented_ar_features(changed),andriy_documented_ar_features(x))
    np.testing.assert_array_equal(x,original)


def test_amplitude_and_sign_invariance_above_thresholds():
    x=np.random.default_rng(622).normal(size=(2,300))
    np.testing.assert_allclose(andriy_documented_ar_features(-7*x),andriy_documented_ar_features(x),atol=1e-10,rtol=1e-10)


@pytest.mark.parametrize('x',[np.ones(30),np.ones((0,30)),np.ones((1,19)),np.full((1,30),np.nan),np.ones((1,30),dtype=complex)])
def test_invalid_windows(x):
    with pytest.raises(ValueError):
        andriy_documented_ar_features(x)
