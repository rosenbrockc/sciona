import numpy as np
import pytest
from scipy.linalg import toeplitz
from sciona.atoms.riemannian_bci.signal_processing.andriy_ar_coefficients import andriy_ar_coefficients


def test_all_orders_match_uncentered_yule_walker_equations():
    x=np.random.default_rng(601).normal(size=(2,300))+4
    original=x.copy()
    actual=andriy_ar_coefficients(x)
    for index,row in enumerate(x):
        correlations=np.array([row[:len(row)-lag]@row[lag:] for lag in range(10)])
        for order in range(1,10):
            expected=np.linalg.solve(toeplitz(correlations[:order]),-correlations[1:order+1])
            np.testing.assert_allclose(actual[index,order-1,1:order+1],expected,rtol=1e-11,atol=1e-12)
            assert actual[index,order-1,0]==1
            assert not np.any(actual[index,order-1,order+1:])
    np.testing.assert_array_equal(x,original)


def test_zero_series_has_identity_polynomials():
    actual=andriy_ar_coefficients(np.zeros((1,30)))
    np.testing.assert_array_equal(actual[:,:,0],1)
    np.testing.assert_array_equal(actual[:,:,1:],0)


def test_amplitude_and_sign_invariance():
    x=np.random.default_rng(602).normal(size=(2,300))
    np.testing.assert_allclose(andriy_ar_coefficients(-7*x),andriy_ar_coefficients(x),atol=1e-13)


@pytest.mark.parametrize('x',[np.ones(30),np.ones((0,30)),np.ones((1,9)),np.full((1,30),np.nan),np.ones((1,30),dtype=complex)])
def test_invalid_model_windows(x):
    with pytest.raises(ValueError):
        andriy_ar_coefficients(x)
