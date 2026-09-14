"""Synthetic numeric feature oracles including missing values and floor quirks."""
import numpy as np
import pytest
from sciona.amex_numeric import denoise_numeric,summarize


def test_denoising_floor_negative_and_nan():
    x=np.array([[.019,-.019,np.nan],[1.001,-1.001,0.]])
    np.testing.assert_allclose(denoise_numeric(x),[[1,-2,np.nan],[100,-101,0]],equal_nan=True)
    assert x[0,0]==.019


def test_aggregate_scalar_formulas_and_feature_order():
    x=np.array([[1.,2.],[2.,4.],[3.,6.]])
    expected=np.array([[2,1,1,3,6,3],[4,2,2,6,12,6]],float)//.01
    np.testing.assert_array_equal(summarize(x),expected.ravel())
    assert summarize([[1.]])[0]==99. # Floating floor division, not rounded scaling.


def test_missing_last_and_all_missing_sum():
    x=[[1.,np.nan],[3.,np.nan],[np.nan,np.nan]]
    v=summarize(x).reshape(2,6)
    assert v[0,-1]==3.//.01 and v[1,4]==0
    assert np.isnan(v[1,[0,1,2,3,5]]).all()


def test_difference_retains_missing_adjacencies():
    x=[[1.],[3.],[np.nan],[7.],[10.]]
    expected=np.array([2.5,np.sqrt(.5),2,3,5,3])//.01
    np.testing.assert_allclose(summarize(x,differences=True),expected)
    assert summarize([[5.]],differences=True)[4]==0


def test_window_and_rank_modes():
    np.testing.assert_array_equal(summarize([[1.],[3.],[5.]],last_window=2),np.array([4,np.sqrt(2),3,5,8])//.01)
    np.testing.assert_array_equal(summarize([[.1],[.7],[np.nan]],ranked=True),[.7])


@pytest.mark.parametrize('x',[[],[1,2],[[True,False]],[[float('inf')]],[["1"]]])
def test_invalid_values(x):
    with pytest.raises(ValueError):denoise_numeric(x)


def test_invalid_modes():
    with pytest.raises(ValueError):summarize([[1]],last_window=True)
    with pytest.raises(ValueError):summarize([[1]],ranked=True,differences=True)
