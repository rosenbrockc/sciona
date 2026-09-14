"""Synthetic phase boundaries, missing-group semantics and full measurement path."""
import numpy as np
import pytest
from sciona.vsb_measurement import fundamental,phase_quadrants,aggregate,measurement_features


def test_fourier_scalar_oracle():
    x=np.random.default_rng(93).normal(size=(128,3))
    expected=[sum(x[t,j]*np.exp(-2j*np.pi*t/128) for t in range(128)) for j in range(3)]
    np.testing.assert_allclose(fundamental(x),expected,atol=5e-14)


def test_right_closed_phase_boundaries():
    np.testing.assert_array_equal(phase_quadrants(np.array([0,1,2,3]),1+0j,length=4),[0,1,2,-1])
    np.testing.assert_array_equal(phase_quadrants(np.array([0]),-1j,length=4),[-1])


def test_filter_counts_sample_std_and_nan_means():
    g=np.array([0,0,0,0,1]);h=np.array([10.,20.,60.,50.,2.])
    f=np.array([[.2,.4,-2,.1],[np.nan,.8,4,.3],[.4,.2,1,.2],[.4,np.nan,3,.5],[.1,.2,1,.1]])
    q=np.array([0,2,0,1,-1])
    got=aggregate(g,h,f,q,measurement_count=3)
    np.testing.assert_allclose(got[0],[2,3,1,15,np.sqrt(50),.6,.2,3,.2])
    assert got[1,1]==1 and np.isnan(got[1,[0,2,3,4,5,6,7,8]]).all()
    assert np.isnan(got[2]).all()


def test_singleton_and_empty_groups():
    x=aggregate([0],[5.],[[.1,.2,-1,.3]],[0],measurement_count=2)
    assert np.isnan(x[0,4]) and np.isnan(x[1]).all()
    assert np.isnan(aggregate(np.array([],int),np.array([],float),np.empty((0,4)),np.array([],int),measurement_count=2)).all()


def test_actual_three_signal_join_and_permutation():
    x=np.random.default_rng(94).normal(size=(4096,6))
    result=measurement_features(x)
    assert result.shape==(2,9) and np.isfinite(result).all()
    np.testing.assert_allclose(measurement_features(x[:,[3,4,5,0,1,2]]),result[::-1])
    with pytest.raises(ValueError,match='three-signal'):measurement_features(x[:,:4])


def test_invalid_inputs():
    with pytest.raises(ValueError):fundamental([[True],[False],[True]])
    with pytest.raises(ValueError):phase_quadrants([.5],1,length=4)
    with pytest.raises(ValueError):aggregate([1],[1.],[[0.,0.,0.,0.]],[0],measurement_count=1)
    with pytest.raises(ValueError):aggregate([0],[1.],[[0.,0.,np.nan,0.]],[0],measurement_count=1)
