"""Synthetic edge semantics for reconstructed VSB peak selection."""
import numpy as np
import pytest
from sciona.vsb_peaks import window_maxima,plateau_location,select_peaks


def test_plateau_midpoint_intersection():
    np.testing.assert_array_equal(window_maxima([0,2,0],window=1),[1])
    np.testing.assert_array_equal(window_maxima([0,2,2,0],window=1),[])
    np.testing.assert_array_equal(window_maxima([0,2,2,2,0],window=1),[2])
    np.testing.assert_array_equal(window_maxima([3,2,1],window=1),[])


def test_threshold_counter_and_offset():
    assert plateau_location([1,-1,1,-1,1],threshold=0,count=3)==1
    assert plateau_location([1,1,1],threshold=0,count=3)==-1
    assert plateau_location([],count=3)==0
    assert plateau_location([0,0,0],threshold=0,count=3)==0


def test_full_knee_negative_slice():
    x=np.random.default_rng(87).normal(size=4096)
    result=select_peaks(x)
    assert result['knee']==-4 and len(result['positions'])==result['candidate_count']-4
    assert np.all(np.diff(result['positions'])>0)
    np.testing.assert_array_equal(result['heights'],np.abs(result['residual'][result['positions']]))
    assert not np.shares_memory(x,result['residual'])


@pytest.mark.parametrize('x',[[1],[1,1,1],[0,1,0]])
def test_undefined_knee_rejected(x):
    with pytest.raises(ValueError,match='At least two'):select_peaks(x)


@pytest.mark.parametrize('window',[0,-1,True,1.5])
def test_invalid_window(window):
    with pytest.raises(ValueError):window_maxima([0,1,0],window=window)


def test_nonfinite_rejected():
    with pytest.raises(ValueError):window_maxima([0,float('nan'),0])
    with pytest.raises(ValueError):plateau_location([float('inf')])
