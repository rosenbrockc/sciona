"""Synthetic descriptor formulas, clipping and invalid-peak contracts."""
import numpy as np
import pytest
from sciona.vsb_descriptors import peak_descriptors,process_signal


def test_exact_sawtooth_and_neighbor_ratios():
    x=np.array([0.,0.,3.,1.,-1.,-3.,0.])
    f=peak_descriptors([2],x,small_radius=2,large_radius=4)
    np.testing.assert_allclose(f,[[1/3,0,2,0]],atol=1e-30)
    np.testing.assert_allclose(peak_descriptors([2],-7*x,small_radius=2,large_radius=4),f,atol=1e-30)


def test_clipped_offset_and_missing_neighbors():
    f=peak_descriptors([0],[3.,1.,-1.,-3.,0.])
    assert np.isnan(f[0,1]) and f[0,2]==-2 and f[0,3]<1e-30
    g=peak_descriptors([4],[0.,0.,0.,0.,3.])
    assert np.isnan(g[0,0]) and g[0,1]==0 and g[0,2]==-5 and g[0,3]==0


def test_independent_mse_oracle():
    x=[.2,-.3,2.,.8,-.2,-.7,.1]
    expected=sum((v/2-t)**2 for v,t in zip(x,[0,0,1,1/3,-1/3,-1,0]))/7
    assert peak_descriptors([2],x,large_radius=4)[0,3]==pytest.approx(expected,abs=1e-15)


def test_empty_and_order():
    x=[0.,3.,0.,0.,0.,0.,0.,0.,0.,0.,0.,4.,0.]
    assert peak_descriptors([],x).shape==(0,4)
    a=peak_descriptors([1,11],x,large_radius=2)
    np.testing.assert_array_equal(peak_descriptors([11,1],x,large_radius=2),a[::-1])


@pytest.mark.parametrize('positions',[[True],[1.5],[-1],[3],[[1]]])
def test_invalid_positions(positions):
    with pytest.raises(ValueError):peak_descriptors(positions,[0.,1.,0.])


def test_zero_and_unaligned_maximum():
    with pytest.raises(ValueError,match='Nonzero'):peak_descriptors([1],[0.,0.,0.])
    with pytest.raises(ValueError,match='align'):peak_descriptors([1],[0.,1.,2.])


def test_actual_joined_preprocessing():
    x=np.random.default_rng(91).normal(size=4096)
    result=process_signal(x)
    assert len(result['positions'])>0
    assert result['descriptors'].shape==(len(result['positions']),4)
    assert np.isfinite(result['descriptors']).all()
    assert np.all(result['heights']>0)
