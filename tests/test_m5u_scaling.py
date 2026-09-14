import numpy as np
import pytest
from sciona.m5u_scaling import transform


def test_sentinel_selected_roles_and_training_filter():
    x=np.array([[4.,9.],[-10.,8.],[6.,7.],[8.,6.]])
    before=x.copy()
    out=transform(x,[8.,6.,np.nan,10.],[2.,3.,2.,2.],[1,1,1,2],[0],2,scale_range=0)
    np.testing.assert_equal(out['rows'],[0,1])
    np.testing.assert_equal(out['features'],[[2.,9.],[-10.,8.]])
    np.testing.assert_equal(out['targets'],[4.,2.])
    np.testing.assert_equal(x,before)


def test_random_scaling_scalar_oracle_and_oos_retention():
    n=40;seed=91;volume=np.arange(n)+1.;x=np.column_stack((np.arange(n)+3.,np.ones(n)))
    scales=volume*np.exp(.3*np.random.RandomState(seed).normal(0,.5,n))
    out=transform(x,np.full(n,-1.),volume,np.full(n,2),[0],2,out_of_sample=True,scale_range=.3,seed=seed)
    np.testing.assert_allclose(out['scales'],scales,rtol=1e-14)
    np.testing.assert_allclose(out['targets'],-1/scales,rtol=1e-14)
    np.testing.assert_allclose(out['features'][:,0],x[:,0]/scales,rtol=1e-14)
    np.testing.assert_equal(out['features'][:,1],x[:,1])
    np.testing.assert_equal(out['rows'],np.arange(n))


@pytest.mark.parametrize('volume',[0.,-1.,np.nan,np.inf])
def test_undefined_volume_rejects(volume):
    with pytest.raises(ValueError):transform([[1.]],[1.],[volume],[1],[0],2)


def test_invalid_scaled_role_rejects():
    with pytest.raises(ValueError):transform([[1.]],[1.],[1.],[1],[0,0],2)
