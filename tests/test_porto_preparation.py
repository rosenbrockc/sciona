import numpy as np
import pytest
from sciona.porto_preparation import prepare_populations


def test_joint_categories_tree_raw_and_dae_binary_passthrough():
    x=np.array([[99.,-1.,0.,2.],[98.,3.,1.,3.]])
    q=np.array([[97.,8.,0.,4.]])
    r=prepare_populations(x,q,dropped_columns=[0],categorical_columns=[3],binary_columns=[2])
    expected=np.array([[-1.,0.,1.,0.,0.],[3.,1.,0.,1.,0.],[8.,0.,0.,0.,1.]])
    np.testing.assert_array_equal(np.vstack((r['tree_training'],r['tree_query'])),expected)
    normalized=np.vstack((r['dae_training'],r['dae_query']))
    np.testing.assert_array_equal(normalized[:,1:],expected[:,1:])
    assert np.isfinite(normalized).all() and abs(normalized[:,0].mean())<1e-15
    assert r['prepared_width']==5 and r['binary_output_columns']==(1,2,3,4)
    x[:,0]=10000;q[:,0]=-20000
    other=prepare_populations(x,q,dropped_columns=[0],categorical_columns=[3],binary_columns=[2])
    for key in ('tree_training','tree_query','dae_training','dae_query'):np.testing.assert_array_equal(r[key],other[key])


@pytest.mark.parametrize('roles',[dict(dropped_columns=[0],categorical_columns=[0],binary_columns=[]),dict(dropped_columns=[0,1],categorical_columns=[],binary_columns=[]),dict(dropped_columns=[],categorical_columns=[2],binary_columns=[])])
def test_invalid_roles(roles):
    with pytest.raises(ValueError):prepare_populations([[0.,1.]],[[1.,0.]],**roles)
