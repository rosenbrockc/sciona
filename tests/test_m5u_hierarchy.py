import numpy as np
import pandas as pd
import pytest
from sciona.m5u_hierarchy import aggregate,normalize_by_outlet,GROUPS


def roles():return np.array([[o,o,p%2,p,p] for o in range(2) for p in range(3)])


def test_all_levels_against_pandas_sums_and_source_order():
    r=roles();values=np.arange(24,dtype=float).reshape(4,6);values[0,:]=np.nan
    result=aggregate(values,r)
    np.testing.assert_equal(result['values'][:,:6],values)
    assert list(dict.fromkeys(result['levels']))==list(range(12,0,-1))
    for level,group in GROUPS:
        expected=pd.DataFrame(values.T).groupby([r[:,i] for i in group] if group else np.zeros(len(r))).sum().to_numpy().T
        np.testing.assert_equal(result['values'][:,result['levels']==level],expected)
    np.testing.assert_equal(result['roles'][result['levels']==1],np.full((1,5),-1))
    item=result['roles'][result['levels']==10]
    assert (item[:,:2]==-1).all() and (item[:,2:]>=0).all()


def test_base_only_normalization_missing_and_zero_denominators():
    v=np.array([[2.,4.,100.],[np.nan,np.nan,5.],[0.,0.,1.]])
    out=normalize_by_outlet(v,[0,0,0],[True,True,False])
    np.testing.assert_allclose(out[0],[2/3,4/3,100/3])
    np.testing.assert_equal(out[1],[np.nan,np.nan,5.])
    assert np.isinf(out[2,2])


def test_inconsistent_hierarchy_and_missing_parent_reject():
    r=roles();r[3,3]=9
    with pytest.raises(ValueError,match='inconsistent'):aggregate(np.ones((2,6)),r)
    with pytest.raises(ValueError,match='denominator'):normalize_by_outlet([[1.,2.]],[0,-1],[True,False])
