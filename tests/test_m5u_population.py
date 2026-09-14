import numpy as np
import pandas as pd
import pytest
from sciona.m5u_population import build


def test_population_alignment_categories_and_new_series_weights():
    n=200;x=np.tile(np.arange(n)[:,None]%7+1.,(1,2));x[:180,1]=np.nan
    roles=np.array([[0,0,0,0,0],[1,1,1,1,1]])
    out=build(x,x/2,x*3,roles,[12,12],np.repeat([0,1],100),np.full(n,2),99,reduced=True)
    assert out['features'].shape==(200,13)
    np.testing.assert_equal(out['days'][:4],[100,100,101,101])
    np.testing.assert_equal(out['series'][:4],[0,1,0,1])
    assert isinstance(out['features']['role_3'].dtype,pd.CategoricalDtype)
    assert (out['weights'][out['series']==1]==0).all()
    assert out['features'].iloc[1]['raw_ewm_15']==-10
    assert 'nonzero_28' not in out['scaled_columns']
    assert out['ewm_columns']==('raw_ewm_15','raw_ewm_30','raw_ewm_100')


def test_no_calendar_eligible_rows_rejects():
    with pytest.raises(ValueError,match='all history'):
        build(np.ones((30,1)),np.ones((30,1)),np.ones((30,1)),[[0]*5],[12],[0]*30,[2]*30,0)
