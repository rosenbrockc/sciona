from datetime import date,timedelta
import numpy as np
import pandas as pd
import pytest
from sciona.m5_preprocessing import prepare


def fixture():
    n=250;days=np.arange(n+28);weeks=days//7
    calendar=dict(day=days,week=weeks,date=[(date(2000,1,1)+timedelta(days=int(d))).isoformat() for d in days],
        categories=[['z' if d%2 else None,'a',None,None,0,1,0] for d in days])
    price_weeks=np.unique(weeks)
    prices=dict(pair=np.tile([7,8],len(price_weeks)),outlet=np.zeros(len(price_weeks)*2,dtype=int),
        product=np.tile([0,1],len(price_weeks)),week=np.repeat(price_weeks,2),value=np.tile([2.,4.],len(price_weeks)))
    groupings=[[0],[1],[2],[3],[0,2],[0,3],[1,2],[1,3],[4],[4,0],[4,1]]
    return dict(history=np.vstack((np.arange(n)%11,np.arange(n)%5)).astype(float),first_day=0,pairs=[7,8],
        roles=[[0,0,0,0,0],[0,0,1,1,1]],calendar=calendar,prices=prices,groupings=groupings)


def test_all_blocks_align_with_grid_and_future_is_missing():
    args=fixture();out=prepare(**args);grid=out['grid'];n=556
    assert len(grid['day'])==n and out['encoding'].shape==(n,22)
    for block in ('categorical','prices','calendar','lags'):
        assert all(len(v)==n for v in out[block].values())
    np.testing.assert_equal(grid['target'][:6],[0,0,1,1,2,2])
    assert np.isnan(grid['target'][-56:]).all()
    assert out['encoding_cutoff']==221
    np.testing.assert_equal(out['prices']['value'][:6],[2,4,2,4,2,4])
    np.testing.assert_equal(out['categorical']['calendar_0'].codes[:6],[-1,-1,0,0,-1,-1])
    np.testing.assert_equal(out['lags']['lag_28'][56:60],[0,0,1,1])
    eligible=args['history'][:,:222].ravel()
    assert out['encoding'][0,0]==np.float16(eligible.mean())
    assert grid['target'].dtype==np.float16


def test_holdout_targets_do_not_change_encoding_but_do_change_lags():
    args=fixture();before=prepare(**args)
    args['history'][:,-28:]=100
    after=prepare(**args)
    np.testing.assert_equal(before['encoding'],after['encoding'])
    assert not np.array_equal(before['lags']['lag_28'][-56:],after['lags']['lag_28'][-56:],equal_nan=True)


def test_category_vocabulary_matches_joined_pandas():
    args=fixture();out=prepare(**args)
    selected=np.asarray(args['calendar']['categories'],dtype=object)[np.repeat(np.arange(278),2)]
    for i in range(7):
        expected=pd.Categorical(selected[:,i])
        np.testing.assert_equal(out['categorical'][f'calendar_{i}'].codes,expected.codes)


def test_wrong_category_width_rejects():
    args=fixture();args['calendar']['categories']=[[0]]*278
    with pytest.raises(ValueError,match='seven'):
        prepare(**args)
