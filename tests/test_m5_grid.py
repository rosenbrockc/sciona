import numpy as np
import pandas as pd
import pytest
from sciona.m5_grid import build


def test_day_major_order_future_missing_and_release_filter():
    days=np.arange(31);weeks=days//3
    result=build([[0.,2.,3.],[9.,8.,7.]],0,[11,22],days,weeks,[11,22],[0,1])
    np.testing.assert_equal(result['series'][:5],[0,0,0,0,1])
    np.testing.assert_equal(result['day'][:5],[0,1,2,3,3])
    np.testing.assert_equal(result['target'][:5],[0.,2.,3.,np.nan,np.nan])
    assert np.isnan(result['target'][3:]).all()
    np.testing.assert_equal(result['release'][:5],[0,0,0,0,1])
    assert len(result['day'])==59


def test_join_filter_matches_independent_pandas_construction():
    rng=np.random.default_rng(293)
    history=rng.integers(0,10,(4,20)).astype(float)
    days=np.arange(100,148);weeks=(days-100)//7+50
    pair=[8,3,9,2];price_pair=[8,8,3,9];price_week=[50,53,51,49]
    out=build(history,100,pair,days,weeks,price_pair,price_week)
    wide=pd.DataFrame(history,columns=days[:20]);wide['series']=np.arange(4)
    expected=wide.melt(id_vars='series',var_name='day',value_name='target')
    future=pd.DataFrame({'series':np.tile(np.arange(4),28),'day':np.repeat(days[20:],4),'target':np.nan})
    expected=pd.concat([expected,future],ignore_index=True)
    expected['pair']=np.asarray(pair)[expected.series]
    releases=pd.DataFrame({'pair':price_pair,'release':price_week}).groupby('pair',as_index=False).min()
    expected=expected.merge(releases,on='pair',how='left').merge(pd.DataFrame({'day':days,'week':weeks}),on='day',how='left')
    expected=expected[expected.week>=expected.release]
    expected.release=(expected.release-expected.release.min()).astype(np.int16)
    for name in out:
        np.testing.assert_equal(out[name],expected[name].to_numpy())


def test_offset_minimum_uses_surviving_rows_only():
    out=build([[1.]],0,[9],np.arange(29),np.zeros(29,dtype=int)+50,[9,99],[50,1])
    np.testing.assert_equal(out['release'],np.zeros(29,dtype=np.int16))


def test_ambiguous_or_missing_calendar_rejects():
    for days in [np.arange(28),np.concatenate((np.arange(29),[0]))]:
        with pytest.raises(ValueError):
            build([[1.]],0,[9],days,days,[9],[0])


def test_no_released_rows_rejects():
    with pytest.raises(ValueError,match='entire grid'):
        build([[1.]],0,[9],np.arange(29),np.arange(29),[9],[50])
