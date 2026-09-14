import numpy as np
import pytest
from sciona.m5_join import first_per_key,left_indices,take_numeric
from sciona.m5_price_block import build


def population():
    return dict(pair=[7,8,7],outlet=[0,0,0],product=[0,1,0],week=[10,10,11],value=[2.,4.,6.])


def calendar():
    return dict(week=[10,10,11],month=[1,2,2],year=[0,0,0])


def test_connected_price_block_and_first_calendar_week_row():
    result=build([7,7,8,7],[11,10,10,12],population(),calendar())
    np.testing.assert_equal(result['value'],[6,2,4,np.nan])
    np.testing.assert_equal(result['month_ratio'],[1,1,1,np.nan])
    np.testing.assert_equal(result['year_ratio'],[1.5,.5,1,np.nan])
    np.testing.assert_equal(result['previous_ratio'],[3,np.nan,np.nan,np.nan])
    np.testing.assert_equal(result['distinct_prices'],[2,2,1,np.nan])
    assert result['value'].dtype==np.float16
    assert result['distinct_prices'].dtype==np.float64


def test_gather_dtypes_and_order():
    indices=left_indices([[2],[1],[2],[9]],[[1],[2]])
    np.testing.assert_equal(indices,[1,0,1,-1])
    np.testing.assert_equal(take_numeric(np.array([4,5],dtype=np.int8),indices),[5,4,5,np.nan])
    assert take_numeric(np.array([4,5],dtype=np.int8),[1,0]).dtype==np.int8
    np.testing.assert_equal(first_per_key([[2],[1],[2],[1]]),[0,1])


def test_duplicate_price_week_rejects():
    p=population();p['week'][-1]=10
    with pytest.raises(ValueError,match='Duplicate'):
        build([7],[10],p,calendar())


def test_inconsistent_pair_mapping_rejects():
    p=population();p['pair'][-1]=99
    with pytest.raises(ValueError,match='pair coding'):
        build([7],[10],p,calendar())


def test_missing_calendar_rejects():
    c=calendar();c['week'][-1]=12
    with pytest.raises(ValueError,match='calendar'):
        build([7],[10],population(),c)
