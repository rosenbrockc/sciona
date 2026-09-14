"""Synthetic joint manual-block routing and rank-window regression checks."""
import numpy as np
import pytest
from sciona.amex_manual import build


def fixture():
    return ([[[1.],[3.],[np.nan],[7.]],[[2.],[4.],[6.],[8.]]],
        [[[0.],[1.],[0.],[2.]],[[2.],[1.],[1.],[0.]]],
        [[1.,4.,3.,2.],[1.,2.,3.,4.]],[[0,0,1,1],[0,0,1,1]])


def test_all_blocks_zero_fill_and_source_order():
    args=fixture();result=build(*args,zero_fill_columns=[0])
    assert set(result)=={'full_categorical','full_numeric','full_difference','customer_rank','month_rank','last3_numeric','last3_difference','last3_categorical','last6_numeric'}
    assert result['full_numeric'].shape==(2,6)
    assert result['last3_numeric'][0,0]==(10/3)//.01
    assert result['last3_difference'][0,4]==4.//.01
    assert result['customer_rank'][0,0]==1.
    assert result['month_rank'][0,-1]==.75//.01
    assert np.isnan(args[0][0][2][0])


def test_timestamp_ties_can_remove_entire_window_group():
    numeric=[np.arange(7.)[:,None],np.ones((2,1))];cats=[np.zeros((7,1)),np.zeros((2,1))]
    result=build(numeric,cats,[np.ones(7),[1,2]],[np.zeros(7,int),[0,0]],zero_fill_columns=[])
    assert np.isnan(result['last3_numeric'][0]).all()
    assert np.isnan(result['last3_categorical'][0]).all()
    assert np.isfinite(result['last6_numeric'][0]).all()


def test_joint_month_ranks_change_with_query_population():
    args=fixture();a=build(*args,zero_fill_columns=[0]);args[0][1]=[[100.]]*4
    b=build(*args,zero_fill_columns=[0])
    np.testing.assert_array_equal(a['customer_rank'][0],b['customer_rank'][0])
    assert not np.array_equal(a['month_rank'][0],b['month_rank'][0])


def test_invalid_fill_role():
    with pytest.raises(ValueError):build(*fixture(),zero_fill_columns=[True])
