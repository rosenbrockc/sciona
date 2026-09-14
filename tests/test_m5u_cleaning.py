import numpy as np
import pytest
from sciona.m5u_cleaning import retained_rows,missing_sentinel


def test_full_calendar_earliest_year_and_month_exclusion():
    years=[0,0,1,1,1,1,1,2];months=[2,7,1,2,9,10,12,3]
    np.testing.assert_equal(retained_rows([7,3,4,2,5,0,3],years,months,3),[0,1,2,6])


def test_pivot_missing_boundary_preserves_earlier_eligible_rows():
    years=[0,1,1,1,1];months=[2]*5
    np.testing.assert_equal(retained_rows([1,2,4],years,months,3,stacked=False),[0,1,2])
    np.testing.assert_equal(retained_rows([1,2,4],years,months,3,stacked=True),[2])


def test_pivot_boundary_is_positional_slice():
    np.testing.assert_equal(retained_rows([4,3,2],[0,1,1,1,1],[2]*5,3,stacked=False),[1,2])


def test_sentinel_and_infinity_are_distinct():
    x=np.array([np.nan,-10.,np.inf,-np.inf,2.]);result=missing_sentinel(x)
    np.testing.assert_equal(result,[-10.,-10.,np.inf,-np.inf,2.])
    assert np.isnan(x[0])


def test_invalid_day_and_duplicate_pivot_reject():
    with pytest.raises(ValueError):retained_rows([-1],[1],[2],0)
    with pytest.raises(ValueError):retained_rows([0,0],[1],[2],0,stacked=False)
