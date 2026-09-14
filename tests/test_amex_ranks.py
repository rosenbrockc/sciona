"""Synthetic grouped ranking and average-tie time window checks."""
import numpy as np
import pandas as pd
import pytest
from sciona.amex_ranks import percentile_ranks,latest_window


def test_percentile_ties_nan_and_order_against_group_oracle():
    x=np.array([[4.,np.nan],[1.,4.],[4.,2.],[7.,4.],[np.nan,2.]])
    groups=np.array([1,0,1,0,1])
    expected=pd.DataFrame(x).groupby(groups).rank(pct=True).to_numpy()
    np.testing.assert_allclose(percentile_ranks(x,groups),expected,equal_nan=True)
    assert percentile_ranks(x,groups)[0,0]==.75


def test_tied_window_is_not_tail_and_preserves_order():
    t=np.array([2.,5.,5.,1.,np.nan]);g=np.zeros(5,int)
    np.testing.assert_array_equal(latest_window(t,g,count=1),[False]*5)
    np.testing.assert_array_equal(latest_window(t,g,count=2),[False,True,True,False,False])


def test_random_window_against_pandas():
    rng=np.random.default_rng(104);t=rng.integers(0,8,100).astype(float);t[0]=np.nan;g=rng.integers(0,5,100)
    for n in (3,6):
        expected=(pd.Series(t).groupby(g).rank(ascending=False)<=n).to_numpy()
        np.testing.assert_array_equal(latest_window(t,g,count=n),expected)


def test_invalid_groups_and_count():
    with pytest.raises(ValueError):percentile_ranks([[1.]],['a'])
    with pytest.raises(ValueError):latest_window([1.],[0],count=True)
