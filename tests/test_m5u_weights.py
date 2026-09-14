import numpy as np
import pandas as pd
import pytest
from sciona.m5u_weights import statistics,level_factors


def test_weight_stack_against_pandas_with_missing_gaps():
    rng=np.random.default_rng(31);x=rng.uniform(0,8,(80,5));revenue=x*rng.uniform(1,4,(80,5))
    x[:3,0]=np.nan;x[8:12,1]=np.nan;revenue[:5,3]=np.nan
    levels=np.array([12,12,12,3,3]);out=statistics(x,revenue,levels)
    h=pd.DataFrame(x);r=pd.DataFrame(revenue)
    np.testing.assert_equal(out['observations'],h.notna().expanding().sum().astype(np.int16))
    np.testing.assert_allclose(out['volatility'],h.diff().abs().expanding().mean(),rtol=1e-14,equal_nan=True)
    rolling=r.rolling(28,min_periods=1).sum();denominator=rolling.T.groupby(levels).sum().loc[levels].T.to_numpy()
    np.testing.assert_allclose(out['weights'],rolling/denominator,rtol=1e-14,equal_nan=True)


def test_missing_gaps_are_not_bridged_and_zero_revenue_is_undefined():
    out=statistics([[1.],[np.nan],[5.],[7.]],np.zeros((4,1)),[12])
    np.testing.assert_equal(out['volatility'][:,0],[np.nan,np.nan,np.nan,2.])
    assert np.isnan(out['weights']).all()
    np.testing.assert_equal(out['observations'][:,0],[1,1,2,3])


def test_level_factors_use_population_sizes():
    np.testing.assert_equal(level_factors([12,12,12,3,1]),[1,1,1,1/3,1/3])
    with pytest.raises(ValueError):level_factors([1,2])
