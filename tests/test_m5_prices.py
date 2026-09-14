import numpy as np
import pandas as pd
import pytest
from sciona.m5_prices import features


def test_grouped_price_features_against_pandas():
    rng = np.random.default_rng(621)
    p = rng.integers(0, 8, 500).astype(float)
    p[::23] = np.nan
    a,b,m,y = rng.integers(0, 3, (4, 500))
    actual = features(p,a,b,m,y)
    f = pd.DataFrame(dict(p=p,a=a,b=b,m=m,y=y))
    g = f.groupby(['a','b']).p
    expected = dict(maximum=g.transform('max'),minimum=g.transform('min'),
                    std=g.transform('std'),mean=g.transform('mean'),
                    normalized=f.p/g.transform('max'),distinct_prices=g.transform('nunique'),
                    distinct_products=f.groupby(['a','p']).b.transform('nunique'),
                    previous_ratio=f.p/g.shift(1),
                    month_ratio=f.p/f.groupby(['a','b','m']).p.transform('mean'),
                    year_ratio=f.p/f.groupby(['a','b','y']).p.transform('mean'))
    assert list(actual) == list(expected)
    for name in actual:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-12, atol=1e-12, equal_nan=True)


def test_month_role_pools_across_years_and_previous_uses_row_order():
    out = features([2.,8.,6.], [0]*3, [0]*3, [1,2,1], [0,0,1])
    np.testing.assert_equal(out['month_ratio'], [.5,1,1.5])
    np.testing.assert_equal(out['previous_ratio'], [np.nan,4,.75])
    np.testing.assert_equal(out['normalized'], [.25,1,.75])


def test_missing_and_zero_price_semantics():
    out = features([0.,2.,np.nan,np.nan], [0]*4, [0,0,1,1], [0]*4, [0]*4)
    assert np.isinf(out['previous_ratio'][1])
    np.testing.assert_equal(out['distinct_prices'], [2,2,0,0])
    assert np.isnan(out['mean'][2:]).all()
    assert np.isnan(out['distinct_products'][2:]).all()


@pytest.mark.parametrize('p', [[], [True], [-1.], [np.inf]])
def test_invalid_prices_reject(p):
    with pytest.raises(ValueError):
        features(p,[0]*len(p),[0]*len(p),[0]*len(p),[0]*len(p))
