import numpy as np
import pandas as pd
import pytest
from sciona.m5u_rolling import statistic,blocks


@pytest.mark.parametrize('kind',['mean','median','std','skew','kurt','q10','q90'])
def test_history_statistic_against_pandas(kind):
    rng=np.random.default_rng(276);x=rng.normal(size=(200,3));x[10:13,1]=np.nan;x[:,2]=3
    for w in (7,14,28,168):
        rolling=pd.DataFrame(x).rolling(w)
        expected=rolling.quantile(.1 if kind=='q10' else .9) if kind.startswith('q') else getattr(rolling,kind)()
        np.testing.assert_allclose(statistic(x,w,kind),expected,rtol=1e-10,atol=1e-10,equal_nan=True)


def test_full_and_reduced_inventory_and_zero_filled_lags():
    x=np.ones((40,2));x[0,0]=np.nan
    full=blocks(x,x/2);small=blocks(x,x/2,reduced=True)
    assert len(full)==87 and len(small)==9
    np.testing.assert_equal(full['lag_1'][:2],[[0,0],[0,1]])
    assert full['nonzero_7'][6,0]==pytest.approx(6/7)
    assert not any(k.startswith('scaled') for k in small)
