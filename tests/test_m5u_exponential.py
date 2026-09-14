import numpy as np
import pandas as pd
import pytest
from sciona.m5u_exponential import mean,histories,crosses


@pytest.mark.parametrize('adjust',[False,True])
@pytest.mark.parametrize('span',[1,3,15,100])
def test_exponential_mean_against_pandas_with_missing_gaps(adjust,span):
    rng=np.random.default_rng(415);x=rng.normal(size=(180,4))
    x[:5,0]=np.nan;x[10:25,1]=np.nan;x[::3,2]=np.nan;x[:,3]=np.nan
    minimum=int(np.ceil(span**.8))
    expected=pd.DataFrame(x).ewm(span=span,min_periods=minimum,adjust=adjust).mean()
    np.testing.assert_allclose(mean(x,span,minimum=minimum,adjust=adjust),expected,rtol=1e-13,atol=1e-13,equal_nan=True)


def test_history_inventory_and_cross_order():
    assert list(histories(np.ones((10,1))))==[3,7,15,30,100]
    assert list(histories(np.ones((10,1)),reduced=True))==[15,30,100]
    x=np.array([[2.,4.,1.],[0.,0.,2.]])
    values,pairs=crosses(x)
    assert pairs==[(0,1),(0,2),(1,2)]
    np.testing.assert_equal(values[0],[-2.,.5,1.,2.,3.,4.])
    assert np.isnan(values[1,1])


def test_missing_and_constant_windows():
    np.testing.assert_equal(mean([[1.],[np.nan],[1.]],3),[[1.],[1.],[1.]])
    with pytest.raises(ValueError):mean([[np.inf]],3)
