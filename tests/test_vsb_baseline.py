"""Synthetic mathematical oracles for independent recursive baseline removal."""
import numpy as np
import pytest
from sciona.vsb_baseline import subtract_baseline


def test_scalar_recurrence():
    x=np.random.default_rng(81).normal(size=500)
    baseline=x[0];expected=[0.]
    for value in x[1:]:
        baseline=.99*baseline+.01*value
        expected.append(value-baseline)
    np.testing.assert_allclose(subtract_baseline(x),expected,atol=2e-14,rtol=1e-13)


def test_impulse_decay_and_step():
    impulse=np.zeros(50);impulse[1]=1
    expected=np.r_[0.,.99,-.01*.99**np.arange(1,49)]
    np.testing.assert_allclose(subtract_baseline(impulse),expected,atol=2e-16)
    step=np.r_[0.,np.ones(20)]
    np.testing.assert_allclose(subtract_baseline(step),np.r_[0.,.99**np.arange(1,21)],atol=1e-15)


def test_offset_scale_and_no_input_mutation():
    x=np.array([0.,2.,-1.,4.,3.]);saved=x.copy()
    np.testing.assert_allclose(subtract_baseline(3*x+20),3*subtract_baseline(x),atol=1e-14)
    np.testing.assert_array_equal(x,saved)
    assert not np.shares_memory(x,subtract_baseline(x))


def test_singleton_constant_endpoints_and_integer_input():
    np.testing.assert_array_equal(subtract_baseline([5]),[0.])
    np.testing.assert_allclose(subtract_baseline([7]*8),0.,atol=1e-14)
    np.testing.assert_array_equal(subtract_baseline([1,4,2],retention=0),[0.,0.,0.])
    np.testing.assert_array_equal(subtract_baseline([1,4,2],retention=1),[0.,3.,1.])
    assert subtract_baseline([0,1,0]).dtype==np.float64


@pytest.mark.parametrize('value',[[],[[1,2]],[True,False],['1','2'],[0,float('nan')],[float('inf')]])
def test_invalid_signals(value):
    with pytest.raises(ValueError):subtract_baseline(value)


@pytest.mark.parametrize('retention',[True,-.01,1.01,float('nan'),'0.99'])
def test_invalid_retention(retention):
    with pytest.raises(ValueError):subtract_baseline([0.,1.],retention=retention)
