from decimal import Decimal,localcontext
import numpy as np
import pytest
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.constant_acceleration import constant_acceleration,witness_constant_acceleration
from sciona.ghost.abstract import AbstractArray


def reference(u,a,t):
    with localcontext() as ctx:
        ctx.prec=2500
        u,a,t=(Decimal.from_float(float(x)) for x in (u,a,t))
        final=u+a*t
        mean=(u+final)/2
        return tuple(float(x) for x in (final,mean*t,mean))


@pytest.mark.parametrize('row',[(1.,-1.,2.),(-3.,0.,2.),(2.,3.,0.),
    (1e308,-1e308,2.),(0.,1e308,1e-308),(np.nextafter(0.,1.),0.,1.),
    (1.,-2.,1.),(0.,0.,0.)])
def test_boundaries_and_cancellation(row):
    values=constant_acceleration(*(np.array(v) for v in row))
    np.testing.assert_array_equal([v.item() for v in values],reference(*row))


def test_array_reference_shape_copy():
    rng=np.random.default_rng(447)
    args=[rng.normal(size=(2,3,4))*1e80,rng.normal(size=(2,3,4))*1e80,np.exp(rng.uniform(-80,80,(2,3,4)))]
    copies=[v.copy() for v in args];actual=constant_acceleration(*args)
    rows=[reference(*row) for row in zip(*(a.flat for a in args))]
    for i,arr in enumerate(actual):
        np.testing.assert_array_equal(arr,np.array([r[i] for r in rows]).reshape(args[0].shape))
        assert arr.dtype==np.float64 and not any(np.shares_memory(arr,a) for a in args)
    for a,b in zip(args,copies):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('row',[(1.,1.,-1.),(np.inf,0.,1.),(0.,np.nan,1.),(0.,1.,np.inf),
    (1e308,1e308,1.),(1e308,0.,2.),(0.,np.nextafter(0.,1.),1.)])
def test_domain_and_output_range(row):
    with pytest.raises(ValueError):constant_acceleration(*(np.array(v) for v in row))


@pytest.mark.parametrize('bad',[np.array([]),np.array([True]),np.array([1j]),np.array(['1']),np.array([1],dtype=object)])
def test_nonreal_or_empty(bad):
    with pytest.raises(ValueError):constant_acceleration(bad,np.zeros(bad.shape),np.ones(bad.shape))


def test_no_broadcasting_and_witness():
    with pytest.raises(ValueError):constant_acceleration(np.ones(2),np.array(0.),np.ones(2))
    a=AbstractArray(shape=(2,3),dtype='float64')
    assert all(v.shape==(2,3) for v in witness_constant_acceleration(a,a,a))
    with pytest.raises(ValueError):witness_constant_acceleration(a,a,AbstractArray(shape=(1,),dtype='float64'))
