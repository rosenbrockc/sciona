from decimal import Decimal,localcontext
import numpy as np
import pytest
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.lorentz_boost import lorentz_boost,witness_lorentz_boost
from sciona.ghost.abstract import AbstractArray


def reference(c,v,t,x,y,z):
    # Independent null-coordinate Doppler scaling rather than direct boost.
    with localcontext() as ctx:
        ctx.prec=2500
        c,v,t,x,y,z=(Decimal.from_float(float(a)) for a in (c,v,t,x,y,z))
        forward=((c-v)/(c+v)).sqrt();backward=1/forward
        plus=forward*(c*t+x);minus=backward*(c*t-x)
        return tuple(float(a) for a in ((forward+backward)/2,(plus+minus)/(2*c),(plus-minus)/2,y,z))


@pytest.mark.parametrize('row',[(1.,0.,-2.,3.,-4.,5.),(1.,.5,1.,1.,2.,3.),
    (1.,-.5,1.,-1.,2.,3.),(1.,np.nextafter(1.,0.),1.,1.,0.,0.),
    (1e308,5e307,2.,1e308,1.,-1.),(1e-308,0.,1.,np.nextafter(0.,1.),0.,0.),
    (1.,.5,.5,1.,0.,0.),(1.,.5,1e308,-1e308,0.,0.),
    (1.,.5,np.nextafter(0.,1.),np.nextafter(0.,1.),0.,0.)])
def test_extremes_and_cancellation(row):
    actual=lorentz_boost(*(np.array(a) for a in row))
    np.testing.assert_array_equal([a.item() for a in actual],reference(*row))


def test_arrays_and_identity_copies():
    rng=np.random.default_rng(312)
    c=np.exp(rng.uniform(-80,80,(2,3,4)))
    args=[c,c*rng.uniform(-.99,.99,c.shape)]+[rng.normal(size=c.shape) for _ in range(4)]
    before=[a.copy() for a in args];values=lorentz_boost(*args)
    rows=[reference(*r) for r in zip(*(a.flat for a in args))]
    for i,a in enumerate(values):
        np.testing.assert_array_equal(a,np.array([r[i] for r in rows]).reshape(c.shape))
        assert a.dtype==np.float64 and all(not np.shares_memory(a,b) for b in args)
    for a,b in zip(args,before):np.testing.assert_array_equal(a,b)
    args[1]=np.zeros(c.shape)
    identity=lorentz_boost(*args)
    np.testing.assert_array_equal(identity[0],np.ones(c.shape))
    for a,b in zip(identity[1:],args[2:]):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('c,v',[(0.,0.),(-1.,0.),(1.,1.),(1.,-1.),(1.,2.),(np.inf,0.),(1.,np.nan)])
def test_domain(c,v):
    with pytest.raises(ValueError):lorentz_boost(*(np.array(a) for a in [c,v,1.,1.,0.,0.]))


@pytest.mark.parametrize('row',[(1.,.75,1e308,-1e308,0.,0.),
    (1.,.75,np.nextafter(0.,1.),np.nextafter(0.,1.),0.,0.)])
def test_output_range(row):
    with pytest.raises(ValueError):lorentz_boost(*(np.array(a) for a in row))


@pytest.mark.parametrize('bad',[np.array([]),np.array([True]),np.array([1j]),np.array(['1']),np.array([1],dtype=object)])
def test_input_types(bad):
    with pytest.raises(ValueError):lorentz_boost(bad,*[np.zeros(bad.shape) for _ in range(5)])


def test_shape_and_witness():
    with pytest.raises(ValueError):lorentz_boost(np.ones(2),np.array(0.),*[np.zeros(2) for _ in range(4)])
    a=AbstractArray(shape=(2,3),dtype='float64')
    assert len(witness_lorentz_boost(*[a]*6))==5
    with pytest.raises(ValueError):witness_lorentz_boost(*[a]*5,AbstractArray(shape=(1,),dtype='float64'))
