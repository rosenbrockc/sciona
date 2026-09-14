from decimal import Decimal,localcontext
import numpy as np
import pytest
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.parallel_resistance import parallel_resistance


def reference(a,b,v):
    with localcontext() as ctx:
        ctx.prec=2500
        r1,r2,voltage=map(lambda x:Decimal.from_float(float(x)),[a,b,v])
        g1,g2=1/r1,1/r2
        return tuple(float(x) for x in [1/(g1+g2),voltage*g1,voltage*g2,voltage*(g1+g2)])


@pytest.mark.parametrize('shape',[(),(20,),(2,3,4)])
def test_reference_and_inputs(shape):
    rng=np.random.default_rng(420)
    args=[np.asarray(np.exp(rng.uniform(-30,30,size=shape))) for _ in range(2)]+[np.asarray(rng.normal(size=shape))]
    copies=[x.copy() for x in args];out=parallel_resistance(*args)
    for i,row in enumerate(zip(*(x.flat for x in args))):
        assert all(o.flat[i]==e for o,e in zip(out,reference(*row)))
    assert all(o.shape==shape and o.dtype==np.float64 for o in out)
    assert all(np.array_equal(a,b) for a,b in zip(args,copies))


@pytest.mark.parametrize('v',[0.,12.,-12.])
def test_circuit(v):
    out=parallel_resistance(np.array(3.),np.array(6.),np.array(v))
    assert tuple(float(o) for o in out)==(2.,v/3,v/6,v/2)


@pytest.mark.parametrize('a,b,v',[(1e308,1e308,1.),(1e-308,1e-308,1e-308),(1.,1e308,0.)])
def test_extremes(a,b,v):
    assert tuple(float(x) for x in parallel_resistance(a,b,v))==reference(a,b,v)


def test_exchange_symmetry():
    out=parallel_resistance(3.,7.,-4.);rev=parallel_resistance(7.,3.,-4.)
    assert all(a==b for a,b in zip(out,[rev[0],rev[2],rev[1],rev[3]]))


@pytest.mark.parametrize('a,b,v',[
    (0,1,1),(-1,1,1),(1,0,1),(1,-1,1),(np.inf,1,1),(1,1,np.nan),
    ([],[],[]),([1,2],[1],[1]),([True],[1],[1]),([1+1j],[1],[1]),
    (['1'],[1],[1]),(np.nextafter(0.,1.),np.nextafter(0.,1.),0),
    (1e-308,1e-308,1e308),(1e308,1e308,np.nextafter(0.,1.)),
])
def test_invalid(a,b,v):
    with pytest.raises(ValueError):parallel_resistance(a,b,v)
