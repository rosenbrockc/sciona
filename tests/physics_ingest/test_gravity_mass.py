from decimal import Decimal,localcontext
import numpy as np
import pytest
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.gravity_mass import gravity_mass


def reference(g,r,G):
    with localcontext() as ctx:
        ctx.prec=2500
        a,b,c=map(lambda x:Decimal.from_float(float(x)),[g,r,G])
        return float((a/c)*b*b)


@pytest.mark.parametrize('shape',[(),(20,),(2,3,4)])
def test_oracle(shape):
    rng=np.random.default_rng(394)
    args=[np.asarray(np.exp(rng.uniform(-100,100,size=shape))) for _ in range(3)]
    copies=[a.copy() for a in args];out=gravity_mass(*args)
    assert out.shape==shape and out.dtype==np.float64
    for got,row in zip(out.flat,zip(*(a.flat for a in args))):assert got==reference(*row)
    assert all(np.array_equal(a,b) for a,b in zip(args,copies))


def test_source_endpoint_discrepancy():
    args=(9.80665,6378100.,6.67430e-11)
    actual=float(gravity_mass(*args))
    assert actual==reference(*args)
    assert 5.9771e24<actual<5.9773e24
    assert actual!=5.972e24


@pytest.mark.parametrize('args',[(1e-300,1e300,1e300),(1e300,1e-300,1e-300),(np.nextafter(0.,1.),1.,1.)])
def test_no_premature_overflow_underflow(args):
    assert float(gravity_mass(*args))==reference(*args)


def test_known_inverse_and_scaling():
    assert gravity_mass(2.,3.,6.)==3.
    assert gravity_mass(2.,6.,6.)==12.


@pytest.mark.parametrize('args',[(0,1,1),(-1,1,1),(1,0,1),(1,1,0),(1,1,-1),(np.nan,1,1),(1,np.inf,1),([],[],[]),([1,2],[1],[1]),([True],[1],[1]),([1+1j],[1],[1]),(['1'],[1],[1]),(1e308,1e308,1),(1e-308,1e-308,1)])
def test_invalid(args):
    with pytest.raises(ValueError):gravity_mass(*args)
