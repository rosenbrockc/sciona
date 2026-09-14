from decimal import Decimal,localcontext
import numpy as np
import pytest
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.langmuir import langmuir


def reference(*row):
    with localcontext() as ctx:
        ctx.prec=2500
        ka,kd,p,N=(Decimal.from_float(float(v)) for v in row)
        # Solve the equilibrium linear system via determinants.
        determinant=ka*p+kd
        occupied=ka*p*N/determinant
        vacant=kd*N/determinant
        assert abs(ka*p*vacant-kd*occupied)<Decimal('1e-1500')
        return tuple(float(v) for v in (occupied/N,occupied,vacant,kd*occupied))


@pytest.mark.parametrize('shape',[(),(20,),(2,3,4)])
def test_precision_and_preservation(shape):
    rng=np.random.default_rng(745)
    args=[np.asarray(np.exp(rng.uniform(-100,100,size=shape))) for _ in range(4)]
    copies=[a.copy() for a in args];out=langmuir(*args)
    for i,row in enumerate(zip(*(a.flat for a in args))):assert tuple(a.flat[i] for a in out)==reference(*row)
    assert all(a.shape==shape and a.dtype==np.float64 for a in out)
    assert all(np.array_equal(a,b) for a,b in zip(args,copies))


@pytest.mark.parametrize('args',[(2.,3.,0.,10.),(2.,3.,1.5,10.),(1e300,1.,1.,1.),
    (1e308,1e308,1e308,1.),(1.,1.,1.,np.nextafter(0.,1.)*2)])
def test_special(args):
    assert tuple(float(v) for v in langmuir(*args))==reference(*args)


def test_saturation_does_not_erase_vacancy():
    theta,occupied,vacant,rate=langmuir(1e300,1,1,1)
    assert theta==occupied==1 and vacant>0 and rate==1
    assert 1-theta==0 and vacant==1e-300


def test_zero_pressure_and_half_coverage():
    assert tuple(float(v) for v in langmuir(2,3,0,10))==(0,0,10,0)
    assert tuple(float(v) for v in langmuir(2,3,1.5,10))==(.5,5,5,15)


@pytest.mark.parametrize('args',[(0,1,1,1),(1,0,1,1),(1,1,-1,1),(1,1,1,0),
    (np.inf,1,1,1),(1,np.nan,1,1),(1,1,np.inf,1),(1,1,1,np.nan),
    (True,1,1,1),('1',1,1,1),(1j,1,1,1),([],[],[],[]),([1,2],[1],[1],[1]),
    (1e-308,1e308,1e-308,1),(1,1e308,1e308,1e308)])
def test_invalid(args):
    with pytest.raises(ValueError):langmuir(*args)
