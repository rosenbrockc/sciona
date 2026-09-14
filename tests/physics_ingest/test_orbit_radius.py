import mpmath
import numpy as np
import pytest
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.orbit_radius import orbit_radius


def reference(g,m,t):
    ctx=mpmath.mp.clone();ctx.dps=1200
    G,M,T=(ctx.mpf(float(v)) for v in (g,m,t))
    # Independent log-domain radius and direct Kepler speed formula.
    radius=ctx.exp((ctx.log(G)+ctx.log(M)+2*ctx.log(T)-ctx.log(4*ctx.pi**2))/3)
    speed=ctx.exp((ctx.log(2*ctx.pi)+ctx.log(G)+ctx.log(M)-ctx.log(T))/3)
    return float(radius),float(speed)


@pytest.mark.parametrize('shape',[(),(20,),(2,3,4)])
def test_precision_and_preservation(shape):
    rng=np.random.default_rng(712)
    args=[np.asarray(np.exp(rng.uniform(-300,300,size=shape))) for _ in range(3)]
    copies=[a.copy() for a in args];result=orbit_radius(*args)
    for i,row in enumerate(zip(*(a.flat for a in args))):
        assert tuple(out.flat[i] for out in result)==reference(*row)
    assert all(out.shape==shape and out.dtype==np.float64 for out in result)
    assert all(np.array_equal(a,b) for a,b in zip(args,copies))


@pytest.mark.parametrize('args',[(1.,8.,2*np.pi),(1e308,1e308,1e-300),
    (1e-308,1e-308,1e308),(1.,1.,1.),(np.nextafter(0.,1.),1.,1.)])
def test_extremes(args):
    assert tuple(float(v) for v in orbit_radius(*args))==reference(*args)


def test_kepler_force_and_period_scaling():
    r,v=orbit_radius(1.,8.,2*np.pi)
    assert abs(float(v*v/r-8/r**2))<1e-14
    other=orbit_radius(1.,8.,16*np.pi)
    np.testing.assert_allclose(other,[4*r,v/2],rtol=3e-16)


@pytest.mark.parametrize('args',[(0,1,1),(1,-1,1),(1,1,0),(np.nan,1,1),
    (1,np.inf,1),(1,1,np.inf),(True,1,1),('1',1,1),(1j,1,1),([],[],[]),
    ([1,2],[1],[1]),(1e308,1e308,1e308),(1e308,1e308,1e-308),(np.nextafter(0.,1.),np.nextafter(0.,1.),np.nextafter(0.,1.))])
def test_invalid(args):
    with pytest.raises(ValueError):orbit_radius(*args)


def test_context_unchanged():
    dps=mpmath.mp.dps;orbit_radius(1,2,3)
    assert mpmath.mp.dps==dps
