import mpmath
import numpy as np
import pytest
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.brewster_angle import brewster_angle


def reference(n1,n2):
    ctx=mpmath.mp.clone();ctx.dps=1000
    a,b=ctx.mpf(float(n1)),ctx.mpf(float(n2))
    return float(ctx.atan(b/a)),float(ctx.atan(a/b))


@pytest.mark.parametrize('shape',[(),(20,),(2,3,4)])
def test_precision_and_preservation(shape):
    rng=np.random.default_rng(413);args=[np.asarray(np.exp(rng.uniform(-300,300,size=shape))) for _ in range(2)]
    copies=[a.copy() for a in args];out=brewster_angle(*args)
    for i,row in enumerate(zip(*(a.flat for a in args))):assert all(v.flat[i]==w for v,w in zip(out,reference(*row)))
    assert all(a.shape==shape and a.dtype==np.float64 for a in out)
    assert all(np.array_equal(a,b) for a,b in zip(args,copies))


@pytest.mark.parametrize('a,b',[(1.,1.5),(1.5,1.),(1.,1.),(1e308,1e308),(np.nextafter(0.,1.),1.)])
def test_special(a,b):
    assert tuple(float(x) for x in brewster_angle(a,b))==reference(a,b)


def test_source_ratio_and_complement():
    i,t=brewster_angle(1.,1.5)
    assert float(i)>np.pi/4>float(t)
    assert abs(float(i+t)-np.pi/2)<1e-15


def test_independent_tiny_complement():
    i,t=brewster_angle(1e-300,1.)
    assert i==np.pi/2 and t>0
    assert np.pi/2-i==0 and t!=0


@pytest.mark.parametrize('n1,n2',[(1.,1.5),(1.5,1.),(2.,3.)])
def test_snell_fresnel_moderate_indices(n1,n2):
    i,t=brewster_angle(n1,n2)
    assert abs(n1*np.sin(i)-n2*np.sin(t))<1e-14
    assert abs(n2*np.cos(i)-n1*np.cos(t))<1e-14


@pytest.mark.parametrize('a,b',[(0,1),(-1,1),(1,0),(1,-1),(np.nan,1),(1,np.inf),([],[]),([1,2],[1]),([True],[1]),([1j],[1]),(['1'],[1]),(np.nextafter(0.,1.),1e308)])
def test_invalid(a,b):
    with pytest.raises(ValueError):brewster_angle(a,b)


def test_context_unchanged():
    dps=mpmath.mp.dps
    brewster_angle(1.,2.)
    assert mpmath.mp.dps==dps
