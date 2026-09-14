import concurrent.futures
import mpmath
import numpy as np
import pytest
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.circular_orbit_speed import circular_orbit_speed


def oracle(r,t):
    ctx=mpmath.mp.clone();ctx.dps=1000
    circumference=ctx.mpf(float(r))*ctx.pi*2
    return float(circumference),float(circumference/ctx.mpf(float(t)))


@pytest.mark.parametrize('shape',[(),(20,),(2,3,4)])
def test_independent_precision(shape):
    rng=np.random.default_rng(9023)
    r=np.asarray(np.exp(rng.uniform(-30,30,size=shape)))
    t=np.asarray(np.exp(rng.uniform(-30,30,size=shape)))
    original=(r.copy(),t.copy());result=circular_orbit_speed(r,t)
    for i,(a,b) in enumerate(zip(r.flat,t.flat)):
        expected=oracle(a,b)
        assert all(v.flat[i]==e for v,e in zip(result,expected))
    assert all(v.shape==shape and v.dtype==np.float64 for v in result)
    assert np.array_equal(r,original[0]) and np.array_equal(t,original[1])


def test_source_example_without_premature_rounding():
    _,v=circular_orbit_speed(np.array(149600000000.),np.array(31536000.))
    assert v==oracle(149600000000.,31536000.)[1]
    assert round(float(v)/1000,1)==29.8
    _,early=circular_orbit_speed(np.array(149600000000.),np.array(31600000.))
    assert round(float(early)/1000,1)==29.7


@pytest.mark.parametrize('r,t',[(1e307,1e307),(np.nextafter(0.,1.),1.),(1.,1e308)])
def test_extremes(r,t):
    assert tuple(float(x) for x in circular_orbit_speed(np.array(r),np.array(t)))==oracle(r,t)


@pytest.mark.parametrize('r,t',[
    (0,1),(-1,1),(1,0),(1,-1),(np.nan,1),(1,np.inf),
    ([],[]),([1,2],[1]),([1+1j],[1]),([True],[1]),(['1'],[1]),
    (1e308,1e308),(1,np.nextafter(0.,1.)),(np.nextafter(0.,1.),1e308),
])
def test_invalid(r,t):
    with pytest.raises(ValueError):circular_orbit_speed(r,t)


def test_context_isolation():
    previous=mpmath.mp.dps
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(lambda r:circular_orbit_speed(np.array(r),np.array(3.)),[1.,2.,4.]))
    assert mpmath.mp.dps==previous
    for r,out in zip([1.,2.,4.],results):assert tuple(float(x) for x in out)==oracle(r,3.)
