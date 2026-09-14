from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal,localcontext
import mpmath
import numpy as np
import pytest
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.wave_relations import wave_relations


def reference(frequency,speed):
    ctx=mpmath.mp.clone();ctx.dps=1000
    result=[[],[],[],[]]
    for f,v in zip(np.asarray(frequency).flat,np.asarray(speed).flat):
        with localcontext() as decimal:
            decimal.prec=2500
            df,dv=Decimal.from_float(float(f)),Decimal.from_float(float(v))
            T,L=Decimal(1)/df,dv/df
        # Independently use cycle duration and wavelength to derive angular values.
        mt=1/ctx.mpf(float(f));ml=ctx.mpf(float(v))/ctx.mpf(float(f))
        for bucket,value in zip(result,[T,L,2*ctx.pi/mt,2*ctx.pi/ml]):bucket.append(float(value))
    return tuple(np.asarray(v).reshape(np.asarray(frequency).shape) for v in result)


@pytest.mark.parametrize('shape',[(),(20,),(2,3,4)])
def test_independent_reference_and_preserved_inputs(shape):
    rng=np.random.default_rng(114);f=np.exp(rng.uniform(-300,300,size=shape));v=np.exp(rng.uniform(-300,300,size=shape))
    before=f.copy(),v.copy()
    for a,b in zip(wave_relations(f,v),reference(f,v)):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(f,before[0]);np.testing.assert_array_equal(v,before[1])


def test_cycle_conventions():
    T,L,w,k=wave_relations(np.array(3.),np.array(12.))
    assert T==1/3 and L==4
    assert w==float(6*mpmath.pi) and k==float(mpmath.pi/2)
    assert k!=1/L


@pytest.mark.parametrize('f,v',[(1e307,1e308),(1e-307,1.),(1.,np.finfo(float).max),(np.finfo(float).tiny,np.finfo(float).tiny)])
def test_extreme_representable_outputs(f,v):
    for a,b in zip(wave_relations(np.array(f),np.array(v)),reference(np.array(f),np.array(v))):
        np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('f,v',[(np.nextafter(0.,1.),1.),(np.finfo(float).max,1.),(1.,np.nextafter(0.,1.)),(1e307,1e-307)])
def test_any_unrepresentable_output_rejects(f,v):
    with pytest.raises(ValueError):wave_relations(np.array(f),np.array(v))


@pytest.mark.parametrize('bad',[[],[0.],[-1.],[np.nan],[np.inf],[True],['1'],[1j],[None]])
def test_invalid_inputs(bad):
    for index in [0,1]:
        args=[np.ones(1),np.ones(1)];args[index]=np.asarray(bad)
        with pytest.raises(ValueError):wave_relations(*args)


def test_no_broadcasting():
    with pytest.raises(ValueError):wave_relations(np.ones(2),np.ones(1))


def test_precision_isolation():
    old=mpmath.mp.dps
    try:
        mpmath.mp.dps=7
        with ThreadPoolExecutor(max_workers=2) as pool:
            outputs=list(pool.map(lambda _:wave_relations(np.array(3.),np.array(12.)),range(2)))
        for result in outputs:
            for a,b in zip(result,reference(np.array(3.),np.array(12.))):np.testing.assert_array_equal(a,b)
        assert mpmath.mp.dps==7
    finally:mpmath.mp.dps=old
