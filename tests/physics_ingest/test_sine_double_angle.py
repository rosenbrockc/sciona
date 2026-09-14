from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import mpmath
import numpy as np
import pytest
import sympy as sp
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.sine_double_angle import sine_double_angle
from sciona.physics_ingest.sine_double_angle_proof import build_proof,verify_proof,verify_source_equations


def reference(values):
    ctx=mpmath.mp.clone();ctx.dps=1000
    values=np.asarray(values)
    return np.asarray([float(ctx.sin(2*ctx.mpf(float(x)))) for x in values.flat]).reshape(values.shape)


@pytest.mark.parametrize('shape',[(),(30,),(2,3,4)])
def test_independent_doubled_argument_reference(shape):
    rng=np.random.default_rng(902)
    values=rng.choice([-1.,1.],size=shape)*np.exp(rng.uniform(-700,700,size=shape))
    before=values.copy()
    result=sine_double_angle(values)
    np.testing.assert_array_equal(result,reference(values))
    np.testing.assert_array_equal(values,before)
    assert result.dtype==np.float64 and result.shape==shape


@pytest.mark.parametrize('value',[0.,-0.,np.nextafter(0.,1.),-np.nextafter(0.,1.),np.finfo(float).tiny,
                                np.finfo(float).max,-np.finfo(float).max,1e308,np.pi/2,np.pi,np.nextafter(np.pi,0.)])
def test_extremes_and_near_roots(value):
    np.testing.assert_array_equal(sine_double_angle(np.array(value)),reference(np.array(value)))


@pytest.mark.parametrize('bad',[[],[np.nan],[np.inf],[-np.inf],[True],['1'],[1j],[None]])
def test_invalid_inputs(bad):
    with pytest.raises(ValueError):sine_double_angle(np.asarray(bad))


def test_precision_context_isolation():
    old=mpmath.mp.dps
    try:
        mpmath.mp.dps=7
        values=np.array([1e308,1e-308,np.pi])
        with ThreadPoolExecutor(max_workers=2) as pool:
            outputs=list(pool.map(sine_double_angle,[values,values]))
        for output in outputs:np.testing.assert_array_equal(output,reference(values))
        assert mpmath.mp.dps==7
    finally:mpmath.mp.dps=old


def test_six_step_proof():
    assert all(verify_proof(build_proof())['checks'].values())


@pytest.mark.parametrize('i',range(6))
def test_corrupted_transition_rejected(i):
    proof=build_proof();steps=list(proof.steps)
    steps[i]=sp.Eq(sp.Symbol('bad'),0,evaluate=False)
    with pytest.raises(ValueError):verify_proof(replace(proof,steps=tuple(steps)))


def test_unreviewed_source_identity_rejected():
    with pytest.raises(ValueError):
        verify_source_equations({'bad':dict(sympy_lhs="Symbol('arbitrary_i')",sympy_rhs='Integer(0)')},{})
