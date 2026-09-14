from dataclasses import replace
from decimal import Decimal, localcontext
import numpy as np
import pytest
import sympy as sp
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.ideal_gas_expansion import ideal_gas_expansion
from sciona.physics_ingest.ideal_gas_expansion_proof import build_proof, verify_proof


def reference(values):
    values = np.asarray(values)
    with localcontext() as ctx:
        ctx.prec = 2500
        result = [float(Decimal(1)/Decimal.from_float(float(t))) for t in values.flat]
    return np.asarray(result).reshape(values.shape)


@pytest.mark.parametrize('shape', [(), (20,), (2,3,4)])
def test_decimal_reference_and_input_preserved(shape):
    values = np.exp(np.random.default_rng(903).uniform(-700,700,size=shape))
    before = values.copy()
    actual = ideal_gas_expansion(values)
    np.testing.assert_array_equal(actual, reference(values))
    np.testing.assert_array_equal(values, before)
    assert actual.dtype == np.float64 and actual.shape == shape


@pytest.mark.parametrize('n,p,r', [(1,2,3), (3,4,5), (7,11,13)])
def test_independent_constant_pressure_volume_path(n,p,r):
    # Differentiate the separately constructed ideal-gas volume function.
    T = sp.Symbol('T', positive=True)
    volume = sp.Rational(n*r,p)*T
    alpha = sp.diff(volume,T)/volume
    for t in [1,2,4,8]:
        assert ideal_gas_expansion(np.array(t)) == float(alpha.subs(T,t))


@pytest.mark.parametrize('value', [np.finfo(float).max, np.finfo(float).tiny, 1e-308, 1., 300.])
def test_extremes(value):
    np.testing.assert_array_equal(ideal_gas_expansion(np.array(value)),reference(np.array(value)))


@pytest.mark.parametrize('bad', [[], [0.], [-1.], [np.nan], [np.inf], [True], ['300'], [1j], [None]])
def test_invalid_input(bad):
    with pytest.raises(ValueError):
        ideal_gas_expansion(np.array(bad))


def test_reciprocal_overflow_rejected():
    with pytest.raises(ValueError):
        ideal_gas_expansion(np.array(np.nextafter(0.,1.)))


def test_five_step_proof():
    assert all(verify_proof(build_proof())['checks'].values())


@pytest.mark.parametrize('i',range(5))
def test_corrupt_transition_rejected(i):
    proof=build_proof();steps=list(proof.steps)
    steps[i]=sp.Eq(sp.Symbol('bad'),0,evaluate=False)
    with pytest.raises(ValueError):
        verify_proof(replace(proof,steps=tuple(steps)))
