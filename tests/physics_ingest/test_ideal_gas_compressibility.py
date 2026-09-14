from decimal import Decimal, localcontext
import numpy as np
import pytest
import sympy as sp
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.ideal_gas_compressibility import ideal_gas_compressibility


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
    actual = ideal_gas_compressibility(values)
    np.testing.assert_array_equal(actual, reference(values))
    np.testing.assert_array_equal(values, before)
    assert actual.dtype == np.float64 and actual.shape == shape


@pytest.mark.parametrize('n,temperature,r', [(1,2,3), (3,4,5), (7,11,13)])
def test_independent_constant_temperature_volume_path(n,temperature,r):
    # Differentiate the separately constructed ideal-gas volume function.
    P = sp.Symbol('P', positive=True)
    volume = sp.Integer(n*r*temperature)/P
    kappa = -sp.diff(volume,P)/volume
    for pressure in [1,2,4,8]:
        assert ideal_gas_compressibility(np.array(pressure)) == float(kappa.subs(P,pressure))


@pytest.mark.parametrize('value', [np.finfo(float).max, np.finfo(float).tiny, 1e-308, 1., 300.])
def test_extremes(value):
    np.testing.assert_array_equal(ideal_gas_compressibility(np.array(value)),reference(np.array(value)))


@pytest.mark.parametrize('bad', [[], [0.], [-1.], [np.nan], [np.inf], [True], ['300'], [1j], [None]])
def test_invalid_input(bad):
    with pytest.raises(ValueError):
        ideal_gas_compressibility(np.array(bad))


def test_reciprocal_overflow_rejected():
    with pytest.raises(ValueError):
        ideal_gas_compressibility(np.array(np.nextafter(0.,1.)))

