"""Synthetic numeric validation against independent impulse/work integration."""
from decimal import Decimal, localcontext
import numpy as np
import pytest
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.mechanical_energy import mechanical_energy, witness_mechanical_energy
from sciona.ghost.abstract import AbstractArray


def reference(*row):
    with localcontext() as context:
        context.prec = 2500
        m, f, x0, u, t = (Decimal.from_float(float(v)) for v in row)
        impulse = f*t
        v = (m*u+impulse)/m
        displacement = (u+v)*t/2
        x = x0+displacement
        initial_kinetic = m*u*u/2
        work = f*displacement
        kinetic = initial_kinetic+work
        total = initial_kinetic-f*x0
        potential = total-kinetic
        return tuple(float(a) for a in (x, v, kinetic, potential, total, work))


def test_random_multidimensional_reference():
    rng = np.random.default_rng(721)
    args = [np.exp(rng.uniform(-80, 80, (2, 3, 4))), rng.normal(size=(2, 3, 4)),
            rng.normal(size=(2, 3, 4)), rng.normal(size=(2, 3, 4)), np.exp(rng.uniform(-20, 20, (2, 3, 4)))]
    before = [a.copy() for a in args]
    result = mechanical_energy(*args)
    expected = [reference(*row) for row in zip(*(a.flat for a in args))]
    for i, actual in enumerate(result):
        np.testing.assert_array_equal(actual, np.array([r[i] for r in expected]).reshape(2, 3, 4))
        assert actual.dtype == np.float64
        assert all(not np.shares_memory(actual, a) for a in args)
    for a, b in zip(args, before):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('row', [(2, -4, 3, 2, 2), (2, 0, -3, -2, 4), (2, 4, 1, 2, 0),
                               (2, 4, 1, 2, 1), (1e-308, 1e-308, 0, 0, 1),
                               (2, 1, 0, 0, 0), (2, -4, 3, 2, 1)])
def test_signed_boundary_states(row):
    args = tuple(np.array(v, dtype=float) for v in row)
    actual = mechanical_energy(*args)
    np.testing.assert_array_equal([a.item() for a in actual], reference(*row))
    assert all(a.shape == () for a in actual)


def test_cancellation_avoids_intermediate_velocity_overflow():
    row = (1e-308, -2, 0, 1e308, 1)
    actual = mechanical_energy(*(np.array(v) for v in row))
    np.testing.assert_array_equal([a.item() for a in actual], reference(*row))


@pytest.mark.parametrize('index,value', [(0, 0), (0, -1), (4, -1), (1, np.inf), (2, np.nan),
                                        (3, True), (1, 1j), (2, '3')])
def test_invalid_domain(index, value):
    args = [np.array(v) for v in (2., 1., 3., 2., 1.)]
    args[index] = np.array(value)
    with pytest.raises(ValueError):
        mechanical_energy(*args)


def test_empty_and_broadcast_inputs_rejected():
    with pytest.raises(ValueError):
        mechanical_energy(*(np.array([]) for _ in range(5)))
    with pytest.raises(ValueError, match='shapes'):
        mechanical_energy(np.ones(2), *(np.ones(1) for _ in range(4)))


@pytest.mark.parametrize('row', [(1, 0, 0, 1e308, 0), (1, 0, 0, 1e-308, 0),
                               (1, 1e308, 0, 0, 2)])
def test_unrepresentable_outputs_rejected(row):
    with pytest.raises(ValueError, match='range'):
        mechanical_energy(*(np.array(v) for v in row))


def test_witness_shape_contract():
    a = AbstractArray(shape=(2, 3), dtype='float64')
    assert len(witness_mechanical_energy(a, a, a, a, a)) == 6
    with pytest.raises(ValueError, match='shapes'):
        witness_mechanical_energy(a, a, a, a, AbstractArray(shape=(1,), dtype='float64'))
