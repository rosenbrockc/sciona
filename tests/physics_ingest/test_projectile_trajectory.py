from decimal import Decimal, localcontext
from dataclasses import replace
import numpy as np
import pytest
import sympy as sp
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.projectile_trajectory import projectile_trajectory
from sciona.physics_ingest.projectile_proof import build_proof, verify_proof, symbols


def reference(args):
    with localcontext() as ctx:
        ctx.prec = 2500
        outputs = []
        for row in zip(*(np.asarray(a).flat for a in args)):
            x, x0, y0, vx, vy, g = [Decimal.from_float(float(v)) for v in row]
            t = (x-x0)/vx
            outputs.append((float(t), float(y0+vy*t-g*t*t/2)))
    return tuple(np.array([r[i] for r in outputs]).reshape(np.asarray(args[0]).shape) for i in range(2))


def arguments():
    return [np.array([v]) for v in [12., 2., 3., 5., 7., 10.]]


@pytest.mark.parametrize('shape', [(), (20,), (2, 3, 4)])
def test_independent_decimal_reference(shape):
    rng = np.random.default_rng(890)
    x0 = rng.normal(size=shape); vx = rng.uniform(1, 8, size=shape)*rng.choice([-1, 1], size=shape)
    args = [x0+vx*rng.uniform(0, 10, size=shape), x0, rng.normal(size=shape), vx,
            rng.normal(size=shape), rng.uniform(0, 20, size=shape)]
    before = [a.copy() for a in args]
    for actual, expected in zip(projectile_trajectory(*args), reference(args)):
        np.testing.assert_array_equal(actual, expected)
    for a, b in zip(args, before): np.testing.assert_array_equal(a, b)


def test_gravity_linear_and_initial_vertical_velocity():
    t, y = projectile_trajectory(*arguments())
    np.testing.assert_array_equal(t, [2.]); np.testing.assert_array_equal(y, [-3.])


@pytest.mark.parametrize('row', [[-12., -2., 3., -5., 7., 10.], [2., 2., 3., 5., 7., 10.],
                               [1e308, -1e308, 0., 1e308, 0., 1.],
                               [np.nextafter(0., 1.), 0., 0., 1., 1., 0.],
                               [1., 0., -1e308, 1., 1e308, 0.]])
def test_direction_initial_point_and_extremes(row):
    args = [np.array(v) for v in row]
    for actual, expected in zip(projectile_trajectory(*args), reference(args)):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('index,value', [(0, np.nan), (1, np.inf), (3, 0.), (5, -1.), (0, -3.)])
def test_invalid_domain(index, value):
    args = arguments(); args[index][:] = value
    with pytest.raises(ValueError): projectile_trajectory(*args)


@pytest.mark.parametrize('bad', [np.array([]), np.ones((2,)), np.ones((1, 1)), np.array([1j]), np.array([True]), np.array(['1'])])
def test_bad_input_shape_and_type(bad):
    args = arguments(); args[0] = bad
    with pytest.raises(ValueError): projectile_trajectory(*args)


@pytest.mark.parametrize('row', [[1., 0., 0., 1e-308, 0., 10.],
                               [np.nextafter(0., 1.), 0., 0., 2., 0., 0.]])
def test_output_overflow_or_underflow_fails(row):
    with pytest.raises(ValueError): projectile_trajectory(*[np.array(v) for v in row])


def test_corrected_proof():
    assert verify_proof(build_proof())['checks']['time_substitution']


@pytest.mark.parametrize('index', [0, 1])
def test_corrupted_proof_step(index):
    proof = build_proof(); steps = list(proof.steps); steps[index] = sp.Eq(sp.Symbol('bad'), 0, evaluate=False)
    with pytest.raises(ValueError): verify_proof(replace(proof, steps=tuple(steps)))


def test_squared_gravity_rejected():
    proof = build_proof(); s = symbols(); steps = list(proof.steps)
    steps[1] = sp.Eq(steps[1].lhs, steps[1].rhs.subs(s['g'], s['g']**2), evaluate=False)
    with pytest.raises(ValueError): verify_proof(replace(proof, steps=tuple(steps)))
