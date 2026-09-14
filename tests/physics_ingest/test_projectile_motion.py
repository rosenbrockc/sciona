"""Synthetic complex-velocity and endpoint-average integration reference."""
import mpmath
import numpy as np
import pytest
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.projectile_motion import projectile_motion


def reference(*row):
    ctx = mpmath.mp.clone(); ctx.dps = 3000
    x0, y0, speed, theta, g, t = (ctx.mpf(float(v)) for v in row)
    initial = speed*ctx.exp(ctx.j*theta)
    final = initial-ctx.j*g*t
    position = ctx.mpc(x0, y0)+(initial+final)*t/2
    return tuple(float(v) for v in (ctx.re(position), ctx.im(position), ctx.re(final), ctx.im(final)))


def test_random_multidimensional_reference():
    rng = np.random.default_rng(441)
    args = (rng.normal(size=(2, 3, 4)), rng.normal(size=(2, 3, 4)),
            np.exp(rng.uniform(-80, 80, (2, 3, 4))), rng.uniform(-100, 100, (2, 3, 4)),
            np.exp(rng.uniform(-80, 80, (2, 3, 4))), np.exp(rng.uniform(-20, 20, (2, 3, 4))))
    before = [a.copy() for a in args]
    actual = projectile_motion(*args)
    expected = [reference(*r) for r in zip(*(a.flat for a in args))]
    for i, value in enumerate(actual):
        np.testing.assert_array_equal(value, np.array([r[i] for r in expected]).reshape(2, 3, 4))
        assert value.dtype == np.float64
        assert all(not np.shares_memory(value, a) for a in args)
    for a, b in zip(args, before): np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('row', [(1, 2, 3, np.pi/2, 1, 4), (3, 4, 2, -3*np.pi/4, 1, 2),
    (1, 2, 0, 1e308, 1, 3), (1, 2, 3, -1e308, 0, 2), (1, 2, 3, 1e308, 1, 0),
    (-1e308, 1e308, 1e308, 0, 5e307, 2), (0, 0, np.nextafter(0., 1.), 0, 0, 1)])
def test_boundary_and_cancellation_states(row):
    actual = projectile_motion(*(np.array(v, dtype=float) for v in row))
    np.testing.assert_array_equal([v.item() for v in actual], reference(*row))
    assert all(v.shape == () for v in actual)


def test_rounded_vertical_angle_is_not_snapped():
    result = projectile_motion(*(np.array(v) for v in (0., 0., 1., np.pi/2, 0., 1.)))
    assert result[0] > 0 and result[2] > 0


@pytest.mark.parametrize('index,value', [(2, -1), (4, -1), (5, -1), (0, np.inf), (1, np.nan),
                                        (3, np.inf), (2, True), (1, 1j), (5, '1')])
def test_invalid_domains(index, value):
    args = [np.array(v) for v in (1., 2., 3., 0., 1., 1.)]; args[index] = np.array(value)
    with pytest.raises(ValueError): projectile_motion(*args)


def test_empty_and_broadcast_rejected():
    with pytest.raises(ValueError): projectile_motion(*(np.array([]) for _ in range(6)))
    with pytest.raises(ValueError, match='shapes'):
        projectile_motion(np.ones(2), *(np.ones(1) for _ in range(5)))


@pytest.mark.parametrize('row', [(0, 0, 1e308, 0, 0, 2), (0, 0, 0, 0, 1e308, 2),
                               (0, 0, 1e-308, np.pi/2, 0, 1)])
def test_nonzero_range_failure(row):
    with pytest.raises(ValueError, match='range'):
        projectile_motion(*(np.array(v, dtype=float) for v in row))
