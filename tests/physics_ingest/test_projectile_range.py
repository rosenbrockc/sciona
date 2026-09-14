import mpmath
import numpy as np
import pytest
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.projectile_range import projectile_range, witness_projectile_range
from sciona.ghost.abstract import AbstractArray


def reference(speed, gravity, angle):
    # Independent Cartesian decomposition using complex exponential, followed
    # by the nonzero root of vertical motion and horizontal transport.
    ctx = mpmath.mp.clone()
    ctx.dps = 1200
    v, g, theta = (ctx.mpf(float(x)) for x in (speed, gravity, angle))
    phase = ctx.exp(ctx.j*theta)
    vx, vy = v*phase.real, v*phase.imag
    flight = 2*vy/g
    return tuple(float(x) for x in (flight, vx*flight, ctx.atan(1), (vx*vx+vy*vy)/g))


@pytest.mark.parametrize('row', [
    (1., 1., 0.), (1., 1., np.pi/4), (1., 1., np.pi/2),
    (1e308, 1e308, .3), (1e-308, 1e-308, .3),
    (1., 1., np.nextafter(0., 1.)), (1e150, 1., .7),
])
def test_extremes_against_cartesian_oracle(row):
    actual = projectile_range(*(np.array(x) for x in row))
    np.testing.assert_array_equal([x.item() for x in actual], reference(*row))


def test_array_shapes_copy_and_reference():
    rng = np.random.default_rng(624)
    args = [np.exp(rng.uniform(-80, 80, (2, 3, 4))),
            np.exp(rng.uniform(-80, 80, (2, 3, 4))), rng.uniform(.001, 1.57, (2, 3, 4))]
    before = [a.copy() for a in args]
    actual = projectile_range(*args)
    expected = [reference(*row) for row in zip(*(a.flat for a in args))]
    for j, output in enumerate(actual):
        assert output.shape == args[0].shape and output.dtype == np.float64
        np.testing.assert_array_equal(output, np.array([r[j] for r in expected]).reshape(args[0].shape))
        assert all(not np.shares_memory(output, a) for a in args)
    for a, b in zip(args, before):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('row', [
    (0., 1., .3), (-1., 1., .3), (1., 0., .3), (1., -1., .3),
    (1., 1., -.1), (1., 1., np.nextafter(np.pi/2, np.inf)),
    (np.inf, 1., .3), (1., np.nan, .3), (1., 1., np.inf),
    (1e308, 1., .3), (1e-308, 1., .3),
    (1., 1e308, np.nextafter(0., 1.)),
])
def test_domain_and_output_range_rejected(row):
    with pytest.raises(ValueError):
        projectile_range(*(np.array(x) for x in row))


@pytest.mark.parametrize('bad', [np.array([]), np.array([True]), np.array([1j]),
                               np.array(['1']), np.array([1], dtype=object)])
def test_non_real_or_empty_rejected(bad):
    with pytest.raises(ValueError):
        projectile_range(bad, np.ones(bad.shape), np.zeros(bad.shape))


def test_broadcasting_rejected():
    with pytest.raises(ValueError):
        projectile_range(np.ones(2), np.array(1.), np.zeros(2))


def test_zero_and_near_vertical_semantics():
    zero = projectile_range(np.array(1.), np.array(1.), np.array(0.))
    vertical = projectile_range(np.array(1.), np.array(1.), np.array(np.pi/2))
    assert zero[0] == zero[1] == 0
    assert vertical[1] > 0 and vertical[1] < 1e-15
    assert zero[2] == np.pi/4 and zero[3] == 1


def test_witness_shapes():
    a = AbstractArray(shape=(2, 3), dtype='float64')
    assert all(x.shape == (2, 3) for x in witness_projectile_range(a, a, a))
    with pytest.raises(ValueError):
        witness_projectile_range(a, a, AbstractArray(shape=(1,), dtype='float64'))
