import mpmath
import numpy as np
import pytest
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.spring_mass import spring_mass


def reference(mass, stiffness, amplitude, time):
    ctx = mpmath.mp.clone()
    ctx.dps = 1800
    m, k, A, t = (ctx.mpf(float(v)) for v in (mass, stiffness, amplitude, time))
    w = ctx.sqrt(k)/ctx.sqrt(m)
    z = ctx.exp(ctx.j*w*t)
    return tuple(float(v) for v in (w, A*z.real, -A*w*z.imag, -A*w*w*z.real))


@pytest.mark.parametrize('shape', [(), (20,), (2, 3, 4)])
def test_precision_and_preservation(shape):
    rng = np.random.default_rng(844)
    args = [np.asarray(np.exp(rng.uniform(-100, 100, size=shape))) for _ in range(4)]
    args[2] *= -1
    copies = [a.copy() for a in args]
    result = spring_mass(*args)
    for i, row in enumerate(zip(*(a.flat for a in args))):
        assert tuple(out.flat[i] for out in result) == reference(*row)
    assert all(a.shape == shape and a.dtype == np.float64 for a in result)
    assert all(np.array_equal(a, b) for a, b in zip(args, copies))


@pytest.mark.parametrize('args', [(2., 8., 3., 0.), (2., 8., -3., -2.),
                                 (1., 1., 0., 1e308), (1e-308, 1e308, 1e-308, 1e308),
                                 (1e308, 1e308, 1., 1e308),
                                 (1., 1., np.nextafter(0., 1.), 0.)])
def test_special(args):
    assert tuple(float(v) for v in spring_mass(*args)) == reference(*args)


def test_initial_conditions_and_source_regression():
    w, x, v, a = spring_mass(2., 8., 3., 0.)
    assert (w, x, v, a) == (2., 3., 0., -12.)
    assert w != 8./2.


def test_energy_and_time_parity():
    t = np.linspace(-10, 10, 41)
    m, k, A = (np.full_like(t, v) for v in (2., 8., -3.))
    w, x, v, a = spring_mass(m, k, A, t)
    np.testing.assert_allclose(m*v*v/2+k*x*x/2, k*A*A/2, rtol=3e-16)
    np.testing.assert_allclose(m*a+k*x, 0, atol=1e-14)
    reverse = spring_mass(m, k, A, -t)
    assert np.array_equal(x, reverse[1]) and np.array_equal(v, -reverse[2])


@pytest.mark.parametrize('args', [(0, 1, 1, 1), (1, -1, 1, 1), (1, 1, np.inf, 1),
                                 (1, 1, 1, np.nan), (True, 1, 1, 1),
                                 (1, 1, 1j, 1), ('1', 1, 1, 1),
                                 ([], [], [], []), ([1, 2], [1], [1], [1]),
                                 (np.nextafter(0., 1.), 1e308, 0, 0),
                                 (1e308, np.nextafter(0., 1.), 1, 0)])
def test_invalid(args):
    with pytest.raises(ValueError):
        spring_mass(*args)


def test_context_unchanged():
    original = mpmath.mp.dps
    spring_mass(1, 2, 3, 4)
    assert mpmath.mp.dps == original
