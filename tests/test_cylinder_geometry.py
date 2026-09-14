import numpy as np
import pytest
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.particle_tracking.track_matching.cylinder_geometry import next_cylinder_intersection


def inputs(target=5., direction=1., phase=0., pitch=2*np.pi):
    return {k: np.array([v]) for k, v in dict(x0=2+np.cos(phase), y0=np.sin(phase), z0=0.,
        hel_xm=2., hel_ym=0., hel_r=1., hel_pitch=pitch, sign_uz=direction, target_r2sqr=target).items()}


@pytest.mark.parametrize('direction', [-1., 1.])
@pytest.mark.parametrize('pitch', [-2*np.pi, 2*np.pi])
@pytest.mark.parametrize('phase', [0., .3, 2., 5.])
def test_valid_surface_and_direction(direction, pitch, phase):
    args = inputs(direction=direction, pitch=pitch, phase=phase)
    old = {k: v.copy() for k, v in args.items()}
    x, y, z, dp, valid = next_cylinder_intersection(**args)
    assert valid[0]
    np.testing.assert_allclose(x*x+y*y, 5., rtol=1e-12)
    np.testing.assert_allclose((x-2)**2+y*y, 1., rtol=1e-12)
    assert direction*z[0] > 0
    assert direction*np.sign(pitch)*dp[0] > 0
    for k in args:
        np.testing.assert_array_equal(args[k], old[k])


@pytest.mark.parametrize('direction', [-1., 1.])
def test_next_turn_is_not_zero(direction):
    _, _, _, phase, valid = next_cylinder_intersection(**inputs(target=9., direction=direction))
    assert valid[0]
    np.testing.assert_allclose(phase, direction*2*np.pi, rtol=1e-8)


@pytest.mark.parametrize('target', [.25, 16.])
def test_unreachable_is_masked(target):
    *values, valid = next_cylinder_intersection(**inputs(target=target))
    assert not valid[0]
    assert all(np.array_equal(v, [0.]) for v in values)


@pytest.mark.parametrize('name,value', [('sign_uz', 0.), ('hel_pitch', 0.), ('hel_r', -1.), ('target_r2sqr', 0.), ('x0', 4.), ('z0', np.nan)])
def test_invalid_geometry_fails(name, value):
    args = inputs(); args[name] = np.array([value])
    with pytest.raises(ValueError):
        next_cylinder_intersection(**args)


def test_coaxial_is_explicitly_excluded():
    args = inputs(); args['hel_xm'][:] = 0.; args['x0'][:] = 1.
    with pytest.raises(ValueError, match='Coaxial'):
        next_cylinder_intersection(**args)


@pytest.mark.parametrize('direction,expected', [(1., 1.5*np.pi-3.), (-1., .5*np.pi-3.)])
def test_earliest_forward_crossing(direction, expected):
    _, _, _, phase, valid = next_cylinder_intersection(**inputs(direction=direction, phase=3.))
    assert valid[0]
    np.testing.assert_allclose(phase, expected, rtol=1e-12)


def test_mixed_batch_keeps_validity_per_row():
    first, second = inputs(), inputs(target=16.)
    args = {k: np.concatenate([first[k], second[k]]) for k in first}
    x, y, z, phase, valid = next_cylinder_intersection(**args)
    np.testing.assert_array_equal(valid, [True, False])
    np.testing.assert_allclose(x[0]**2+y[0]**2, 5.)
    assert x[1] == y[1] == z[1] == phase[1] == 0
