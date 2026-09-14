"""Synthetic physical identities; no competition records or schema fixtures."""
import numpy as np
import pytest

from sciona.flavours_kinematics import mass_proxy


def test_mass_proxy_known_geometry_and_scaling():
    # Three 3-4-5 triangles give total longitudinal momentum 12.
    p = np.full((2, 3), 5.0)
    pt = np.full((2, 3), 3.0)
    actual = mass_proxy(p, pt, [5., 5.], [2., 4.], [1., 1.])
    np.testing.assert_allclose(actual, [6.5, 3.25], rtol=1e-14)
    np.testing.assert_allclose(
        mass_proxy(p * 2, pt * 2, [10., 10.], [2., 4.], [1., 1.]),
        actual * 2, rtol=1e-14,
    )


def test_zero_momentum_and_collinear_boundary():
    result = mass_proxy([[0., 4., 0.]], [[0., 4., 0.]], [3.], [2.], [2.])
    np.testing.assert_array_equal(result, [3.])


@pytest.mark.parametrize('change', ['shape', 'nonfinite', 'transverse', 'distance', 'time'])
def test_rejects_invalid_physical_inputs(change):
    args = [np.full((2, 3), 5.), np.full((2, 3), 3.),
            np.ones(2), np.ones(2), np.ones(2)]
    if change == 'shape': args[2] = np.ones((2, 1))
    elif change == 'nonfinite': args[0][0, 0] = np.nan
    elif change == 'transverse': args[1][0, 0] = 6.
    elif change == 'distance': args[3][0] = 0.
    else: args[4][0] = -1.
    with pytest.raises(ValueError):
        mass_proxy(*args)
