import math
import mpmath as mp
import numpy as np
import pytest

from sciona.atoms.physical_quantities.circular_orbital_period import circular_orbital_period


def reference(r, a, b, g):
    with mp.workdps(180):
        return float(2*mp.pi*mp.sqrt(mp.mpf(float(r))**3/(mp.mpf(float(g))*(mp.mpf(float(a))+mp.mpf(float(b))))))


@pytest.mark.parametrize('values', [
    (1., 2., 3., 4.), (1e308, 1e308, 1e308, 1e308),
    (1e-300, 1e-300, 1e-300, 1e-300), (1e-200, 1e100, 1e100, 1e-100),
    (np.nextafter(0., 1.), np.nextafter(0., 1.), np.nextafter(0., 1.), 1.),
])
def test_high_precision_extremes(values):
    actual = circular_orbital_period(*values)
    assert actual.shape == ()
    np.testing.assert_array_max_ulp(actual, np.asarray(reference(*values)), maxulp=1)


def test_shapes_symmetry_no_mutation_and_dynamic_range():
    r = np.logspace(-100, 100, 201).reshape(3, 67)
    a = r*2
    b = r*3
    before = [v.copy() for v in (r, a, b)]
    actual = circular_orbital_period(r, a, b, 5.)
    expected = np.asarray([reference(*v, 5.) for v in zip(r.flat, a.flat, b.flat)]).reshape(r.shape)
    np.testing.assert_array_max_ulp(actual, expected, maxulp=1)
    np.testing.assert_array_equal(actual, circular_orbital_period(r, b, a, 5.))
    for source, prior in zip((r, a, b), before):
        np.testing.assert_array_equal(source, prior)


@pytest.mark.parametrize('bad', [0., -1., np.nan, np.inf, True, 1j, '1', [], [1.]])
@pytest.mark.parametrize('index', range(4))
def test_reject_invalid_inputs_or_broadcasting(bad, index):
    values = [1., 1., 1., 1.]
    values[index] = bad
    with pytest.raises((ValueError, FloatingPointError)):
        circular_orbital_period(*values)


@pytest.mark.parametrize('values', [(1e308, 1e-300, 1e-300, 1e-300), (1e-300, 1e300, 1e300, 1e300)])
def test_reject_unrepresentable_output(values):
    with pytest.raises(ValueError, match='outside'):
        circular_orbital_period(*values)
