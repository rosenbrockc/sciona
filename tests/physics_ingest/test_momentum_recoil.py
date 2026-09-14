from decimal import Decimal, localcontext
import numpy as np
import pytest
from sciona.atoms.physical_quantities.momentum_recoil import momentum_recoil


def reference(a, b):
    with localcontext() as ctx:
        ctx.prec = 2500
        differences = [[Decimal.from_float(float(x))-Decimal.from_float(float(y)) for x, y in zip(u, v)]
                       for u, v in zip(a.reshape(-1, 3), b.reshape(-1, 3))]
        return (np.asarray([[float(d) for d in row] for row in differences]).reshape(a.shape),
                np.asarray([float(sum(d*d for d in row)) for row in differences]).reshape(a.shape[:-1]))


@pytest.mark.parametrize('a,b', [([3., 4., 0.], [1., 2., 2.]), ([0., 0., 0.], [0., 0., 0.]),
    ([1e308]*3, [1e308]*3), ([1., 2., 3.], [np.nextafter(1., 0.), 2., 3.]),
    ([2e-162, 0., 0.], [0., 0., 0.]), ([-1e150, 2e150, -3e150], [1e150, -2e150, 3e150])])
def test_exact_decimal_reference(a, b):
    a, b = np.asarray(a), np.asarray(b)
    actual, expected = momentum_recoil(a, b), reference(a, b)
    for output, target in zip(actual, expected):
        np.testing.assert_array_equal(output, target)


def test_batch_shapes_no_mutation_and_direction_symmetry():
    rng = np.random.default_rng(829)
    a, b = rng.normal(size=(2, 40, 3)), rng.normal(size=(2, 40, 3))
    old_a, old_b = a.copy(), b.copy()
    vector, norm = momentum_recoil(a, b)
    expected = reference(a, b)
    np.testing.assert_array_equal(vector, expected[0])
    np.testing.assert_array_equal(norm, expected[1])
    reverse, reverse_norm = momentum_recoil(b, a)
    np.testing.assert_array_equal(vector, -reverse)
    np.testing.assert_array_equal(norm, reverse_norm)
    np.testing.assert_array_equal(a, old_a)
    np.testing.assert_array_equal(b, old_b)


@pytest.mark.parametrize('bad', [1., [], [1., 2.], [[1., 2.]], [True]*3, [1j]*3, ['1']*3,
    [np.nan, 0., 0.], [np.inf, 0., 0.], np.ones((2, 3))])
@pytest.mark.parametrize('index', [0, 1])
def test_bad_inputs_rejected(bad, index):
    inputs = [np.zeros(3), np.zeros(3)]
    inputs[index] = bad
    with pytest.raises(ValueError):
        momentum_recoil(*inputs)


@pytest.mark.parametrize('a,b', [([1e308]*3, [-1e308]*3), ([1e200]*3, [0.]*3), ([1e-200]*3, [0.]*3)])
def test_unrepresentable_output_rejected(a, b):
    with pytest.raises(ValueError, match='outside'):
        momentum_recoil(a, b)
