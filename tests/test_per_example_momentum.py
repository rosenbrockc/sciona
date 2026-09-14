import numpy as np
import pytest
from sciona.atoms.ml.gradient_attacks.per_example import per_example_l1_momentum, per_example_std_momentum

FUNCTIONS = [per_example_l1_momentum, per_example_std_momentum]


@pytest.mark.parametrize('fn', FUNCTIONS)
def test_batch_split_permutation_and_input_preservation(fn):
    rng = np.random.default_rng(603)
    gradient = rng.normal(size=(3, 2, 3, 2))*np.array([1., 100., .01])[:, None, None, None]
    previous = rng.normal(size=gradient.shape)
    originals = gradient.copy(), previous.copy()
    result = fn(gradient, previous, .7)
    split = np.concatenate([fn(g[None], p[None], .7) for g, p in zip(gradient, previous)])
    np.testing.assert_array_equal(result, split)
    np.testing.assert_array_equal(fn(gradient[::-1], previous[::-1], .7), result[::-1])
    np.testing.assert_array_equal(gradient, originals[0])
    np.testing.assert_array_equal(previous, originals[1])


def test_l1_known_values_and_nonzero_constant_gradient():
    g = np.array([1., -3.]).reshape(1, 1, 2, 1)
    np.testing.assert_array_equal(per_example_l1_momentum(g, np.ones_like(g), .5), [[[[1.], [-1.]]]])
    np.testing.assert_array_equal(per_example_l1_momentum(np.full_like(g, 2.), np.zeros_like(g)), np.ones_like(g))


def test_std_does_not_subtract_mean():
    g = np.array([1., 3.]).reshape(1, 1, 2, 1)
    np.testing.assert_array_equal(per_example_std_momentum(g, np.zeros_like(g)), g)


@pytest.mark.parametrize('fn', FUNCTIONS)
def test_zero_gradient_fails_without_epsilon(fn):
    with pytest.raises(ValueError, match='denominator'):
        fn(np.zeros((1, 2, 2, 1)), np.zeros((1, 2, 2, 1)))


def test_std_constant_gradient_and_canceled_history_fail():
    g = np.array([1., 3.]).reshape(1, 1, 2, 1)
    with pytest.raises(ValueError, match='denominator'):
        per_example_std_momentum(np.ones_like(g), np.zeros_like(g))
    with pytest.raises(ValueError, match='denominator'):
        per_example_std_momentum(g, -g)


@pytest.mark.parametrize('fn', FUNCTIONS)
@pytest.mark.parametrize('momentum', [-1., np.inf, np.nan, True, [1.]])
def test_invalid_momentum_rejected(fn, momentum):
    g = np.arange(1., 5.).reshape(1, 2, 2, 1)
    with pytest.raises(ValueError):
        fn(g, np.zeros_like(g), momentum)


@pytest.mark.parametrize('fn', FUNCTIONS)
@pytest.mark.parametrize('bad', [np.ones((2, 2, 1)), np.ones((2, 2, 2, 1)), np.empty((0, 2, 2, 1)),
    np.ones((1, 2, 2, 1), dtype=bool), np.ones((1, 2, 2, 1), dtype=complex), np.full((1, 2, 2, 1), np.nan)])
def test_invalid_gradient_rejected(fn, bad):
    with pytest.raises(ValueError):
        fn(bad, np.zeros((1, 2, 2, 1)))
