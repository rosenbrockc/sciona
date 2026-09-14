"""Synthetic source-example and explicit reconstruction boundary checks."""
import math

import numpy as np
import pytest

from sciona.aptos_group_refinement import bound_group_targets


def test_source_lower_example_and_explicit_symmetric_reference():
    keys = ['synthetic-a', 'synthetic-b']
    predictions = np.array([1.1, 3.3])
    result = bound_group_targets(keys, (keys, [2, 2]), (keys, predictions),
                                lower_deviation=.5, upper_deviation=.5)
    np.testing.assert_allclose(result, [1.7, 2.7], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(predictions, [1.1, 3.3])


def test_scalar_oracle_separate_groups_and_independent_input_orders():
    rng = np.random.default_rng(83)
    keys = [f'synthetic-{i}' for i in range(35)]
    labels = rng.integers(0, 4, len(keys))
    predictions = rng.normal(2., 2., len(keys))
    left, right = rng.permutation(len(keys)), rng.permutation(len(keys))
    result = bound_group_targets(keys,
        ([keys[i] for i in left], labels[left]),
        ([keys[i] for i in right], predictions[right]),
        lower_deviation=.5, upper_deviation=.8)
    means = {label: math.fsum(float(p) for y, p in zip(labels, predictions) if y == label)
             / sum(y == label for y in labels) for label in set(labels)}
    expected = [min(max(p, means[y] - .5), means[y] + .8) for y, p in zip(labels, predictions)]
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-14)


def test_singletons_and_unclipped_regression_range():
    keys = ['synthetic-a', 'synthetic-b']
    result = bound_group_targets(keys, (keys, [0, 3]), (keys, [-2., 7.]),
                                lower_deviation=.5, upper_deviation=.5)
    np.testing.assert_array_equal(result, [-2., 7.])


@pytest.mark.parametrize('deviation', [-1., np.nan, np.inf, True, '0.5'])
@pytest.mark.parametrize('side', ['lower_deviation', 'upper_deviation'])
def test_invalid_bound_rejected(deviation, side):
    options = {'lower_deviation': .5, 'upper_deviation': .5, side: deviation}
    with pytest.raises(ValueError, match='deviations'):
        bound_group_targets(['synthetic-a'], (['synthetic-a'], [0]),
                            (['synthetic-a'], [1.]), **options)


def test_large_finite_group_mean_does_not_overflow():
    keys = ['synthetic-a', 'synthetic-b']
    result = bound_group_targets(keys, (keys, [0, 0]), (keys, [1e308, 1e308]),
                                lower_deviation=0., upper_deviation=0.)
    np.testing.assert_array_equal(result, [1e308, 1e308])


def test_fifth_supplied_group_label_rejected():
    with pytest.raises(ValueError, match='four levels'):
        bound_group_targets(['synthetic-a'], (['synthetic-a'], [4]),
                            (['synthetic-a'], [2.]), lower_deviation=.5, upper_deviation=.5)
