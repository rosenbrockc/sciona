"""Synthetic identity and arithmetic checks for specified label averaging."""
import numpy as np
import pytest
from sciona.aptos_refinement import average_ordinal_targets


def test_reorders_both_inputs_and_preserves_soft_values():
    labels = (['synthetic-b', 'synthetic-a'], np.array([4, 0]))
    predictions = (['synthetic-a', 'synthetic-b'], np.array([1.5, 2.5]))
    result = average_ordinal_targets(['synthetic-a', 'synthetic-b'], labels, predictions)
    np.testing.assert_array_equal(result, [.75, 3.25])
    np.testing.assert_array_equal(labels[1], [4, 0])
    np.testing.assert_array_equal(predictions[1], [1.5, 2.5])
    assert result.dtype == np.float64


def test_no_implicit_clipping_of_teacher_predictions():
    keys = ['synthetic-a', 'synthetic-b']
    result = average_ordinal_targets(keys, (keys, np.array([0, 4])), (keys, np.array([-2., 8.])))
    np.testing.assert_array_equal(result, [-1., 6.])


@pytest.mark.parametrize('fault', ['fractional', 'out_of_range', 'boolean', 'nonfinite',
                                  'rank', 'missing', 'duplicate', 'integer_teacher'])
def test_invalid_label_or_teacher_contract_rejects(fault):
    keys = ['synthetic-a', 'synthetic-b']
    labels = (keys, np.array([0, 4]))
    predictions = (keys, np.array([1., 3.]))
    if fault == 'fractional': labels = (keys, np.array([0., 2.5]))
    elif fault == 'out_of_range': labels = (keys, np.array([0, 5]))
    elif fault == 'boolean': labels = (keys, np.array([False, True]))
    elif fault == 'nonfinite': predictions = (keys, np.array([np.nan, 2.]))
    elif fault == 'rank': predictions = (keys, np.ones((2, 1)))
    elif fault == 'missing': predictions = (keys[:1], np.array([1.]))
    elif fault == 'duplicate': predictions = ([keys[0]] * 2, np.array([1., 2.]))
    else: predictions = (keys, np.array([1, 2]))
    with pytest.raises(ValueError):
        average_ordinal_targets(keys, labels, predictions)
