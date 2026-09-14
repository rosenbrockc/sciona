"""Explicit group-mean bounding for the second APTOS supplemental role.

The winning writeup demonstrates a lower deviation of 0.5. It does not specify
the full interval. Both deviations are therefore explicit caller choices;
symmetric 0.5 bounds are a reference reconstruction, not historical parity.
"""
import numbers

import numpy as np

from sciona.aptos_ensemble import _keys
from sciona.aptos_refinement import _align


def bound_group_targets(expected_keys, supplied_labels, first_stage_predictions,
                        *, lower_deviation, upper_deviation):
    """Bound predictions around their supplied-ordinal group's teacher mean.

    Groups use supplied four-level labels encoded 0..3, never rounded teacher predictions. Means use
    the complete supplied population before any bounding. Outputs stay soft
    and are not clipped to the ordinal label range. Caller identity order is
    preserved; arrays and teacher predictions are not changed in place.
    """
    for deviation in (lower_deviation, upper_deviation):
        if (isinstance(deviation, (bool, np.bool_))
                or not isinstance(deviation, numbers.Real)
                or not np.isfinite(deviation) or deviation < 0):
            raise ValueError('Finite nonnegative lower and upper deviations required')
    expected = _keys(expected_keys)
    labels = _align(expected, supplied_labels, ordinal=True)
    if (labels > 3).any():
        raise ValueError('Group-bounded supplemental labels must encode four levels as zero through three')
    predictions = _align(expected, first_stage_predictions, ordinal=False)
    result = predictions.copy()
    for label in np.unique(labels):
        group = labels == label
        # Divide first to avoid overflow when summing large finite predictions.
        mean = np.sum(predictions[group] / np.count_nonzero(group), dtype=np.float64)
        with np.errstate(over='ignore', invalid='ignore'):
            lower, upper = mean - lower_deviation, mean + upper_deviation
        if not np.isfinite([mean, lower, upper]).all():
            raise ValueError('Group mean or bound is not representable as finite float64')
        result[group] = np.clip(predictions[group], lower, upper)
    return result
