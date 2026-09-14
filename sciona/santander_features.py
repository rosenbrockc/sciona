"""Feature operations for the pending corrected Santander realization.

Independent implementation of the verified test-population singleton rule.
This module alone is not a complete competition solution or approved CDG.
"""
import numpy as np


def real_test_mask(test_features):
    """Retain rows with at least one column value occurring once in this input.

The counting population is exclusively the supplied test matrix. Exact numeric
value equality applies; there is no rounding, tolerance or training population.
All columns participate. This competition-specific rule is not a general-purpose
synthetic-data detector. Inputs must be a nonempty finite numeric matrix.
"""
    values = np.asarray(test_features)
    if values.ndim != 2 or not all(values.shape):
        raise ValueError('Expected a nonempty two-dimensional matrix')
    if values.dtype.kind not in 'iuf' or not np.isfinite(values).all():
        raise ValueError('Expected finite real numeric values')
    retained = np.zeros(values.shape[0], dtype=bool)
    for column in values.T:
        _, inverse, counts = np.unique(column, return_inverse=True, return_counts=True)
        retained |= counts[inverse] == 1
    return retained
