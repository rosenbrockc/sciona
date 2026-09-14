"""Specified ordinal/teacher averaging for one APTOS refinement role.

The caller supplies first-stage predictions. The winning writeup does not
establish whether they came from individual teachers or an ensemble, so this
component does not choose or imply that upstream policy.
"""
import numpy as np
from sciona.aptos_ensemble import _keys


def _align(expected, entry, *, ordinal):
    if not isinstance(entry, (list, tuple)) or len(entry) != 2:
        raise ValueError('Runtime identities and a value vector required')
    keys = _keys(entry[0])
    values = np.asarray(entry[1])
    if (set(keys) != set(expected) or values.shape != (len(keys),)
            or values.dtype.kind not in ('i', 'u', 'f') or not np.isfinite(values).all()):
        raise ValueError('Complete aligned finite numeric values required')
    if ordinal:
        if ((values < 0).any() or (values > 4).any() or (values != np.floor(values)).any()):
            raise ValueError('Supplied ordinal labels must be integers from zero through four')
    elif values.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('Floating first-stage regression predictions required')
    positions = {key: i for i, key in enumerate(keys)}
    return values[[positions[key] for key in expected]].astype(np.float64)


def average_ordinal_targets(expected_keys, supplied_labels, first_stage_predictions):
    """Average aligned supplied ordinals and first-stage regression predictions.

    Returns float64 soft targets without rounding or clipping. Input identities
    remain private runtime values. This does not implement the separate
    group-mean outlier-bounding role or a complete second-stage population.
    """
    expected = _keys(expected_keys)
    labels = _align(expected, supplied_labels, ordinal=True)
    predictions = _align(expected, first_stage_predictions, ordinal=False)
    return (labels + predictions) / 2.
