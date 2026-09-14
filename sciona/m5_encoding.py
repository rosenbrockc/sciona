"""Independent cutoff-fitted grouped target moments for M5 preprocessing.

The caller supplies encoded grouping roles and the source-appropriate cutoff.
Statistics use the entire eligible history, including a training row's own
target. This is not out-of-fold target encoding. Future rows receive fitted
group moments without contributing their targets.
"""
import numpy as np


def encode(values, times, keys, cutoff, groupings):
    """Produce mean/std column pairs in the requested grouping order.

    Keys must be complete integer codes. NaN targets are excluded. Singletons
    have undefined sample standard deviation; unseen groups have two NaNs.
    """
    values, times, keys = np.asarray(values), np.asarray(times), np.asarray(keys)
    if (values.ndim != 1 or not values.size or values.dtype.kind not in 'iuf'
            or times.shape != values.shape or times.dtype.kind not in 'iu'
            or keys.ndim != 2 or keys.shape[0] != values.size or keys.shape[1] == 0
            or keys.dtype.kind not in 'iu'):
        raise ValueError('Expected real targets, integer times and aligned integer role codes')
    if isinstance(cutoff, (bool, np.bool_)) or not isinstance(cutoff, (int, np.integer)):
        raise ValueError('Cutoff must be an integer')
    values = values.astype(np.float64)
    if np.isinf(values).any():
        raise ValueError('Infinite targets are unsupported')
    if not isinstance(groupings, (list, tuple)) or not groupings:
        raise ValueError('At least one grouping is required')
    normalized = []
    for grouping in groupings:
        if (not isinstance(grouping, (list, tuple)) or not grouping
                or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
                       or i < 0 or i >= keys.shape[1] for i in grouping)
                or len(set(grouping)) != len(grouping)):
            raise ValueError('Grouping must contain distinct valid role indices')
        normalized.append(tuple(grouping))
    if len(set(normalized)) != len(normalized):
        raise ValueError('Repeated groupings are unsupported')
    eligible = (times <= cutoff) & ~np.isnan(values)
    result = np.full((values.size, 2 * len(normalized)), np.nan)
    for column, grouping in enumerate(normalized):
        _, membership = np.unique(keys[:, grouping], axis=0, return_inverse=True)
        for group in np.unique(membership):
            selected = membership == group
            history = values[selected & eligible]
            if history.size:
                result[selected, 2 * column] = np.mean(history)
                if history.size > 1:
                    result[selected, 2 * column + 1] = np.std(history, ddof=1)
    with np.errstate(over='ignore'):
        stored = result.astype(np.float16)
    if np.isinf(stored).any():
        raise ValueError('Encoded feature exceeds finite float16 range')
    return stored
