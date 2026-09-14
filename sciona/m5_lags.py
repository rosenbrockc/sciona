"""Independent ordered-series lag features for the M5 reconstruction.

Rows are already in caller-defined chronological order within each group.
Missing rows are not filled or interpreted as elapsed calendar time. NaN values
propagate through incomplete rolling windows; stored features use float16,
whereas recursive temporary rolling means retain float64.
"""
import numpy as np


def _inputs(values, groups):
    values = np.asarray(values)
    groups = np.asarray(groups)
    if (values.ndim != 1 or values.size == 0 or values.dtype.kind not in 'iuf'
            or groups.shape != values.shape or groups.dtype.kind not in 'iu'):
        raise ValueError('Expected nonempty real values and aligned integer groups')
    values = values.astype(np.float64)
    if np.isinf(values).any():
        raise ValueError('Infinite values are unsupported')
    return values, groups


def _positive(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError('Shift and window must be positive integers')
    return int(value)


def shifted(values, groups, shift):
    """Shift independently within groups, preserving input row alignment."""
    values, groups = _inputs(values, groups)
    shift = _positive(shift)
    result = np.full(values.shape, np.nan)
    for group in np.unique(groups):
        positions = np.flatnonzero(groups == group)
        if len(positions) > shift:
            result[positions[shift:]] = values[positions[:-shift]]
    return result


def rolling(values, groups, shift, window, statistic='mean'):
    """Full-window mean or sample standard deviation after a grouped shift."""
    values, groups = _inputs(values, groups)
    shift, window = _positive(shift), _positive(window)
    if statistic not in ('mean', 'std'):
        raise ValueError('Unsupported rolling statistic')
    result = np.full(values.shape, np.nan)
    if statistic == 'std' and window == 1:
        return result
    for group in np.unique(groups):
        positions = np.flatnonzero(groups == group)
        sequence = values[positions]
        for current in range(shift + window - 1, len(sequence)):
            stop = current - shift + 1
            sample = sequence[stop - window:stop]
            if np.isnan(sample).any():
                continue
            result[positions[current]] = (np.mean(sample) if statistic == 'mean'
                                          else np.std(sample, ddof=1))
    return result


def _stored(values):
    with np.errstate(over='ignore'):
        result = values.astype(np.float16)
    if np.isinf(result).any():
        raise ValueError('Stored feature exceeds finite float16 range')
    return result


def features(values, groups, *, recursive=False):
    """Return source-ordered lags, fixed rolling statistics and temporary means.

    Recursive mode replaces only temporary means with full-precision values,
    as done when predictions are fed back into the next forecasting step.
    """
    values, groups = _inputs(values, groups)
    if not isinstance(recursive, bool):
        raise ValueError('Recursive mode must be boolean')
    result = {}
    for shift in range(28, 43):
        result[f'lag_{shift}'] = _stored(shifted(values, groups, shift))
    for window in (7, 14, 30, 60, 180):
        for statistic in ('mean', 'std'):
            result[f'{statistic}_{window}'] = _stored(rolling(values, groups, 28, window, statistic))
    for shift in (1, 7, 14):
        for window in (7, 14, 30, 60):
            value = rolling(values, groups, shift, window)
            result[f'temporary_{shift}_{window}'] = value if recursive else _stored(value)
    return result
