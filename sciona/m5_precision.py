"""Independent source-style column downcasting for M5 preprocessing.

Selection is based on strict range bounds, not numerical error. Values that
fit float16 can lose precision. All-missing or infinite floating columns use
float64. Unsigned, boolean, categorical and int8 columns are left unchanged.
"""
import numpy as np


def downcast(values):
    """Return a detached one-dimensional column at source-selected precision."""
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError('Expected a one-dimensional column')
    if str(values.dtype) not in ('int16', 'int32', 'int64', 'float16', 'float32', 'float64'):
        return values.copy()
    valid = values[~np.isnan(values)]
    if values.dtype.kind == 'i':
        if not valid.size:
            return values.copy()
        low, high = valid.min(), valid.max()
        for dtype in (np.int8, np.int16, np.int32, np.int64):
            limits = np.iinfo(dtype)
            if low > limits.min and high < limits.max:
                return values.astype(dtype)
        return values.copy()
    if valid.size:
        low, high = valid.min(), valid.max()
        for dtype in (np.float16, np.float32):
            limits = np.finfo(dtype)
            if low > limits.min and high < limits.max:
                return values.astype(dtype)
    return values.astype(np.float64)
