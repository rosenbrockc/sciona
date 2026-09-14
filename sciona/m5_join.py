"""Ordered, cardinality-preserving joins for independent M5 preprocessing."""
import numpy as np


def _keys(value):
    value=np.asarray(value)
    if value.ndim!=2 or not value.shape[1] or value.dtype.kind not in 'iu':
        raise ValueError('Expected a matrix of complete integer join keys')
    return value


def left_indices(left, right):
    """Return right-row indices, or -1 for missing keys, in left-row order.

    Duplicate right keys reject instead of silently multiplying feature rows.
    """
    left,right=_keys(left),_keys(right)
    if left.shape[1]!=right.shape[1]:
        raise ValueError('Join key widths differ')
    lookup={}
    for i,row in enumerate(right):
        key=tuple(map(int,row))
        if key in lookup:
            raise ValueError('Duplicate right join key')
        lookup[key]=i
    return np.array([lookup.get(tuple(map(int,row)),-1) for row in left],dtype=np.int64)


def take_numeric(values, indices):
    """Gather a numeric column, preserving its dtype unless missing keys require NaN."""
    values,indices=np.asarray(values),np.asarray(indices)
    if (values.ndim!=1 or values.dtype.kind not in 'iuf' or indices.ndim!=1
            or indices.dtype.kind not in 'iu' or (indices < -1).any()
            or (indices >= len(values)).any()):
        raise ValueError('Invalid numeric column or gather indices')
    missing=indices==-1
    if not missing.any():
        return values[indices].copy()
    dtype=values.dtype if values.dtype.kind=='f' else np.dtype('float64')
    result=np.full(indices.shape,np.nan,dtype=dtype)
    result[~missing]=values[indices[~missing]]
    return result


def first_per_key(keys):
    """Source calendar week deduplication keeps the first input row, unsorted."""
    keys=_keys(keys)
    seen=set();positions=[]
    for i,row in enumerate(keys):
        key=tuple(map(int,row))
        if key not in seen:
            seen.add(key);positions.append(i)
    return np.array(positions,dtype=np.int64)
