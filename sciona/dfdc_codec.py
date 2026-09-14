"""Versioned JSON numerical transport for private DFDC runtime inputs.

Only explicit fixed-width numerical dtypes are supported. No pickle, object
arrays, file paths, compression or automatic downloads. Payloads are private
runtime data and must never be committed or included in evidence reports.
"""
import base64
import binascii
import math
from collections.abc import Mapping

import numpy as np
import torch

_DTYPES = {'uint8': np.dtype('u1'), 'int16': np.dtype('<i2'),
           'int64': np.dtype('<i8'), 'float16': np.dtype('<f2'), 'float32': np.dtype('<f4')}
DEFAULT_ARRAY_BYTES = 1024**3


def _limit(max_bytes):
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('numerical transport byte limit must be a positive integer')


def encode_array(value, *, max_bytes=DEFAULT_ARRAY_BYTES):
    _limit(max_bytes)
    if not isinstance(value, np.ndarray) or value.dtype.name not in _DTYPES:
        raise ValueError('unsupported numerical array dtype')
    if value.ndim > 8 or value.nbytes > max_bytes or not np.isfinite(value).all():
        raise ValueError('numerical array exceeds byte limit or contains nonfinite values')
    dtype = value.dtype.name
    canonical = np.asarray(value, dtype=_DTYPES[dtype], order='C')
    return {'version': 1, 'dtype': dtype, 'shape': list(value.shape),
            'data': base64.b64encode(canonical.tobytes(order='C')).decode('ascii')}


def decode_array(value, *, max_bytes=DEFAULT_ARRAY_BYTES):
    _limit(max_bytes)
    if not isinstance(value, dict) or set(value) != {'version', 'dtype', 'shape', 'data'}:
        raise ValueError('invalid numerical transport fields')
    if type(value['version']) is not int or value['version'] != 1:
        raise ValueError('unsupported numerical transport version')
    if not isinstance(value['dtype'], str) or value['dtype'] not in _DTYPES:
        raise ValueError('unsupported numerical transport dtype')
    shape = value['shape']
    if not isinstance(shape, list) or len(shape) > 8 or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError('invalid numerical transport shape')
    # Also reject impossible dimensions on empty arrays before NumPy conversion.
    if any(n > np.iinfo(np.intp).max for n in shape):
        raise ValueError('numerical transport dimension is too large')
    dtype = _DTYPES[value['dtype']]
    size = math.prod(shape) * dtype.itemsize
    data = value['data']
    if size > max_bytes or not isinstance(data, str) or len(data) != 4*((size+2)//3):
        raise ValueError('numerical transport byte count or limit is invalid')
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError('invalid numerical transport encoding') from error
    if len(raw) != size or base64.b64encode(raw).decode('ascii') != data:
        raise ValueError('noncanonical numerical transport encoding')
    try:
        result = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    except ValueError as error:
        raise ValueError('invalid numerical transport dimensions') from error
    if not np.isfinite(result).all():
        raise ValueError('numerical transport contains nonfinite values')
    return result


def encode_state(state, *, max_bytes=DEFAULT_ARRAY_BYTES):
    _limit(max_bytes)
    if not isinstance(state, Mapping) or not state:
        raise ValueError('expected a nonempty tensor state')
    result = {}; remaining = max_bytes
    for name, tensor in state.items():
        if (not isinstance(name, str) or not name or not isinstance(tensor, torch.Tensor)
                or tensor.layout != torch.strided or tensor.dtype not in (torch.float32, torch.int64)):
            raise ValueError('unsupported tensor state entry')
        if tensor.numel()*tensor.element_size() > remaining:
            raise ValueError('tensor state exceeds aggregate byte limit')
        # Empty buffers are representable even when the remaining budget is zero.
        result[name] = encode_array(tensor.detach().cpu().numpy(), max_bytes=max(1, remaining))
        remaining -= tensor.numel()*tensor.element_size()
    return result


def decode_state(state, *, max_bytes=DEFAULT_ARRAY_BYTES):
    _limit(max_bytes)
    if not isinstance(state, dict) or not state:
        raise ValueError('expected a nonempty encoded tensor state')
    result = {}; remaining = max_bytes
    for name, record in state.items():
        if (not isinstance(name, str) or not name or not isinstance(record, dict)
                or record.get('dtype') not in ('float32', 'int64')):
            raise ValueError('unsupported encoded tensor state entry')
        array = decode_array(record, max_bytes=max(1, remaining))
        if array.nbytes > remaining:
            raise ValueError('tensor state exceeds aggregate byte limit')
        result[name] = torch.from_numpy(array)
        remaining -= array.nbytes
    return result
