import numpy as np
import pytest
from sciona.m5_precision import downcast


@pytest.mark.parametrize('values,dtype', [
    (np.array([-127,126],dtype=np.int64),np.int8),
    (np.array([-128,126],dtype=np.int64),np.int16),
    (np.array([127],dtype=np.int64),np.int16),
    (np.array([32767],dtype=np.int64),np.int32),
    (np.array([-32768],dtype=np.int64),np.int32),
    (np.array([2147483647],dtype=np.int64),np.int64),
    (np.array([9223372036854775807],dtype=np.int64),np.int64),
    (np.array([65503.],dtype=np.float64),np.float16),
    (np.array([65504.],dtype=np.float64),np.float32),
    (np.array([-65504.],dtype=np.float64),np.float32),
    (np.array([np.finfo(np.float32).max],dtype=np.float64),np.float64),
    (np.array([np.nan],dtype=np.float16),np.float64),
    (np.array([np.inf],dtype=np.float32),np.float64),
    (np.array([np.nan,1.2],dtype=np.float64),np.float16),
    (np.array([0,2],dtype=np.uint64),np.uint64),
    (np.array([True]),np.bool_),
    (np.array([1],dtype=np.int8),np.int8),
])
def test_strict_boundaries_and_source_exclusions(values,dtype):
    result=downcast(values)
    assert result.dtype==dtype
    np.testing.assert_equal(result,values.astype(dtype))
    assert not np.shares_memory(result,values)


def test_range_selection_deliberately_permits_rounding():
    values=np.array([1.0001,1.0002])
    np.testing.assert_equal(downcast(values),[1.,1.])


def test_matrix_rejects():
    with pytest.raises(ValueError):
        downcast([[1.]])
