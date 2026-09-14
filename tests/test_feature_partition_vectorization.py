import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.feature_vectorization import vectorize_feature_partitions


def test_feature_order_noncontiguous_input_and_independent_outputs():
    train = np.arange(24.).reshape(2, 3, 4)[:, :, ::-1]
    prediction = np.arange(12.).reshape(1, 3, 4)[:, :, ::-1]
    a, b = vectorize_feature_partitions(train, prediction)
    assert a.shape == (2, 12) and b.shape == (1, 12)
    np.testing.assert_array_equal(a[0], [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8])
    np.testing.assert_array_equal(b[0], a[0])
    a[:] = -1
    b[:] = -2
    assert np.all(train >= 0) and np.all(prediction >= 0)


def test_equal_flat_size_does_not_hide_feature_axis_mismatch():
    with pytest.raises(ValueError, match='shapes'):
        vectorize_feature_partitions(np.ones((2, 3, 4)), np.ones((1, 4, 3)))


@pytest.mark.parametrize('value', [np.ones(4), np.ones((0, 2)), np.full((2, 3), np.nan), np.ones((2, 3), dtype=complex)])
def test_invalid_feature_tensors_rejected(value):
    with pytest.raises(ValueError):
        vectorize_feature_partitions(value, value)
