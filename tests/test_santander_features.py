"""Synthetic invariants for the verified singleton detector."""
import numpy as np
import pytest
from sciona.santander_features import real_test_mask


def test_singletons_retain_rows_not_exclude_them():
    values = np.array([[1, 4], [1, 5], [2, 4], [2, 6], [3, 4]])
    before = values.copy()
    np.testing.assert_array_equal(real_test_mask(values), [False, True, False, True, True])
    np.testing.assert_array_equal(values, before)


def test_row_and_column_permutation_equivariance():
    values = np.array([[1, 4], [1, 5], [2, 4], [2, 6], [3, 4]])
    order = np.array([4, 2, 0, 3, 1])
    np.testing.assert_array_equal(real_test_mask(values[order, ::-1]), real_test_mask(values)[order])


def test_counting_population_changes_result_and_must_be_test_only():
    test = np.array([[1], [2], [2]])
    np.testing.assert_array_equal(real_test_mask(test), [True, False, False])
    # A synthetic training observation would destroy the test singleton.
    combined = np.concatenate([test, [[1]]])
    assert not real_test_mask(combined).any()


def test_duplicate_population_has_no_singletons():
    assert not real_test_mask([[1, 2], [1, 2]]).any()
    np.testing.assert_array_equal(real_test_mask([[1, 2]]), [True])


def test_exact_equality_and_integer_precision_preserved():
    tiny = np.nextafter(1.0, 2.0)
    np.testing.assert_array_equal(real_test_mask([[1.0], [tiny]]), [True, True])
    values = np.array([[2**63], [2**63 + 1]], dtype=np.uint64)
    assert real_test_mask(values).all()
    assert not real_test_mask([[0.0], [-0.0]]).any()


@pytest.mark.parametrize('values', [[], [[]], [1, 2], [[[1]]], [[float('nan')]],
    [[float('inf')]], [['a']], [[True]], [[1+2j]]])
def test_invalid_matrices(values):
    with pytest.raises(ValueError):
        real_test_mask(values)
