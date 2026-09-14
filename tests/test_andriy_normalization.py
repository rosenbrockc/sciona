import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.andriy_normalization import andriy_joint_normalization


def test_joint_sample_variance_and_invalid_prediction_exclusion():
    train = np.array([[1., 10.], [3., 30.]])
    prediction = np.array([[5., 50.], [np.nan, 900.]])
    before = prediction.copy()
    a, b = andriy_joint_normalization(train, prediction, np.array([True, False]))
    np.testing.assert_array_equal(a, [[-1., -1.], [0., 0.]])
    np.testing.assert_array_equal(b[0], [1., 1.])
    assert np.isnan(b[1, 0]) and b[1, 1] == 43.5
    np.testing.assert_array_equal(prediction, before)
    assert not np.shares_memory(a, train) and not np.shares_memory(b, prediction)


def test_zero_variance_source_nonfinite_output_is_preserved():
    a, b = andriy_joint_normalization(np.ones((2, 1)), np.array([[2.]]), np.array([False]))
    assert np.all(np.isnan(a)) and np.isposinf(b[0, 0])


def test_valid_prediction_changes_training_transform():
    train = np.array([[0.], [2.]])
    a, _ = andriy_joint_normalization(train, np.array([[10.]]), np.array([True]))
    b, _ = andriy_joint_normalization(train, np.array([[10.]]), np.array([False]))
    assert not np.allclose(a, b)


@pytest.mark.parametrize('bad', ['width', 'mask_type', 'mask_shape', 'fit_nan', 'inf', 'few'])
def test_invalid_normalization_contract(bad):
    train, prediction, valid = np.zeros((2, 2)), np.zeros((1, 2)), np.array([True])
    if bad == 'width': prediction = np.zeros((1, 3))
    if bad == 'mask_type': valid = np.array([1])
    if bad == 'mask_shape': valid = np.array([True, False])
    if bad == 'fit_nan': prediction[0, 0] = np.nan
    if bad == 'inf': train[0, 0] = np.inf
    if bad == 'few': train, valid = train[:1], np.array([False])
    with pytest.raises(ValueError):
        andriy_joint_normalization(train, prediction, valid)
