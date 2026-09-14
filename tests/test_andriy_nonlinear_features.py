import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.andriy_nonlinear_features import andriy_nonlinear_features


def test_linear_ramp_energy_length_and_entropy():
    x = np.arange(4.)[None, :]
    original = x.copy()
    np.testing.assert_allclose(andriy_nonlinear_features(x)[0], [1., 3., np.log(4)+.125])
    np.testing.assert_array_equal(x, original)


def test_histogram_boundary_ties_enter_upper_bin():
    actual = andriy_nonlinear_features(np.array([[0., 1.5, 1.5, 3.]]))[0, 2]
    p = np.array([.25, .75])
    assert actual == pytest.approx(-p@np.log(p)+np.log(2)+.125)


def test_sinusoidal_energy_and_amplitude_scaling():
    x = np.sin(np.arange(100)*.3)[None, :]
    before = andriy_nonlinear_features(x)[0]
    after = andriy_nonlinear_features(-3*x)[0]
    assert before[0] == pytest.approx(np.sin(.3)**2)
    np.testing.assert_allclose(after, [9*before[0], 3*before[1], before[2]+np.log(3)])


@pytest.mark.parametrize('x', [np.ones(30), np.ones((1, 2)), np.ones((0, 30)), np.full((1, 30), np.nan), np.full((1, 30), np.inf), np.ones((1, 30), dtype=complex), np.ones((1, 30))])
def test_invalid_windows(x):
    with pytest.raises(ValueError):
        andriy_nonlinear_features(x)
