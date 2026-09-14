"""Synthetic independent high-precision and negative-domain checks."""
import mpmath as mp
import numpy as np
import pytest
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.hermitian_expectation import hermitian_expectation


def reference(a, psi):
    with mp.workdps(2500):
        matrix = mp.matrix([[mp.mpc(float(z.real), float(z.imag)) for z in row] for row in a])
        vector = mp.matrix([mp.mpc(float(z.real), float(z.imag)) for z in psi])
        value = (vector.H*matrix*vector)[0]/(vector.H*vector)[0]
        assert mp.im(value) == 0
        return float(mp.re(value))


@pytest.mark.parametrize('n', [1, 2, 3, 7])
def test_random_exact_hermitian_against_independent_complex_matrix_algebra(n):
    rng = np.random.default_rng(890+n)
    x = rng.normal(size=(4, n, n))+1j*rng.normal(size=(4, n, n))
    a = x+x.swapaxes(-1, -2).conj()
    psi = rng.normal(size=(4, n))+1j*rng.normal(size=(4, n))
    before = a.copy(), psi.copy()
    actual = hermitian_expectation(a, psi)
    np.testing.assert_array_equal(actual, [reference(m, p) for m, p in zip(a, psi)])
    np.testing.assert_array_equal(a, before[0]); np.testing.assert_array_equal(psi, before[1])


@pytest.mark.parametrize('scale', [1e-300, 1e300, 1., -8., 1j])
def test_state_scale_and_phase_do_not_change_expectation(scale):
    a = np.array([[1, -1j], [1j, 3]])
    state = np.array([1, 1j])*scale
    assert hermitian_expectation(a, state) == 3.


def test_off_diagonal_imaginary_sign_and_conjugation():
    a = np.array([[0, -1j], [1j, 0]])
    assert hermitian_expectation(a, np.array([1, 1j])) == 1.
    assert hermitian_expectation(a, np.array([1, -1j])) == -1.


@pytest.mark.parametrize('value', [0., np.nextafter(0., 1.), -np.nextafter(0., 1.), 1e308])
def test_scalar_operator_extremes(value):
    assert hermitian_expectation(np.array([[value]]), np.array([1e-300])) == value


def test_exact_cancellation_survives_huge_intermediates():
    assert hermitian_expectation(np.diag([1e308, -1e308]), np.array([1e308, 1e308])) == 0.


@pytest.mark.parametrize('a,p', [
    (np.eye(2), np.zeros(2)), (np.zeros((0, 0)), np.zeros(0)),
    (np.eye(2), np.ones(3)), (np.eye(2)[None], np.ones(2)),
    (np.eye(2), np.ones((1, 2))), (np.ones(2), np.ones(2)),
    (np.array([[1j]]), np.ones(1)), (np.array([[0., 1.], [0., 0.]]), np.ones(2)),
    (np.array([[np.nan]]), np.ones(1)), (np.eye(1), np.array([np.inf])),
    (np.eye(1), np.array([True])), (np.array([[True]]), np.ones(1)),
    (np.eye(1), np.array(['1'])),
])
def test_invalid_domain_rejected(a, p):
    with pytest.raises(ValueError):
        hermitian_expectation(a, p)


def test_one_zero_batch_member_rejected():
    with pytest.raises(ValueError, match='Nonzero state'):
        hermitian_expectation(np.stack([np.eye(2)]*2), np.array([[1., 0.], [0., 0.]]))


def test_nonzero_result_underflow_rejected():
    with pytest.raises(ValueError, match='representable'):
        hermitian_expectation(np.diag([np.nextafter(0., 1.), 0.]), np.array([1., 2.]))


def test_result_overflow_rejected():
    with pytest.raises(ValueError, match='range'):
        hermitian_expectation(np.full((2, 2), 1e308), np.ones(2))
