"""Synthetic boundary, spectral and normalization checks."""
import pytest
import sympy as sp
from sciona.physics_ingest.dirichlet_mode_proof import build_proof, verify_proof, symbols


def test_normalization_ode_and_spectral_exclusions():
    result = verify_proof(build_proof())
    assert len(result['reconstructed_checks']) == 14
    assert all(result['reconstructed_checks'].values())
    assert result['zero_eigenvalue_excluded'] and result['negative_eigenvalue_excluded']
    assert result['source_ast_parity'] is False


def test_wrong_normalization_rejected():
    proof = build_proof(); proof['wave'] *= 2
    with pytest.raises(ValueError, match='reconstruction changed'):
        verify_proof(proof)


def test_noninteger_mode_fails_right_boundary():
    x, width, n, phase = symbols()
    wave = build_proof()['wave'].subs({width: 2, n: sp.Rational(3, 2), phase: 0})
    assert wave.subs(x, 0) == 0
    assert wave.subs(x, 2) == -1


def test_zero_sine_mode_cannot_normalize():
    x, width, n, phase = symbols()
    wave = build_proof()['wave'].subs(n, 0)
    assert wave == 0
    assert sp.integrate(wave*sp.conjugate(wave), (x, 0, width)) == 0


def test_complex_phase_requires_conjugate_norm():
    x, width, n, phase = symbols()
    wave = build_proof()['wave'].subs({width: 2, n: 1, phase: sp.pi/2})
    assert sp.integrate(sp.simplify(wave*sp.conjugate(wave)), (x, 0, 2)) == 1
    assert sp.integrate(sp.simplify(wave**2), (x, 0, 2)) == -1


def test_opposite_signs_are_same_eigenspace():
    proof = build_proof()
    assert proof['opposite_wave'] == -proof['wave']
    assert sp.simplify(proof['wave']+proof['opposite_wave']) == 0


def test_distinct_modes_are_orthogonal():
    x, width, n, phase = symbols()
    wave = build_proof()['wave'].subs({width: 3, phase: sp.pi/7})
    product = sp.conjugate(wave.subs(n, 2))*wave.subs(n, 5)
    assert sp.integrate(sp.simplify(product), (x, 0, 3)) == 0
