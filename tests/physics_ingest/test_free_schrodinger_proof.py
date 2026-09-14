"""Synthetic free-particle modes and counterexamples to source overclaims."""
import pytest
import sympy as sp
from sciona.physics_ingest.free_schrodinger_proof import build_proof, verify_proof, residual, symbols


def test_derivative_energy_linearity_and_continuity_checks():
    report = verify_proof(build_proof())
    assert len(report['reconstructed_checks']) == 16
    assert all(report['reconstructed_checks'].values())
    assert report['source_ast_parity'] is False


def test_missing_hbar_power_is_rejected():
    proof = build_proof()
    _, _, _, _, hbar, _ = symbols()
    proof['laplacian'] *= hbar
    with pytest.raises(ValueError, match='reconstruction changed'):
        verify_proof(proof)


def test_scalar_wavelength_source_error():
    wavelength = sp.Integer(3)
    k = 2*sp.pi/wavelength
    assert k/(2*sp.pi) == 1/wavelength
    assert k/(2*sp.pi) != wavelength


def test_arbitrary_function_is_not_a_free_solution():
    q, t, _, m, hbar, _ = symbols()
    assert sp.simplify(residual(q[0]**2, q, t, m, hbar)) == hbar**2/m


def test_wrong_dispersion_and_time_sign_fail():
    x, t = sp.symbols('x t', real=True)
    # m=2, hbar=3, p=4 requires E=4 and temporal phase -4*t/3.
    valid = sp.exp(sp.I*(4*x-4*t)/3)
    assert sp.simplify(residual(valid, (x,), t, 2, 3)) == 0
    for wrong in [sp.exp(sp.I*(4*x-5*t)/3), sp.exp(sp.I*(4*x+4*t)/3)]:
        assert sp.simplify(residual(wrong, (x,), t, 2, 3)) != 0


def test_zero_momentum_and_zero_amplitude_modes():
    proof = build_proof()
    q, t, p, m, hbar, amplitude = symbols()
    zero = proof['wave'].subs(dict.fromkeys(p, 0))
    assert zero == amplitude
    assert residual(zero, q, t, m, hbar) == 0
    ar, ai = sp.symbols('ar ai', real=True)
    assert proof['wave'].subs({ar: 0, ai: 0}) == 0


def test_constant_density_is_not_infinite_space_normalization():
    x = sp.Symbol('x', real=True)
    mode = sp.exp(sp.I*x)
    density = sp.simplify(mode*sp.conjugate(mode))
    assert density == 1
    assert sp.integrate(density, (x, -sp.oo, sp.oo)) == sp.oo
