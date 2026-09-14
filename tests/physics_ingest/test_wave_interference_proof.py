"""Synthetic interference cases and counterexamples to overly broad claims."""
import pytest
import sympy as sp
from sciona.physics_ingest.wave_interference_proof import build_proof, verify_proof


def test_product_averaging_and_bounds():
    report = verify_proof(build_proof())
    assert report['source_steps_reviewed'] == 27
    assert len(report['reconstructed_checks']) == 19
    assert all(report['reconstructed_checks'].values())
    assert report['source_ast_parity'] is False


def test_lost_conjugate_is_rejected():
    proof = build_proof()
    proof['product'] = (proof['amplitude_a']+proof['amplitude_b'])**2
    with pytest.raises(ValueError, match='reconstruction changed'):
        verify_proof(proof)


def evaluate(a_value, b_value, phase):
    proof = build_proof()
    a, b = sp.symbols('a b', nonnegative=True)
    delta = sp.Symbol('delta', real=True)
    return {key: sp.simplify(proof[key].subs({a: a_value, b: b_value, delta: phase}))
            for key in ['intensity', 'incoherent_mean', 'constructive', 'destructive', 'ratio']}


def test_coherent_is_not_synonymous_with_constructive():
    result = evaluate(3, 3, sp.pi)
    assert result['intensity'] == 0
    assert result['incoherent_mean'] == 18
    assert result['constructive'] == 36
    assert result['ratio'] == 0


def test_factor_two_requires_equal_amplitudes():
    assert evaluate(3, 3, 0)['ratio'] == 2
    assert evaluate(3, 1, 0)['ratio'] == sp.Rational(8, 5)
    assert evaluate(3, 0, 0)['ratio'] == 1


def test_incoherent_mean_does_not_require_instantaneous_zero_cosine():
    phases = [0, sp.pi]
    assert all(sp.cos(p) != 0 for p in phases)
    assert sum(evaluate(2, 3, p)['intensity'] for p in phases)/2 == 13


def test_amplitude_phase_correlation_invalidates_naive_average():
    # Equal phase weights and mean cosine zero do not suffice if amplitudes vary.
    mean_intensity = (evaluate(1, 1, 0)['intensity']+evaluate(2, 2, sp.pi)['intensity'])/2
    mean_individual_intensity = ((1**2+1**2)+(2**2+2**2))/2
    assert mean_intensity == 2
    assert mean_intensity != mean_individual_intensity


def test_complex_square_is_not_intensity():
    amplitude = 1+sp.I
    assert sp.expand(amplitude**2) == 2*sp.I
    assert sp.expand(amplitude*sp.conjugate(amplitude)) == 2
