"""Synthetic symbolic checks for the reviewed constant-force energy model."""
import copy
import pytest
import sympy as sp
from sciona.physics_ingest.mechanical_energy_proof import build_proof, verify_proof


def test_reconstructed_steps_and_independent_integrals():
    report = verify_proof(build_proof())
    assert report['reconstructed_steps_validated'] == 23
    assert len(report['integration_checks']) == 12
    assert all(report['integration_checks'].values())
    assert report['source_ast_parity'] is False


def test_altered_conservation_step_is_rejected():
    step = 22
    proof = copy.deepcopy(build_proof())
    equation = proof['steps'][step]
    proof['steps'][step] = sp.Eq(equation.lhs, equation.rhs+1, evaluate=False)
    with pytest.raises(ValueError, match='reconstruction changed'):
        verify_proof(proof)


def test_average_velocity_is_not_final_velocity():
    t = sp.Symbol('t', real=True)
    velocity = 2+3*t
    displacement = sp.integrate(velocity, (t, 0, 2))
    assert displacement == 10
    assert displacement/2 == (velocity.subs(t, 0)+velocity.subs(t, 2))/2
    assert displacement/2 != velocity.subs(t, 2)


def test_endpoint_mean_fails_for_variable_acceleration():
    t = sp.Symbol('t', real=True)
    velocity = t**2
    assert sp.integrate(velocity, (t, 0, 1)) != (velocity.subs(t, 0)+velocity.subs(t, 1))/2


def test_zero_duration_and_reversal_energy():
    proof = build_proof()
    symbols = {str(s): s for s in proof['position'].free_symbols | proof['velocity'].free_symbols}
    values = {symbols[k]: v for k, v in dict(m=2, F=-4, x1=3, v1=2).items()}
    t = symbols['t']
    X, V = proof['position'].subs(values), proof['velocity'].subs(values)
    assert V.subs(t, 0) == 2 and X.subs(t, 0) == 3
    assert V.subs(t, 2) == -2
    assert sp.expand(V**2+4*X) == 16


def test_time_dependent_potential_is_outside_conservation_scope():
    t = sp.Symbol('t', real=True)
    x = sp.Symbol('x', real=True)
    potential = -4*x+t
    trajectory = 3+2*t+t**2
    velocity = sp.diff(trajectory, t)
    assert -sp.diff(potential, x) == 4
    assert 2*sp.diff(velocity, t) == 4
    assert sp.diff(velocity**2+potential.subs(x, trajectory), t) == 1
