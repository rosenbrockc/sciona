from dataclasses import replace
import pytest
import sympy as sp
from sciona.physics_ingest.spring_mass_proof import build_proof, verify_proof, eq


def test_certificate():
    result = verify_proof(build_proof())
    assert result['steps_verified'] == 11
    assert all(result['independent_checks'].values())


@pytest.mark.parametrize('i', range(11))
def test_corrupt_transition(i):
    proof = build_proof()
    steps = list(proof.steps)
    steps[i] = eq(0, 1)
    with pytest.raises(ValueError):
        verify_proof(replace(proof, steps=tuple(steps)))


def test_source_missing_square_root_fails_motion_equation():
    t = sp.Symbol('t', real=True)
    wrong = 3*sp.cos(4*t)  # k=8, m=2, A=3
    correct = 3*sp.cos(2*t)
    assert (2*sp.diff(wrong, t, 2)+8*wrong).subs(t, 0) == -72
    assert sp.simplify(2*sp.diff(correct, t, 2)+8*correct) == 0


def test_zero_crossing_needs_no_division():
    t = sp.Symbol('t', real=True)
    y = 3*sp.cos(2*t)
    assert y.subs(t, sp.pi/4) == 0
    assert sp.diff(y, t).subs(t, sp.pi/4) == -6
    assert sp.diff(y, t, 2).subs(t, sp.pi/4) == 0


def test_scope_released_from_rest_not_general_motion():
    result = verify_proof(build_proof())
    assert any('not the general' in s for s in result['limitations'])
