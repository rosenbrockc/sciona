from dataclasses import replace
import pytest
import sympy as sp
from sciona.physics_ingest.hermitian_expectation_proof import build_proof, verify_proof, symbols


def test_arbitrary_dimension_adjoint_proof():
    report = verify_proof(build_proof())
    assert report['arbitrary_positive_dimension'] and report['scalar_realness']
    assert symbols()[2].is_real is None


@pytest.mark.parametrize('i', range(4))
def test_corrupted_step_rejected(i):
    proof = build_proof()
    steps = list(proof.steps)
    steps[i] = sp.Eq(sp.Symbol('unreviewed'), 0, evaluate=False)
    with pytest.raises(ValueError):
        verify_proof(replace(proof, steps=tuple(steps)))


def test_missing_hermiticity_rejected():
    proof = build_proof()
    with pytest.raises(ValueError):
        verify_proof(replace(proof, premises=proof.premises[1:]))


def test_transpose_is_not_adjoint():
    proof = build_proof(); a, p, q = symbols()
    steps = list(proof.steps)
    steps[1] = sp.Eq(p.T*a.T*p, q*sp.Identity(1), evaluate=False)
    with pytest.raises(ValueError):
        verify_proof(replace(proof, steps=tuple(steps)))
