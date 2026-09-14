"""Synthetic exact-complex examples and rejected proof mutations."""
from dataclasses import replace
import pytest
import sympy as sp
from sciona.physics_ingest.eigenstate_orthogonality_proof import build_proof, verify_proof, eq


def test_seven_steps():
    report = verify_proof(build_proof())
    assert report['steps_verified'] == 7
    assert report['arbitrary_positive_dimension']


@pytest.mark.parametrize('index', range(7))
def test_corrupt_step(index):
    proof = build_proof()
    steps = list(proof.steps)
    steps[index] = eq(steps[index].lhs, steps[index].rhs + sp.Identity(1))
    with pytest.raises(ValueError):
        verify_proof(replace(proof, steps=tuple(steps)))


@pytest.mark.parametrize('index', range(3))
def test_missing_premise(index):
    proof = build_proof()
    with pytest.raises(ValueError):
        verify_proof(replace(proof, premises=proof.premises[:index]+proof.premises[index+1:]))


def test_complex_hermitian_distinct_eigenstates():
    A = sp.Matrix([[2, sp.I], [-sp.I, 2]])
    u, v = sp.Matrix([sp.I, 1]), sp.Matrix([-sp.I, 1])
    assert A.H == A
    assert A*u == 3*u and A*v == v
    assert (u.H*v)[0] == 0
    assert (u.H*A*v)[0] == 0
    # Ordinary transpose is wrong for a complex inner product.
    assert (u.T*v)[0] != 0


def test_degeneracy_does_not_imply_orthogonality():
    A = 3*sp.eye(2)
    u, v = sp.Matrix([1, 0]), sp.Matrix([1, 1])
    assert A*u == 3*u and A*v == 3*v
    assert (u.H*v)[0] == 1
    assert (3-3)*(u.H*v)[0] == 0


def test_nonhermitian_distinct_eigenvectors_counterexample():
    A = sp.Matrix([[1, 1], [0, 2]])
    u, v = sp.Matrix([1, 0]), sp.Matrix([1, 1])
    assert A.H != A
    assert A*u == u and A*v == 2*v
    assert (2-1)*(u.H*v)[0] == 1


def test_vectors_must_be_eigenvectors():
    A = sp.diag(1, 2)
    u, v = sp.Matrix([1, 0]), sp.Matrix([1, 1])
    assert A.H == A
    assert A*v != 2*v
    assert (2-1)*(u.H*v)[0] != 0
