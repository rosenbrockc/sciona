"""Explicit matrix-adjoint reconstruction of the incomplete source derivation."""
from dataclasses import dataclass
import sympy as sp

SOURCE_VERSION = 'c9b2afed-ec6a-569a-9a39-e74086c8d737'
SOURCE_HASH = '55817ed94f0084b2a87173b8d51f307581b87778fcefe39b5f4e7a41a07e06ed'


def symbols():
    n = sp.Symbol('n', integer=True, positive=True)
    return sp.MatrixSymbol('A', n, n), sp.MatrixSymbol('psi', n, 1), sp.Symbol('q', complex=True)


@dataclass(frozen=True)
class HermitianProof:
    premises: tuple
    steps: tuple


def build_proof():
    a, p, q = symbols()
    form = sp.Adjoint(p)*a*p
    scalar = q*sp.Identity(1)
    return HermitianProof(
        (sp.Eq(sp.Adjoint(a), a, evaluate=False), sp.Eq(form, scalar, evaluate=False)),
        (sp.Eq(sp.Adjoint(form), sp.Adjoint(scalar), evaluate=False),
         sp.Eq(sp.Adjoint(p)*sp.Adjoint(a)*p, sp.conjugate(q)*sp.Identity(1), evaluate=False),
         sp.Eq(form, sp.conjugate(q)*sp.Identity(1), evaluate=False),
         sp.Eq(sp.conjugate(q), q, evaluate=False)))


def verify_proof(proof):
    canonical = build_proof()
    if len(proof.premises) != 2 or len(proof.steps) != 4:
        raise ValueError('Two premises and four adjoint steps required')
    if proof.premises != canonical.premises:
        raise ValueError('Explicit Hermitian and quadratic-form premises required')
    if any(not isinstance(eq, sp.Equality) for eq in proof.steps):
        raise ValueError('Unevaluated equalities required')
    definition = proof.premises[1]
    first = sp.Eq(sp.Adjoint(definition.lhs), sp.Adjoint(definition.rhs), evaluate=False)
    if proof.steps[0] != first:
        raise ValueError('Adjoint must apply to both sides')
    distributed = sp.Eq(first.lhs.doit(), first.rhs.doit(), evaluate=False)
    if proof.steps[1] != distributed:
        raise ValueError('Adjoint distribution or factor order differs')
    hermitian = proof.premises[0]
    substituted = sp.Eq(distributed.lhs.xreplace({hermitian.lhs: hermitian.rhs}), distributed.rhs, evaluate=False)
    if proof.steps[2] != substituted:
        raise ValueError('Hermitian substitution differs')
    conclusion = sp.Eq(substituted.rhs[0, 0], definition.rhs[0, 0], evaluate=False)
    if proof.steps[3] != conclusion or conclusion != canonical.steps[3]:
        raise ValueError('Scalar conjugation conclusion differs')
    return dict(source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH,
                arbitrary_positive_dimension=True, adjoint_distribution=True,
                explicit_hermitian_premise=True, scalar_realness=True,
                source_ast_parity=False,
                premises=[sp.srepr(eq) for eq in proof.premises],
                steps=[sp.srepr(eq) for eq in proof.steps],
                normalized_extension='For nonzero psi, psi†psi is positive real; dividing the real quadratic form by it remains real and equals the expectation of the normalized state.',
                limitations=['Finite-dimensional complex vectors in an orthonormal basis.',
                             'Source symbolic fields are incomplete; this is an explicit reconstruction from reviewed source equations.'])
