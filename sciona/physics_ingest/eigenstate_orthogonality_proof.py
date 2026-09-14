"""Reconstruct the source's seven Hermitian eigenstate overlap steps."""
from dataclasses import dataclass
import sympy as sp

SOURCE_VERSION = 'b46294cb-9225-5f6d-b945-1e30d4770bc8'
SOURCE_HASH = 'fc749e2154f48504917a37a9e2a03d1e43a22d03cfddd544d1764f54fef944d9'


def symbols():
    n = sp.Symbol('n', integer=True, positive=True)
    return (sp.MatrixSymbol('A', n, n), sp.MatrixSymbol('u', n, 1),
            sp.MatrixSymbol('v', n, 1), *sp.symbols('a b', real=True))


def eq(lhs, rhs):
    return sp.Eq(lhs, rhs, evaluate=False)


@dataclass(frozen=True)
class OrthogonalityProof:
    premises: tuple
    steps: tuple


def build_proof():
    A, u, v, a, b = symbols()
    overlap = sp.Adjoint(u)*v
    form = sp.Adjoint(u)*A*v
    return OrthogonalityProof(
        (eq(sp.Adjoint(A), A), eq(A*u, a*u), eq(A*v, b*v)),
        (eq(form, sp.Adjoint(u)*(b*v)),
         eq(form, (a*sp.Adjoint(u))*v),
         eq(form, b*overlap), eq(form, a*overlap),
         eq(b*overlap, a*overlap),
         eq(b*overlap-a*overlap, sp.ZeroMatrix(1, 1)),
         eq((b-a)*overlap, sp.ZeroMatrix(1, 1))))


def verify_proof(proof):
    A, u, v, a, b = symbols()
    canonical = build_proof()
    if proof.premises != canonical.premises or len(proof.steps) != 7:
        raise ValueError('Hermitian and two eigenvector premises, seven steps required')
    if any(not isinstance(step, sp.Equality) for step in proof.steps):
        raise ValueError('Unevaluated equalities required')
    # Derive the left-eigenvector equation by adjoint; real a and Hermitian A
    # are necessary here. Do not silently treat arbitrary right eigenvectors
    # as left eigenvectors.
    left = eq(sp.Adjoint(proof.premises[1].lhs).doit(),
              sp.Adjoint(proof.premises[1].rhs).doit())
    left = eq(left.lhs.xreplace({sp.Adjoint(A): A}), left.rhs)
    if left != eq(sp.Adjoint(u)*A, a*sp.Adjoint(u)):
        raise ValueError('Adjoint eigenvector derivation failed')
    form = sp.Adjoint(u)*A*v
    derived = [eq(form, sp.Adjoint(u)*proof.premises[2].rhs),
               eq(form, left.rhs*v)]
    derived += [eq(form, derived[0].rhs.doit()), eq(form, derived[1].rhs.doit())]
    derived += [eq(derived[2].rhs, derived[3].rhs)]
    derived += [eq(derived[4].lhs-derived[4].rhs, sp.ZeroMatrix(1, 1))]
    derived += [eq(sp.factor(derived[5].lhs), sp.ZeroMatrix(1, 1))]
    for index, (actual, expected) in enumerate(zip(proof.steps, derived)):
        if actual != expected:
            raise ValueError(f'Source reconstruction step {index+1} differs')
    return dict(source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH,
                steps_verified=7, arbitrary_positive_dimension=True,
                premises=[sp.srepr(p) for p in proof.premises],
                steps=[sp.srepr(p) for p in proof.steps],
                assumptions=['Finite complex vectors in an orthonormal basis; u and v nonzero eigenvectors.',
                             'A is Hermitian; its eigenvalues a and b are real.',
                             'A, a, b and the matrix element use the same arbitrary operator unit; discrete vector components are dimensionless.'],
                conclusion='(b-a) u†v = 0. Only when a != b does this imply u†v = 0.',
                source_ast_parity=False,
                limitations=['No orthogonality guarantee within a degenerate eigenspace.',
                             'No infinite-dimensional unbounded-operator domain claim.'])
