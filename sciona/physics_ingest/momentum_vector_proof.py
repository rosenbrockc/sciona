"""Explicit real-3D reconstruction of the source momentum derivation.

This proves Euclidean momentum-vector algebra under stated conservation
premises. It is not a Compton energy/angle solver or a source-AST repair.
"""
from dataclasses import dataclass
import sympy as sp
from sciona.ghost.symbolic import serialize_expr

SOURCE_VERSION = 'dbd302ec-4598-54ba-a2b4-2f828a22d66d'
SOURCE_HASH = '2bcac89ca8aed9e452f613f5605412eb0147f4cef04356c23de70a1aa9970265'
IDENTITIES = {'pdg0001302': 'before', 'pdg0005493': 'after',
              'pdg0006029': 'p1', 'pdg0002097': 'p2', 'pdg0004299': 'electron'}


def vectors():
    return {name: sp.ImmutableMatrix(sp.symbols(name+'_x '+name+'_y '+name+'_z', real=True)) for name in IDENTITIES.values()}


def equation(a, b):
    return tuple(sp.Eq(x, y, evaluate=False) for x, y in zip(a, b))


@dataclass(frozen=True)
class MomentumProof:
    premises: tuple
    steps: tuple
    norm_squared: sp.Equality


def build_proof():
    v = vectors()
    before, after, p1, p2, e = (v[k] for k in ('before', 'after', 'p1', 'p2', 'electron'))
    return MomentumProof(
        (equation(before, p1), equation(before, after), equation(after, p2+e)),
        (equation(after, p1), equation(p1, p2+e), equation(p1-p2, e), equation(e, p1-p2)),
        sp.Eq(e.dot(e), p1.dot(p1)+p2.dot(p2)-2*p1.dot(p2), evaluate=False))


def verify_proof(proof):
    canonical = build_proof()
    v = vectors()
    allowed = {symbol for vector in v.values() for symbol in vector}

    def same(actual, expected, label):
        if len(actual) != 3 or len(expected) != 3:
            raise ValueError('Three real components required')
        for a, b in zip(actual, expected):
            if not isinstance(a, sp.Equality) or a.free_symbols-allowed:
                raise ValueError('Unreviewed equation or component')
            if any(sp.expand(x-y) != 0 for x, y in zip(a.args, b.args)):
                raise ValueError('Invalid vector proof: '+label)

    if len(proof.premises) != 3 or len(proof.steps) != 4:
        raise ValueError('Expected three vector premises and four vector steps')
    for a, b in zip(proof.premises, canonical.premises):
        same(a, b, 'premise')
    # From before=p1 and before=after, replace before by p1, then
    # reverse equality. Both operations are explicit in this reconstruction.
    replace_before = {e.lhs: e.rhs for e in proof.premises[0]}
    first = tuple(sp.Eq(e.rhs, e.lhs.xreplace(replace_before), evaluate=False) for e in proof.premises[1])
    same(proof.steps[0], first, 'conservation substitution and reversal')
    replace_after = {e.lhs: e.rhs for e in proof.steps[0]}
    second = tuple(sp.Eq(e.lhs.xreplace(replace_after), e.rhs, evaluate=False) for e in proof.premises[2])
    same(proof.steps[1], second, 'outgoing substitution')
    third = tuple(sp.Eq(e.lhs-b, e.rhs-b, evaluate=False) for e, b in zip(proof.steps[1], v['p2']))
    same(proof.steps[2], third, 'subtract outgoing vector')
    fourth = tuple(sp.Eq(e.rhs, e.lhs, evaluate=False) for e in proof.steps[2])
    same(proof.steps[3], fourth, 'reverse equality')
    expected = sp.Eq(sum(e.lhs**2 for e in proof.steps[3]), sum(e.rhs**2 for e in proof.steps[3]), evaluate=False)
    if not isinstance(proof.norm_squared, sp.Equality) or proof.norm_squared.free_symbols-allowed:
        raise ValueError('Real scalar dot-product conclusion required')
    if any(sp.expand(a-b) != 0 for a, b in zip(proof.norm_squared.args, expected.args)):
        raise ValueError('Invalid dot-product expansion')
    if any(sp.expand(a-b) != 0 for a, b in zip(proof.norm_squared.args, canonical.norm_squared.args)):
        raise ValueError('Terminal norm identity differs')
    return dict(kind='explicit_real3d_momentum_reconstruction.v1', source_version_id=SOURCE_VERSION,
                source_content_hash=SOURCE_HASH, source_parity_claim=False,
                domain='Real three-dimensional Euclidean momentum components in one orthonormal frame and consistent units; three conservation premises.',
                premises=[[serialize_expr(e) for e in eqs] for eqs in proof.premises],
                vector_steps=[[serialize_expr(e) for e in eqs] for eqs in proof.steps],
                norm_squared=serialize_expr(proof.norm_squared),
                checks=dict(componentwise_composition=True, dot_product_expansion=True),
                limitations=['No energy conservation, scattering angle, wavelength shift or physical-event validity claim.',
                             'Before=p1 assumes any other initial momentum is zero in the chosen frame.',
                             'Missing source symbolic dot-product side is explicitly reconstructed, not automatically parsed or repaired.'],
                publication='Unapproved reconstructed proof; source checks and numerical realization remain required.')
