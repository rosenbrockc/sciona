"""Explicit circular two-body algebra, retaining source discrepancies.

The dimensionless p is a parameter here, not an implicit interpretation as pi.
Positive real parameters bound this corrected derivation. Physical premises and
their applicability require separate review before an execution promotion.
"""
from dataclasses import dataclass

import sympy as sp

from sciona.ghost.symbolic import serialize_expr


SOURCE_VERSION = 'ffb043f3-05a3-5ca6-9d94-87a17ff512d3'
SOURCE_SYMBOL_NAMES = {
    'pdg0001687': 'Fc', 'pdg0002867': 'F', 'pdg0002321': 'w',
    'pdg0002530': 'r', 'pdg0002798': 'd2', 'pdg0007652': 'd1',
    'pdg0005156': 'm', 'pdg0004851': 'm2', 'pdg0005022': 'm1',
    'pdg0006277': 'G', 'pdg0003141': 'p', 'pdg0009491': 'T',
}


def symbols():
    return {name: sp.Symbol(name, positive=True) for name in SOURCE_SYMBOL_NAMES.values()}


@dataclass(frozen=True)
class TwoBodyProof:
    premises: tuple
    steps: tuple


def build_proof():
    s = symbols()
    F, Fc, w, r, d1, d2, m, m1, m2, G, p, T = (s[k] for k in ('F', 'Fc', 'w', 'r', 'd1', 'd2', 'm', 'm1', 'm2', 'G', 'p', 'T'))
    eq = lambda a, b: sp.Eq(a, b, evaluate=False)
    premises = (eq(Fc, m*w**2*r), eq(F, Fc), eq(F, G*m1*m2/r**2),
                eq(w, 2*p/T), eq(r, d1+d2), eq(m1*d1, m2*d2))
    q = 4*p**2*r**2*d2/(G*m1)
    rewritten = 4*p**2*r**3*d2/(G*m1*(d1+d2))
    steps = (
        eq(Fc, m2*w**2*d2), eq(Fc, G*m1*m2/r**2),
        eq(m2*w**2*d2, G*m1*m2/r**2),
        eq(4*p**2*m2*d2/T**2, G*m1*m2/r**2),
        eq(4*p**2*d2/T**2, G*m1/r**2),
        eq(1/T**2, G*m1/(4*p**2*r**2*d2)), eq(T**2, q),
        eq(T**2, q), eq(T**2, rewritten), eq(T**2, rewritten),
        eq(T**2, 4*p**2*r**3/(G*(m1 + m1*d1/d2))),
        eq(m1*d1/d2, m2), eq(T**2, 4*p**2*r**3/(G*(m1+m2))),
    )
    return TwoBodyProof(premises, steps)


def verify_proof(proof):
    s = symbols()
    m, m2, r, d1, d2, p = (s[k] for k in ('m', 'm2', 'r', 'd1', 'd2', 'p'))
    canonical = build_proof()
    if len(proof.premises) != 6 or len(proof.steps) != 13:
        raise ValueError('Expected six premises and thirteen steps')

    def same(actual, expected, label, mapping=None):
        if not isinstance(actual, sp.Equality) or not isinstance(expected, sp.Equality):
            raise ValueError('Equations required: '+label)
        for a, b in zip(actual.args, expected.args):
            if mapping:
                a, b = a.xreplace(mapping), b.xreplace(mapping)
            if sp.cancel(a-b) != 0:
                raise ValueError('Invalid corrected proof: '+label)

    for i, (actual, expected) in enumerate(zip(proof.premises, canonical.premises)):
        same(actual, expected, 'premise '+str(i))
    eq = lambda a, b: sp.Eq(a, b, evaluate=False)
    premises, steps = proof.premises, proof.steps
    same(steps[0], premises[0].xreplace({m: m2, r: d2}), 'rename')
    same(steps[1], premises[2].xreplace({premises[1].lhs: premises[1].rhs}), 'force equality')
    same(steps[2], steps[1].xreplace({steps[0].lhs: steps[0].rhs}), 'centripetal substitution')
    same(steps[3], steps[2].xreplace({premises[3].lhs: premises[3].rhs}), 'frequency substitution')
    same(steps[4], eq(steps[3].lhs/m2, steps[3].rhs/m2), 'explicit mass division')
    factor = 4*p**2*d2
    same(steps[5], eq(steps[4].lhs/factor, steps[4].rhs/factor), 'normalization')
    same(steps[6], eq(1/steps[5].lhs, 1/steps[5].rhs), 'reciprocal')
    same(steps[7], steps[6], 'unity with d1+d2 nonzero')
    # The canceled unity feed makes a structural replacement unavailable.
    # Equality of rational expressions after r -> d1+d2 proves the rewrite
    # on the explicit premise r=d1+d2; positivity excludes its denominators.
    same(steps[8], steps[7], 'separation premise rewrite', {r: d1+d2})
    same(steps[9], steps[8], 'unity with d2 nonzero')
    same(steps[10], steps[9], 'rational simplification')
    same(steps[11], eq(premises[5].lhs/d2, premises[5].rhs/d2), 'center of mass division')
    same(steps[12], steps[10].subs(steps[11].lhs, steps[11].rhs), 'mass ratio substitution')
    same(steps[12], canonical.steps[12], 'terminal formula')
    # Ensure the claimed domain suffices for every denominator, including a
    # modified intermediate that introduces an otherwise hidden singularity.
    for expression in (*proof.premises, *steps):
        if not expression.free_symbols.issubset(set(s.values())):
            raise ValueError('Unreviewed symbol or assumptions')
        for node in sp.preorder_traversal(expression):
            if isinstance(node, sp.Pow) and node.exp.is_negative:
                if node.base.is_positive is not True:
                    raise ValueError('Denominator positivity is not established')
    return dict(kind='explicit_corrected_two_body_proof.v1', source_version_id=SOURCE_VERSION,
        source_parity_claim=False, domain='All named symbols positive real; stated premise schema and five equations required.',
        premise_schema='Premise 0 is the centripetal-force schema instantiated with mass m2 and orbital radius d2; it is not inferred by renaming a fixed-object fact.',
        pi_interpretation=False,
        premises=[serialize_expr(e) for e in proof.premises],
        steps=[serialize_expr(e) for e in steps],
        corrections=['Insert RHS at steps 3, 4, 13', 'Explicitly divide by positive m2 at step 5',
                     'Use separation premise for rational rewrite at step 9'],
        checks=dict(composed_algebra=True, denominator_domain=True, terminal_formula=True),
        publication='Unapproved corrected proof; physical review and runtime realization remain required.')
