"""Explicit corrected real quadratic proof; not a rewrite of pinned PDG data.

The two returned branches are ordered by the completed-square sign. Using
abs(a) preserves that order for negative leading coefficients. D=0 retains
two equal branch values; D<0 is outside this real-root proof's domain.
"""
from dataclasses import dataclass
import sympy as sp
from sciona.ghost.symbolic import serialize_expr

SOURCE_VERSION='40ae98f0-f033-5861-870c-6d6408edbd2b'


@dataclass(frozen=True)
class QuadraticProof:
    original: sp.Equality
    divided: sp.Equality
    shifted: sp.Equality
    completed: sp.Equality
    squared: sp.Equality
    negative: sp.Equality
    positive: sp.Equality
    lower: sp.Equality
    upper: sp.Equality


def symbols():
    return sp.Symbol('a',real=True,nonzero=True),*sp.symbols('b c x',real=True)


def build_proof():
    a,b,c,x=symbols();h=b/(2*a);q=b*b/(4*a*a)-c/a
    eq=lambda left,right:sp.Eq(left,right,evaluate=False)
    return QuadraticProof(eq(a*x*x+b*x+c,0),eq(x*x+b*x/a+c/a,0),
        eq(x*x+b*x/a,-c/a),eq(x*x+b*x/a+h*h,q),eq((x+h)**2,q),
        eq(x+h,-sp.sqrt(q)),eq(x+h,sp.sqrt(q)),
        eq(x,-h-sp.sqrt(b*b-4*a*c)/(2*sp.Abs(a))),
        eq(x,-h+sp.sqrt(b*b-4*a*c)/(2*sp.Abs(a))))


def verify_proof(proof):
    a,b,c,x=symbols();h=b/(2*a);D=b*b-4*a*c;q=D/(4*a*a)
    def same(actual,expected,label):
        if not isinstance(actual,sp.Equality) or not isinstance(expected,sp.Equality):raise ValueError('Equations required')
        if any(sp.simplify(left-right)!=0 for left,right in zip(actual.args,expected.args)):
            raise ValueError('Invalid corrected proof step: '+label)
    eq=lambda left,right:sp.Eq(left,right,evaluate=False)
    same(proof.original,eq(a*x*x+b*x+c,0),'premise')
    same(proof.divided,eq(proof.original.lhs/a,proof.original.rhs/a),'division retains b/a')
    same(proof.shifted,eq(proof.divided.lhs-c/a,proof.divided.rhs-c/a),'subtract constant')
    same(proof.completed,eq(proof.shifted.lhs+h*h,proof.shifted.rhs+h*h),'complete square')
    same(proof.squared,proof.completed,'factor square')
    same(proof.squared,eq((x+h)**2,q),'square structure')
    # The branch rule applies to a real square and a nonnegative radicand.
    # These are alternatives, never simultaneous premises of a later step.
    same(proof.negative,eq(x+h,-sp.sqrt(q)),'negative branch')
    same(proof.positive,eq(x+h,sp.sqrt(q)),'positive branch')
    # For real nonzero a, sqrt(a^2)=abs(a); no positive-a assumption.
    same(proof.lower,eq(proof.negative.lhs-h,proof.negative.rhs-h),'negative branch offset')
    same(proof.upper,eq(proof.positive.lhs-h,proof.positive.rhs-h),'positive branch offset')
    lo,hi=proof.lower.rhs,proof.upper.rhs
    for root in (lo,hi):
        if sp.simplify(proof.original.lhs.subs(x,root))!=0:raise ValueError('Root fails original polynomial')
    if sp.simplify(a*(x-lo)*(x-hi)-proof.original.lhs)!=0:raise ValueError('Root factorization incomplete')
    # Under D>=0 this gap is nonnegative, and it is zero precisely at D=0.
    if sp.simplify(hi-lo-sp.sqrt(D)/sp.Abs(a))!=0:raise ValueError('Root ordering differs')
    return dict(source_version_id=SOURCE_VERSION,kind='explicit_corrected_quadratic_proof.v1',
        source_parity_claim=False,domain=['a,b,c,x real','a != 0','b^2-4*a*c >= 0'],
        branch_semantics='Two alternative real solutions, lower then upper; repeated root appears twice.',
        steps={name:serialize_expr(getattr(proof,name)) for name in proof.__dataclass_fields__},
        corrected_steps=['retain b/a in divided equation','preserve branch sign when subtracting offset',
            'use abs(a) when simplifying sqrt(a^2), preserving root order for either sign of a'],
        checks=dict(arithmetic_transforms=True,original_polynomial_roots=True,complete_factorization=True,ordered_branches=True),
        publication='unapproved corrected proof; numerical realization and review remain required')
