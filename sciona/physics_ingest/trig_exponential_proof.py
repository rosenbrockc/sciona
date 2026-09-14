"""Ten-step sine/cosine exponential representation from Euler formula."""
from dataclasses import dataclass
import sympy as sp
from sciona.physics_ingest.euler_formula_proof import build_proof as build_euler,verify_proof as verify_euler
SOURCE_VERSION='4002eca6-d036-5850-a2d9-bf0818791436'
SOURCE_HASH='b4956593fcb7e8e9089b76011bcb6773290e5eddd90897bdaa7e0190bcf71854'


def eq(a,b):return sp.Eq(a,b,evaluate=False)


@dataclass(frozen=True)
class TrigExponentialProof:
    steps:tuple


def derive():
    x=sp.Symbol('x',real=True);I=sp.I
    positive=eq(sp.exp(I*x),sp.cos(x)+I*sp.sin(x))
    # Unevaluated trig arguments retain the two parity transformations separately.
    negative=eq(sp.exp(-I*x),sp.cos(-x,evaluate=False)+I*sp.sin(-x,evaluate=False))
    even=eq(negative.lhs,negative.rhs.xreplace({sp.cos(-x,evaluate=False):sp.cos(x)}))
    odd=eq(even.lhs,even.rhs.xreplace({sp.sin(-x,evaluate=False):-sp.sin(x)}))
    added=eq(positive.lhs+odd.lhs,sp.expand(positive.rhs+odd.rhs))
    divided=eq(added.lhs/2,added.rhs/2)
    cosine=eq(divided.rhs,divided.lhs)
    negated=eq(-odd.lhs,-odd.rhs)
    difference=eq(positive.lhs+negated.lhs,sp.expand(positive.rhs+negated.rhs))
    divided_i=eq(difference.lhs/(2*I),difference.rhs/(2*I))
    sine=eq(divided_i.rhs,divided_i.lhs)
    return (negative,even,odd,added,divided,cosine,negated,difference,divided_i,sine)


def build_proof():return TrigExponentialProof(derive())


def verify_proof(proof):
    prerequisite=verify_euler(build_euler())
    expected=derive()
    if len(proof.steps)!=10:raise ValueError('Ten transformations required')
    for i,(actual,wanted) in enumerate(zip(proof.steps,expected)):
        if actual!=wanted:raise ValueError(f'Transformation{i+1} differs')
    # Independent linear-system solution, treating exponentials as independent
    # symbols and not invoking a trig/exponential rewrite of the desired result.
    p,n,c,s=sp.symbols('positive negative cosine sine')
    solution=sp.solve([c+sp.I*s-p,c-sp.I*s-n],[c,s])
    if sp.expand(solution[c]-(p+n)/2)!=0 or sp.expand(solution[s]-(p-n)/(2*sp.I))!=0:
        raise ValueError('Independent linear-system check failed')
    return dict(source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,steps_verified=10,
                prerequisite_euler=prerequisite,independent_linear_system=True,
                steps=[sp.srepr(s) for s in expected],
                scope='Both cosine and sine exponential representations for finite real dimensionless angles in radians.',
                constant_semantics='i is the imaginary unit; division by2 and2i is always valid.',
                source_ast_parity=False)
