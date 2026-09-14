"""Sine-square half-angle identity with explicit source dependency order."""
from dataclasses import dataclass
import sympy as sp
from sciona.physics_ingest.euler_formula_proof import build_proof as build_euler, verify_proof as verify_euler

SOURCE_VERSION='240e8f71-cd8a-5890-99a4-54d082f4a31d'
SOURCE_HASH='efc5154ec6ca63a04e594e04288bd27b09206629f77979128e9c40b3c63cbd7d'


def eq(a,b):return sp.Eq(a,b,evaluate=False)


@dataclass(frozen=True)
class SineSquaredProof:
    steps:tuple


def derive():
    x=sp.Symbol('x',real=True);s,c=sp.sin(x),sp.cos(x)
    doubled=eq(sp.exp(2*sp.I*x),sp.cos(2*x)+sp.I*sp.sin(2*x))
    product=eq(sp.exp(2*sp.I*x),(c+sp.I*s)**2)
    expanded=eq(product.lhs,sp.expand(product.rhs))
    comparison=eq(doubled.rhs,expanded.rhs)
    real_parts=eq(sp.re(comparison.lhs),sp.re(comparison.rhs))
    added=eq(real_parts.lhs+s**2,sp.expand(real_parts.rhs+s**2))
    pythagorean=eq(c**2,1-s**2)
    swapped=eq(added.rhs,added.lhs)
    combined=eq(swapped.rhs,pythagorean.rhs)
    subtracted=eq(sp.expand(combined.lhs-s**2),sp.expand(combined.rhs-s**2))
    add_twice=eq(subtracted.lhs+2*s**2,sp.expand(subtracted.rhs+2*s**2))
    rearranged=eq(sp.expand(add_twice.lhs-sp.cos(2*x)),add_twice.rhs-sp.cos(2*x))
    answer=eq(rearranged.lhs/2,rearranged.rhs/2)
    return (doubled,comparison,product,expanded,real_parts,added,pythagorean,
            swapped,combined,subtracted,add_twice,rearranged,answer)


def build_proof():return SineSquaredProof(derive())


def verify_proof(proof):
    prerequisite=verify_euler(build_euler())
    expected=derive()
    if len(proof.steps)!=13:raise ValueError('Thirteen steps required')
    for i,(actual,wanted) in enumerate(zip(proof.steps,expected)):
        if actual!=wanted:raise ValueError(f'Step{i+1} differs')
    x=sp.Symbol('x',real=True)
    # Independent differential certificate: both sides solve y'''+4y'=0
    # with the same three initial data. Uniqueness of the linear ODE proves
    # equality on the real line, without invoking a half-angle rewrite.
    checks=[]
    for y in (proof.steps[-1].lhs,proof.steps[-1].rhs):
        residual=sp.expand(sp.diff(y,x,3)+4*sp.diff(y,x))
        initial=[sp.diff(y,x,j).subs(x,0) for j in range(3)]
        if residual!=0 or initial!=[0,0,2]:raise ValueError('Independent differential certificate failed')
        checks.append(dict(residual=sp.srepr(residual),initial_values=[str(v) for v in initial]))
    unit_norm=sp.cos(x)**2+sp.sin(x)**2
    if sp.expand(sp.diff(unit_norm,x))!=0 or unit_norm.subs(x,0)!=1:
        raise ValueError('Pythagorean prerequisite failed')
    return dict(source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,
                steps_verified=13,steps=[sp.srepr(s) for s in expected],
                execution_order=[1,3,4,2,5,6,7,8,9,10,11,12,13],
                euler_prerequisite=prerequisite,independent_differential_certificates=checks,
                source_ast_parity=False,
                assumptions=['Finite real dimensionless angle in radians; i is exact imaginary unit.',
                             'Product rule, exponential addition, and uniqueness for a constant-coefficient linear ODE.',
                             'Pythagorean identity independently follows from zero derivative and value1 at0.'],
                limitations=['Identity does not select the sign of sin(x); no square-root branch claim.',
                             'Source step2 depends on expansion step4; execute in retained dependency order.',
                             'No claim that subtracting rounded cos(2*x) from1 is numerically stable near zeros.'])
