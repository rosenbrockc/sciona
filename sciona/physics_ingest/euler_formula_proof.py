"""Branch-free real-angle Euler formula via normalized linear ODE."""
from dataclasses import dataclass
import sympy as sp
SOURCE_VERSION='d37f885b-673f-5a2f-9a8b-ec0d6eebee99'
SOURCE_HASH='947847810a0aa374bd6e7cac4a243242c0bd231d4bb827c6c9cc25f6ccc749bb'


def eq(a,b):return sp.Eq(a,b,evaluate=False)


@dataclass(frozen=True)
class EulerFormulaProof:
    derivative_residual:sp.Expr
    initial_value:sp.Expr
    integrating_factor_residual:sp.Expr
    conclusion:sp.Equality


def build_proof():
    x=sp.Symbol('x',real=True);y=sp.cos(x)+sp.I*sp.sin(x)
    h=sp.exp(-sp.I*x)*y
    return EulerFormulaProof(sp.expand(sp.diff(y,x)-sp.I*y),y.subs(x,0),
                             sp.expand(sp.diff(h,x)),eq(sp.exp(sp.I*x),y))


def verify_proof(proof):
    x=sp.Symbol('x',real=True);y=sp.cos(x)+sp.I*sp.sin(x)
    residual=sp.expand(sp.diff(y,x)-sp.I*y)
    if proof.derivative_residual!=residual or residual!=0:raise ValueError('ODE derivative identity failed')
    if proof.initial_value!=y.subs(x,0) or proof.initial_value!=1:raise ValueError('Normalization at zero required')
    # Product rule only, no rewriting exp into trig using the desired theorem.
    h=sp.exp(-sp.I*x)*y
    derivative=sp.expand(sp.diff(h,x))
    if proof.integrating_factor_residual!=derivative or derivative!=0:
        raise ValueError('Integrating factor product rule failed')
    if h.subs(x,0)!=1:raise ValueError('Integrating factor initial value differs')
    # Differentiable h on connected real line, h'=0 and h(0)=1 imply h=1
    # by FTC. exp(-ix) never vanishes; multiplication gives y=exp(ix).
    conclusion=eq(sp.exp(sp.I*x),y)
    if proof.conclusion!=conclusion:raise ValueError('Euler conclusion differs')
    return dict(source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,
                source_steps_reviewed=10,ode_residual=sp.srepr(residual),initial_value=1,
                integrating_factor_derivative=sp.srepr(derivative),conclusion=sp.srepr(conclusion),
                proof_method='Product rule and fundamental theorem of calculus on the connected real line.',
                source_ast_parity=False,source_log_steps_replaced=True,
                corrections=['Dependent y must be a function of x, not a constant Symbol.',
                             'Interpret i as imaginary unit and e as exponential base.',
                             'Natural logarithm required for dy/y; source base10 AST is wrong.',
                             'Indefinite integration requires a constant; y(0)=1 fixes it.',
                             'Principal complex log(exp(ix))=ix is not global; use branch-free integrating factor instead.',
                             'Source first integration is mislabeled RHS while displayed LHS changes.'],
                scope='All finite real dimensionless angles in radians; mathematical identity, not merely x=pi.')
