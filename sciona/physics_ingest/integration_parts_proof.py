"""Parametrized C1 reconstruction of source differential integration by parts."""
from dataclasses import dataclass
import sympy as sp
from sciona.ghost.symbolic import serialize_expr

SOURCE_VERSION='caa3b69c-a9bf-5eca-a044-269528195b10'
SOURCE_HASH='9e5293116cf314cf6edb9dc8e462cb3a146c166b39749db7b04f969dc951c7d3'


def symbols():
    x=sp.Symbol('x',real=True)
    return x,sp.Function('u')(x),sp.Function('v')(x),sp.Symbol('C')


@dataclass(frozen=True)
class IntegrationPartsProof:
    premise: sp.Equality
    steps: tuple


def build_proof():
    x,u,v,c=symbols();du=sp.Derivative(u,x);dv=sp.Derivative(v,x);product=sp.Derivative(u*v,x)
    return IntegrationPartsProof(sp.Eq(product,u*dv+v*du,evaluate=False),
        (sp.Eq(product-v*du,u*dv,evaluate=False),sp.Eq(u*dv,product-v*du,evaluate=False),
         sp.Eq(sp.Integral(u*dv,x),u*v-sp.Integral(v*du,x)+c,evaluate=False)))


def verify_proof(proof):
    expected=build_proof();x,u,v,c=symbols()
    if proof.premise!=expected.premise or len(proof.steps)!=3:
        raise ValueError('Product-rule premise and three source transitions required')
    if sp.expand(proof.premise.lhs.doit()-proof.premise.rhs)!=0:
        raise ValueError('Invalid product rule')
    first=sp.Eq(proof.premise.lhs-v*sp.Derivative(u,x),proof.premise.rhs-v*sp.Derivative(u,x),evaluate=False)
    if proof.steps[0]!=first:raise ValueError('Subtract v du from both sides')
    if proof.steps[1]!=sp.Eq(first.rhs,first.lhs,evaluate=False):raise ValueError('Reverse both sides')
    final=proof.steps[2]
    if final!=expected.steps[2]:raise ValueError('Explicit residual integral and independent constant required')
    if sp.expand(sp.diff(final.lhs,x)-sp.diff(final.rhs,x))!=0:
        raise ValueError('Integrated identity fails differentiation')
    return dict(source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,source_ast_parity=False,
                premise=serialize_expr(proof.premise),steps=[serialize_expr(e) for e in proof.steps],
                checks=dict(product_rule=True,subtraction=True,equality_reversal=True,antiderivative_modulo_constant=True),
                domain='u and v are C1 functions of a real parameter on one connected open interval; C is independent of that parameter.',
                reconstruction='Differentials interpreted as derivatives times dx. Source omitted integration constant made explicit.',
                limitations=['Remaining integral is formal and unevaluated; no numerical quadrature or closed-form existence claim.',
                             'Caller establishes a common differentiability interval; no inferred pole, branch or convergence domain.'])
