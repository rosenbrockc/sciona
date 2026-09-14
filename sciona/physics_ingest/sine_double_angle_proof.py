"""Six-step source proof of the real sine double-angle identity."""
from dataclasses import dataclass
import sympy as sp
from sciona.ghost.symbolic import serialize_expr

SOURCE_VERSION='f999aa2b-2d62-57b0-8e7d-277dac1ae7a7'
SOURCE_HASH='46bf59c366a899e8b3c9d52c60661130d08a917fd7f0659d25c33151d593b50f'


@dataclass(frozen=True)
class DoubleAngleProof:
    premises: tuple
    steps: tuple


def build_proof():
    x=sp.Symbol('theta',real=True)
    a,b=sp.exp(sp.I*x),sp.exp(-sp.I*x)
    sine=sp.Eq(sp.sin(x),(a-b)/(2*sp.I),evaluate=False)
    cosine=sp.Eq(sp.cos(x),(a+b)/2,evaluate=False)
    doubled=sp.Eq(sp.sin(2*x),sine.rhs.subs(x,2*x),evaluate=False)
    product=sp.Eq(sine.lhs*cosine.lhs,sine.rhs*cosine.rhs,evaluate=False)
    scaled=sp.Eq(2*product.lhs,2*product.rhs,evaluate=False)
    expanded=sp.Eq(scaled.lhs,sp.expand(scaled.rhs),evaluate=False)
    simplified=sp.Eq(scaled.lhs,(sp.exp(2*sp.I*x)-sp.exp(-2*sp.I*x))/(2*sp.I),evaluate=False)
    endpoint=sp.Eq(doubled.lhs,simplified.lhs,evaluate=False)
    return DoubleAngleProof((sine,cosine),(doubled,product,scaled,expanded,simplified,endpoint))


def equal_sides(a,b):
    return isinstance(a,sp.Equality) and isinstance(b,sp.Equality) and all(
        sp.simplify(x-y)==0 for x,y in zip(a.args,b.args))


def verify_proof(proof):
    expected=build_proof()
    if proof.premises != expected.premises or len(proof.steps)!=6:
        raise ValueError('Two exponential premises and six transitions required')
    for premise in proof.premises:
        if sp.simplify(premise.lhs.rewrite(sp.exp)-premise.rhs)!=0:
            raise ValueError('Exponential premise not established')
    for i,(actual,wanted) in enumerate(zip(proof.steps,expected.steps)):
        if not equal_sides(actual,wanted):
            raise ValueError('Source transition differs: '+str(i+1))
    if sp.simplify(proof.steps[0].rhs-proof.steps[4].rhs)!=0:
        raise ValueError('Common exponential RHS differs')
    if sp.expand_trig(proof.steps[-1].lhs)-proof.steps[-1].rhs!=0:
        raise ValueError('Independent angle-addition identity differs')
    return dict(source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,
                source_ast_parity=False,steps=[serialize_expr(e) for e in proof.steps],
                checks=dict(exponential_premises=True,six_transitions=True,common_rhs=True,angle_addition_crosscheck=True),
                domain='Every real dimensionless angle theta in radians; imaginary-unit source identity explicitly interpreted as I.')


def verify_source_equations(records,mapping):
    # These strings come only from the hash-pinned public source, not runtime inputs.
    from sciona.physics_ingest.source_symbolic import parse_source_srepr
    interpreted={}
    for identity,record in records.items():
        sides=[parse_source_srepr(record[k]) for k in ['sympy_lhs','sympy_rhs']]
        if any(s.free_symbols-set(mapping) for s in sides):
            raise ValueError('Unreviewed source symbol')
        interpreted[identity]=sp.Eq(*(s.xreplace(mapping) for s in sides),evaluate=False)
    expected=build_proof()
    for identity,wanted in zip(['2103023049','4585932229'],expected.premises):
        if not equal_sides(interpreted[identity],wanted):
            raise ValueError('Interpreted source premise differs')
    order=['8483686863','3470587782','9894826550','8699789241','9180861128','2405307372']
    proof=DoubleAngleProof(tuple(interpreted[k] for k in ['2103023049','4585932229']),tuple(interpreted[k] for k in order))
    # Canonical premises may differ in harmless expression construction.
    result=verify_proof(DoubleAngleProof(expected.premises,proof.steps))
    return dict(interpreted_source_equations=len(interpreted),all_source_steps_verified=True,
                literal_source_parity=False,proof=result)
