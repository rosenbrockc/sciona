import pytest
import sympy as sp
from sciona.physics_ingest.pdg_rule_contracts import RuleContract
from sciona.physics_ingest.pdg_real_power_rules import replay_real_power_step

SQUARE=RuleContract('raise both sides to power',1,1,1)


def test_square_retains_radicand_and_denominator_conditions():
    v,g,m,r=sp.symbols('v g m r')
    premise=sp.Eq(v,sp.sqrt(2)*sp.sqrt(g*m/r),evaluate=False)
    conclusion=sp.Eq(v**2,2*g*m/r,evaluate=False)
    actual,bounds=replay_real_power_step(SQUARE,[premise],[sp.Integer(2)],conclusion)
    assert actual==conclusion
    assert sp.Ge(g*m/r,0,evaluate=False) in bounds
    assert sp.Ne(r,0,evaluate=False) in bounds


def test_forward_square_does_not_permit_reverse_branch_selection():
    x,y=sp.symbols('x y')
    with pytest.raises(ValueError,match='forward squaring'):
        replay_real_power_step(SQUARE,[sp.Eq(x**2,y,evaluate=False)],[sp.Rational(1,2)],sp.Eq(x,sp.sqrt(y),evaluate=False))


@pytest.mark.parametrize('radicand',[sp.Integer(-1),sp.sin(sp.Symbol('x')),sp.sqrt(sp.Symbol('x'))])
def test_invalid_or_unsupported_radicands_fail(radicand):
    x=sp.Symbol('x')
    premise=sp.Eq(x,sp.Pow(radicand,sp.Rational(1,2),evaluate=False),evaluate=False)
    with pytest.raises(ValueError):
        replay_real_power_step(SQUARE,[premise],[sp.Integer(2)],sp.Eq(x*x,radicand,evaluate=False))


def test_wrong_conclusion_rejected():
    x,y=sp.symbols('x y')
    with pytest.raises(ValueError,match='differs'):
        replay_real_power_step(SQUARE,[sp.Eq(x,sp.sqrt(y),evaluate=False)],[sp.Integer(2)],sp.Eq(x*x,-y,evaluate=False))


def test_rational_operations_remain_delegated():
    x,y,z=sp.symbols('x y z')
    rule=RuleContract('divide both sides by',1,1,1)
    actual,bounds=replay_real_power_step(rule,[sp.Eq(x,y,evaluate=False)],[z],sp.Eq(x/z,y/z,evaluate=False))
    assert actual==sp.Eq(x/z,y/z,evaluate=False)
    assert sp.Ne(z,0,evaluate=False) in bounds


def test_composed_squaring_retains_renamed_domain():
    from sciona.physics_ingest.pdg_graph_replay_v3 import replay_derivation
    from sciona.ghost.symbolic import serialize_expr
    v,g,m,r,c,R=sp.symbols('v g m r c R')
    expressions={
        'root':sp.Eq(v,sp.sqrt(2)*sp.sqrt(g*m/r),evaluate=False),
        'squared':sp.Eq(v**2,2*g*m/r,evaluate=False),
        'renamed':sp.Eq(c**2,2*g*m/R,evaluate=False),
    }
    nodes=[
        dict(node_id='square',type_signature=dict(inference_rule_id='square',inputs=['root'],output='squared',variable_bindings=dict(feeds=[dict(sympy='Integer(2)')]))),
        dict(node_id='rename',type_signature=dict(inference_rule_id='rename',inputs=['squared'],output='renamed',variable_bindings=dict(feeds=[dict(sympy=serialize_expr(s)) for s in [v,c,r,R]]))),
    ]
    rules={'square':SQUARE,'rename':RuleContract('change two variables in expr',1,4,1)}
    result=replay_derivation(nodes,expressions,rules,edges=[('square','rename')])
    bounds=result['steps'][-1]['required_conditions']
    assert serialize_expr(sp.Ge(g*m/r,0,evaluate=False)) in bounds
    assert serialize_expr(sp.Ge(g*m/R,0,evaluate=False)) in bounds
    assert serialize_expr(sp.Ne(R,0,evaluate=False)) in bounds
