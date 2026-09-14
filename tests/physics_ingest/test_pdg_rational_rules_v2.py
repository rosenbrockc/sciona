import pytest
import sympy as sp
from sciona.physics_ingest.pdg_rule_contracts import RuleContract
from sciona.physics_ingest.pdg_rational_rules_v2 import replay_rational_step


def test_multiplication_and_division_are_not_confused():
    x,y,k=sp.symbols('x y k');rule=RuleContract('multiply both sides by',1,1,1)
    _,conditions=replay_rational_step(rule,[sp.Eq(x,y)],[k],sp.Eq(k*x,k*y,evaluate=False))
    assert not conditions
    with pytest.raises(ValueError,match='differs'):
        replay_rational_step(rule,[sp.Eq(x,y)],[k],sp.Eq(x/k,y/k))


def test_simplification_retains_removed_denominator_domain():
    x,y=sp.symbols('x y')
    quotient=sp.Mul(x,sp.Pow(x,-1,evaluate=False),evaluate=False)
    _,conditions=replay_rational_step(RuleContract('simplify',1,0,1),[sp.Eq(quotient,y,evaluate=False)],[],sp.Eq(1,y))
    assert sp.Ne(x,0,evaluate=False) in conditions


def test_power_requires_integer_and_preserves_negative_power_domain():
    x,y=sp.symbols('x y');rule=RuleContract('raise both sides to power',1,1,1)
    _,conditions=replay_rational_step(rule,[sp.Eq(x,y)],[sp.Integer(-1)],sp.Eq(1/x,1/y))
    assert sp.Ne(x,0,evaluate=False) in conditions and sp.Ne(y,0,evaluate=False) in conditions
    with pytest.raises(ValueError):replay_rational_step(rule,[sp.Eq(x,y)],[sp.Rational(1,2)],sp.Eq(sp.sqrt(x),sp.sqrt(y)))


def test_left_side_transitivity_requires_equal_right_sides():
    a,b,c,d=sp.symbols('a b c d');rule=RuleContract('LHS of expr 1 equals LHS of expr 2',2,0,1)
    replay_rational_step(rule,[sp.Eq(a,c),sp.Eq(b,c)],[],sp.Eq(a,b))
    with pytest.raises(ValueError,match='right sides'):
        replay_rational_step(rule,[sp.Eq(a,c),sp.Eq(b,d)],[],sp.Eq(a,b))


def test_v2_graph_composes_new_operations_and_rejects_wrong_intermediate():
    from sciona.physics_ingest.pdg_graph_replay_v2 import replay_derivation
    x,a,b,c=sp.symbols('x a b c')
    rules={'mul':RuleContract('multiply both sides by',1,1,1),'div':RuleContract('divide both sides by',1,1,1)}
    nodes=[{'node_id':'multiply','type_signature':{'inference_rule_id':'mul','inputs':['root'],'output':'mid','variable_bindings':{'feeds':[{'sympy':sp.srepr(c)}]}}},
           {'node_id':'divide','type_signature':{'inference_rule_id':'div','inputs':['mid'],'output':'out','variable_bindings':{'feeds':[{'sympy':sp.srepr(a)}]}}}]
    expressions={'root':sp.Eq(x,a*b),'mid':sp.Eq(x*c,a*b*c),'out':sp.Eq(x*c/a,b*c)}
    result=replay_derivation(list(reversed(nodes)),expressions,rules,edges=[('multiply','divide')])
    assert result['terminal_expression_ids']==['out']
    assert sp.srepr(sp.Ne(a,0,evaluate=False)) in result['required_conditions']
    with pytest.raises(ValueError,match='differs'):
        replay_derivation(nodes,{**expressions,'mid':sp.Eq(x*c,a*b*c+1)},rules)
