import pytest
import sympy as sp
from sciona.physics_ingest.pdg_graph_replay import replay_derivation
from sciona.physics_ingest.pdg_rule_contracts import RuleContract


RULES={'divide':RuleContract('divide both sides by',1,1,1)}

def step(name,source,output,divisor):
    return {'node_id':name,'type_signature':{'inference_rule_id':'divide','inputs':[source],'output':output,
            'variable_bindings':{'feeds':[{'sympy':sp.srepr(divisor)}]}}}


def test_composition_is_topological_and_carries_conditions():
    x,y,a,b=sp.symbols('x y a b')
    equations={'root':sp.Eq(a*b*x,a*b*y),'mid':sp.Eq(b*x,b*y),'out':sp.Eq(x,y)}
    result=replay_derivation([step('second','mid','out',b),step('first','root','mid',a)],equations,RULES,edges=[('first','second')])
    assert [s['node_id'] for s in result['steps']]==['first','second']
    assert set(result['required_conditions'])=={sp.srepr(sp.Ne(a,0,evaluate=False)),sp.srepr(sp.Ne(b,0,evaluate=False))}
    assert result['root_expression_ids']==['root']
    assert result['terminal_expression_ids']==['out']


def test_source_outputs_cannot_seed_a_cycle():
    x,y=sp.symbols('x y');equations={'a':sp.Eq(x,y),'b':sp.Eq(x,y)}
    with pytest.raises(ValueError,match='cycle'):
        replay_derivation([step('one','a','b',sp.Integer(1)),step('two','b','a',sp.Integer(1))],equations,RULES)


def test_duplicate_producers_and_inconsistent_edges_fail():
    x,y=sp.symbols('x y');equations={'a':sp.Eq(x,y),'b':sp.Eq(x,y),'c':sp.Eq(x,y)}
    one=step('one','a','b',sp.Integer(1))
    with pytest.raises(ValueError,match='multiple producers'):
        replay_derivation([one,step('two','a','b',sp.Integer(1))],equations,RULES)
    with pytest.raises(ValueError,match='edges'):
        replay_derivation([one,step('two','b','c',sp.Integer(1))],equations,RULES,edges=[])


def test_wrong_intermediate_conclusion_prevents_completion():
    x,y=sp.symbols('x y')
    with pytest.raises(ValueError,match='differs'):
        replay_derivation([step('one','a','b',sp.Integer(1))],{'a':sp.Eq(x,y),'b':sp.Eq(x,-y)},RULES)
