import hashlib
import pytest
import sympy as sp
from sciona.physics_ingest.pdg_rule_contracts import RuleContract,load_pinned_rule_contracts,contract_blockers,replay_rational_step


def test_pinned_rule_and_arity_validation():
    data=b'UNWIND [{id:"1", properties:{name_latex:"synthetic",author_name_latex:"synthetic-author",number_of_inputs:4,number_of_feeds:0,number_of_outputs:1}}] AS row\nCREATE (n:inference_rule{id: row.id}) SET n += row.properties;'
    rule=load_pinned_rule_contracts(data,hashlib.sha256(data).hexdigest())['1']
    assert rule.name == 'synthetic'
    assert contract_blockers({'inputs':['a'],'output':'b'},rule)==['inputs_arity_mismatch']
    with pytest.raises(ValueError,match='pin'):load_pinned_rule_contracts(data,'wrong')


def test_renaming_is_simultaneous():
    x,y,z=sp.symbols('x y z')
    actual,conditions=replay_rational_step(RuleContract('change two variables in expr',1,4,1),[sp.Eq(x,y,evaluate=False)],[x,y,y,z],sp.Eq(y,z,evaluate=False))
    assert actual==sp.Eq(y,z,evaluate=False)
    assert conditions==()


def test_substitution_uses_all_premises():
    a,b,c,x,y,z=sp.symbols('a b c x y z')
    rule=RuleContract('substitute LHS of three expressions into expr',4,0,1)
    premises=[sp.Eq(a,x),sp.Eq(b,y),sp.Eq(c,z),sp.Eq(a,b+c)]
    replay_rational_step(rule,premises,[],sp.Eq(x,y+z))
    with pytest.raises(ValueError,match='arity'):replay_rational_step(rule,premises[:1],[],sp.Eq(x,y+z))
    with pytest.raises(ValueError,match='differs'):replay_rational_step(rule,premises,[],sp.Eq(x,y-z))


def test_division_preserves_nonzero_requirement():
    x,y,k=sp.symbols('x y k')
    rule=RuleContract('divide both sides by',1,1,1)
    _,conditions=replay_rational_step(rule,[sp.Eq(k*x,k*y)],[k],sp.Eq(x,y))
    assert sp.Ne(k,0,evaluate=False) in conditions
    with pytest.raises(ValueError,match='division'):replay_rational_step(rule,[sp.Eq(x,y)],[sp.Integer(0)],sp.Eq(x,y))


def test_branch_sensitive_functions_are_not_approved_as_rational_algebra():
    x,y=sp.symbols('x y')
    with pytest.raises(ValueError,match='unsupported'):
        replay_rational_step(RuleContract('divide both sides by',1,1,1),[sp.Eq(sp.sqrt(x),y)],[sp.Integer(1)],sp.Eq(sp.sqrt(x),y))
