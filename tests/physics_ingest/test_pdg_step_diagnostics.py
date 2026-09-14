import sympy as sp
from sciona.physics_ingest.pdg_rule_contracts import RuleContract
from sciona.physics_ingest.pdg_step_diagnostics import arithmetic_step_differences


def test_dividing_quadratic_retains_linear_coefficient():
    a,b,c,x=sp.symbols('a b c x')
    rule=RuleContract('divide both sides by',1,1,1)
    premise=sp.Eq(a*x*x+b*x+c,0,evaluate=False)
    correct=sp.Eq(x*x+b*x/a+c/a,0,evaluate=False)
    incorrect=sp.Eq(x*x+x+c/a,0,evaluate=False)
    assert arithmetic_step_differences(rule,[premise],[a],correct)==(0,0)
    deltas=arithmetic_step_differences(rule,[premise],[a],incorrect)
    assert deltas[0].subs({a:2,b:3,c:1,x:-1})!=0
    assert premise.subs({a:2,b:3,c:1,x:-1})==sp.true
    assert incorrect.subs({a:2,b:3,c:1,x:-1})==sp.false


def test_subtracting_offset_preserves_root_branch():
    x,h,d=sp.symbols('x h d')
    rule=RuleContract('subtract X from both sides',1,1,1)
    premise=sp.Eq(x+h,-sp.sqrt(d),evaluate=False)
    correct=sp.Eq(x,-sp.sqrt(d)-h,evaluate=False)
    wrong=sp.Eq(x,sp.sqrt(d)-h,evaluate=False)
    assert arithmetic_step_differences(rule,[premise],[h],correct)==(0,0)
    deltas=arithmetic_step_differences(rule,[premise],[h],wrong)
    assert deltas[1].subs(d,4)==-4


def test_unsupported_rule_is_not_treated_as_passing():
    x=sp.Symbol('x');rule=RuleContract('square root both sides',1,0,2)
    assert arithmetic_step_differences(rule,[sp.Eq(x*x,1,evaluate=False)],[],sp.Eq(x,1,evaluate=False)) is None
