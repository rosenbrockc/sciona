import sympy as sp
import pytest
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.integration_by_parts import integration_by_parts
from sciona.physics_ingest.source_symbolic import parse_source_srepr

x=sp.Symbol('x',real=True)
C=sp.Symbol('C')


@pytest.mark.parametrize('u,v',[(x,sp.exp(x)),(x*x,sp.sin(x)),(sp.log(x),x),
    (sp.Function('u')(x),sp.Function('v')(x)),(sp.Integer(3),x*x),(x,sp.Integer(0)),
    (sp.exp(-x*x),sp.sin(x)),(1/x,sp.exp(x)),(sp.I*x,sp.cos(x)),
    (sp.Function('f')(x*x),sp.Function('g')(sp.sin(x)))])
def test_verified_rewrite_retains_residual_integral(u,v):
    original,rewritten=integration_by_parts(*map(sp.srepr,[u,v,x,C]))
    integrand,primitive=map(parse_source_srepr,[original,rewritten])
    assert sp.expand(integrand.doit()-u*sp.diff(v,x))==0
    assert primitive.has(sp.Integral)
    assert sp.expand((sp.diff(primitive,x)-integrand).doit())==0
    assert primitive.has(C)


def test_known_closed_form_after_separate_residual_evaluation():
    _,rewritten=integration_by_parts(*map(sp.srepr,[x,sp.exp(x),x,C]))
    primitive=parse_source_srepr(rewritten)
    assert sp.simplify(primitive.doit()-((x-1)*sp.exp(x)+C))==0


def test_variable_dependent_constant_rejected():
    with pytest.raises(ValueError,match='constant'):
        integration_by_parts(*map(sp.srepr,[x,sp.exp(x),x,x]))


def test_bound_dummy_variable_in_constant_is_not_free_dependence():
    constant=sp.Integral(sp.sin(x),(x,0,1))
    _,result=integration_by_parts(*map(sp.srepr,[x,sp.exp(x),x,constant]))
    primitive=parse_source_srepr(result)
    assert sp.simplify(primitive.doit()-((x-1)*sp.exp(x)+1-sp.cos(1)))==0


@pytest.mark.parametrize('variable',[sp.Symbol('x'),sp.Symbol('z',imaginary=True),x+1,sp.Integer(1)])
def test_non_real_symbol_variable_rejected(variable):
    with pytest.raises(ValueError):integration_by_parts(*map(sp.srepr,[x,x,variable,C]))


def test_conflicting_symbol_assumptions_rejected():
    with pytest.raises(ValueError,match='assumptions'):
        integration_by_parts(*map(sp.srepr,[sp.Symbol('x'),x,x,C]))


@pytest.mark.parametrize('bad',["__import__('os').getcwd()","Symbol('x').__class__","[Symbol('x')]","lambda: 1","open('anything')"])
def test_non_expression_code_rejected(bad):
    with pytest.raises(ValueError):integration_by_parts(bad,sp.srepr(x),sp.srepr(x),sp.srepr(C))


@pytest.mark.parametrize('bad',[
    "Integral('Symbol(\"z\")', Symbol('x', real=True))",
    "Derivative('Symbol(\"z\")', Symbol('x', real=True))",
    "Add('Symbol(\"z\")', Integer(1))",
    "Function('f')('Symbol(\"z\")')",
    "Float('Symbol(\"z\")')",
])
def test_constructor_expression_strings_rejected_before_sympification(bad):
    with pytest.raises(ValueError,match='Expression strings'):
        integration_by_parts(bad,sp.srepr(x),sp.srepr(x),sp.srepr(C))


def test_numeric_float_literals_still_supported():
    original,_=integration_by_parts(*map(sp.srepr,[sp.Float('1.25'),x,x,C]))
    assert sp.simplify(parse_source_srepr(original).doit()-sp.Float('1.25'))==0


@pytest.mark.parametrize('bad',[sp.oo,sp.nan,sp.zoo,sp.Eq(x,1,evaluate=False),sp.Symbol('A',commutative=False)])
def test_invalid_scalar_expression_rejected(bad):
    with pytest.raises(ValueError):integration_by_parts(*map(sp.srepr,[bad,x,x,C]))
