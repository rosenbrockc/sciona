from dataclasses import replace
import pytest
import sympy as sp
from sciona.physics_ingest.euler_formula_proof import build_proof,verify_proof,eq


def test_global_ode_certificate():
    r=verify_proof(build_proof())
    assert r['initial_value']==1 and r['source_log_steps_replaced']


@pytest.mark.parametrize('field',['derivative_residual','initial_value','integrating_factor_residual'])
def test_bad_step(field):
    p=build_proof()
    with pytest.raises(ValueError):verify_proof(replace(p,**{field:getattr(p,field)+1}))


def test_bad_conclusion():
    p=build_proof()
    with pytest.raises(ValueError):verify_proof(replace(p,conclusion=eq(p.conclusion.lhs,p.conclusion.rhs+1)))


def test_missing_constant_counterexample():
    x=sp.Symbol('x',real=True);y=2*sp.exp(sp.I*x)
    assert sp.diff(y,x)-sp.I*y==0
    assert y.subs(x,0)!=1


def test_principal_log_not_global():
    x=2*sp.pi
    assert sp.log(sp.exp(sp.I*x))==0
    assert sp.log(sp.exp(sp.I*x))!=sp.I*x


def test_base10_log_wrong_derivative():
    y=sp.Symbol('y',positive=True)
    assert sp.diff(sp.log(y,10),y)!=1/y
    assert sp.diff(sp.log(y),y)==1/y


@pytest.mark.parametrize('x',[0,sp.pi/2,sp.pi,2*sp.pi,-sp.pi/2])
def test_exact_angles(x):
    assert sp.simplify(sp.exp(sp.I*x)-sp.cos(x)-sp.I*sp.sin(x))==0
