import json
import pytest
import sympy as sp
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.euler_formula import euler_formula,_checked_parse


@pytest.mark.parametrize('angle',[0,sp.pi/2,sp.pi,2*sp.pi,-3*sp.pi/2,sp.sqrt(2),sp.Rational(1,7),sp.Symbol('u',real=True),sp.Symbol('u',real=True)**2+1])
def test_general_and_specialized(angle):
    angle=sp.sympify(angle);lhs,rhs,raw=euler_formula(sp.srepr(angle));c=json.loads(raw)
    assert sp.simplify(_checked_parse(lhs)-sp.exp(sp.I*angle))==0
    assert sp.simplify(_checked_parse(rhs)-sp.cos(angle)-sp.I*sp.sin(angle))==0
    assert c['angle_srepr']==sp.srepr(angle) and c['initial_value']==1
    assert c['ode_residual']=='Integer(0)' and c['integrating_factor_derivative']=='Integer(0)'
    assert c['specialized_lhs_srepr']==lhs and c['specialized_rhs_srepr']==rhs
    assert not c['uses_complex_log']


@pytest.mark.parametrize('angle',[sp.I,sp.oo,sp.nan,sp.Symbol('unknown'),sp.Tuple(1,2)])
def test_invalid_domain(angle):
    with pytest.raises(ValueError):euler_formula(sp.srepr(angle))


@pytest.mark.parametrize('source',["Integral('1+1')","Add('1',Integer(1))","__import__('os')",''])
def test_guard(source):
    with pytest.raises(ValueError):euler_formula(source)
