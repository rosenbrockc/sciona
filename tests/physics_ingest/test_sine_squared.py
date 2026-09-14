import json
import pytest
import sympy as sp
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.sine_squared import sine_squared,_checked_parse
from sciona.physics_ingest.sine_squared_proof import build_proof


@pytest.mark.parametrize('angle',[0,sp.pi/2,sp.pi,-sp.pi/3,sp.sqrt(2),sp.Rational(1,10)**100,
    sp.Symbol('u',real=True),sp.Symbol('u',real=True)**2+sp.Rational(1,3)])
def test_identity_and_certificate(angle):
    angle=sp.sympify(angle)
    lhs,rhs,raw=sine_squared(sp.srepr(angle));c=json.loads(raw)
    assert sp.simplify(_checked_parse(lhs)-sp.sin(angle)**2)==0
    assert sp.simplify(_checked_parse(rhs)-(1-sp.cos(2*angle))/2)==0
    assert c['steps']==[sp.srepr(s) for s in build_proof().steps]
    assert c['specialized_lhs_srepr']==lhs and c['specialized_rhs_srepr']==rhs
    assert c['angle_srepr']==sp.srepr(angle) and len(c['differential_certificates'])==2
    assert all(d==dict(residual='Integer(0)',initial_values=['0','0','2']) for d in c['differential_certificates'])
    assert c['execution_order'][:4]==[1,3,4,2]
    assert c['euler_ode_residual']==c['integrating_factor_derivative']=='Integer(0)'
    assert not c['source_ast_parity'] and not c['sine_sign_inferred']


@pytest.mark.parametrize('angle',[sp.I,sp.oo,sp.nan,sp.Symbol('unknown'),sp.Tuple(1,2)])
def test_invalid_domain(angle):
    with pytest.raises(ValueError):sine_squared(sp.srepr(angle))


@pytest.mark.parametrize('source',["Integral('1+1')","Add('1',Integer(1))","__import__('os')",''])
def test_guard(source):
    with pytest.raises(ValueError):sine_squared(source)


def test_symbol_identity_conflict():
    a=sp.Symbol('a',positive=True);b=sp.Symbol('a',negative=True)
    with pytest.raises(ValueError):sine_squared(sp.srepr(a+b))
