import json
import pytest
import sympy as sp
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.trig_exponential import trig_exponential,_checked_parse


@pytest.mark.parametrize('angle',[0,sp.pi/2,sp.pi,2*sp.pi,-sp.pi/2,sp.sqrt(2),sp.Rational(1,7),sp.Symbol('u',real=True),sp.Symbol('u',real=True)**2+1])
def test_both_outputs_and_certificate(angle):
    angle=sp.sympify(angle);a,b,raw=trig_exponential(sp.srepr(angle));c=json.loads(raw)
    assert sp.simplify(_checked_parse(a).rewrite(sp.cos)-sp.cos(angle))==0
    assert sp.simplify(_checked_parse(b).rewrite(sp.sin)-sp.sin(angle))==0
    assert c['angle_srepr']==sp.srepr(angle) and c['initial_value']==1
    assert c['euler_ode_residual']=='Integer(0)' and c['integrating_factor_derivative']=='Integer(0)'
    assert c['linear_system_residuals']==['Integer(0)','Integer(0)']
    assert c['cosine_exponential_srepr']==a and c['sine_exponential_srepr']==b
    assert not c['uses_complex_log']


@pytest.mark.parametrize('angle',[sp.I,sp.oo,sp.nan,sp.Symbol('unknown'),sp.Tuple(1,2)])
def test_invalid_domain(angle):
    with pytest.raises(ValueError):trig_exponential(sp.srepr(angle))


@pytest.mark.parametrize('source',["Integral('1+1')","Add('1',Integer(1))","__import__('os')",''])
def test_guard(source):
    with pytest.raises(ValueError):trig_exponential(source)
