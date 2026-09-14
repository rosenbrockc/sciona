import sympy as sp
import pytest
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.curl_curl import curl_curl,_checked_parse


def run(field,q=None):
    q=q or sp.symbols('x y z',real=True)
    return tuple(_checked_parse(s) for s in curl_curl(sp.srepr(sp.Tuple(*field)),sp.srepr(sp.Tuple(*q))))


def test_explicit_polynomial_components_and_divergence():
    x,y,z=sp.symbols('x y z',real=True)
    double,grad,lap=run([x*x*y,y*y*z,z*z*x])
    assert double.doit()==sp.Tuple(2*z,2*x,2*y)
    assert grad.doit()==sp.Tuple(2*y+2*z,2*x+2*z,2*x+2*y)
    assert lap.doit()==sp.Tuple(2*y,2*z,2*x)


@pytest.mark.parametrize('kind',['composed','complex','constant','trigonometric'])
def test_general_field_identity_and_portable_derivatives(kind):
    x,y,z=sp.symbols('x y z',real=True);f=sp.Function('f')
    field={'composed':[f(x*y),f(z*z),f(x+y+z)],'complex':[sp.I*x*y,sp.exp(z),sp.I*sp.sin(x)],
           'constant':[0,1,2],'trigonometric':[sp.sin(y),sp.cos(z),sp.sin(x)]}[kind]
    double,grad,lap=run(field)
    assert all(e.has(sp.Derivative) for e in [double,grad,lap])
    assert not any(e.has(sp.Subs,sp.Dummy) for e in [double,grad,lap])
    assert all(sp.simplify(a.doit()-b.doit()+c.doit())==0 for a,b,c in zip(double,grad,lap))


@pytest.mark.parametrize('q',[(1,2,3),sp.symbols('a b c'),(sp.Symbol('x',real=True),)*3])
def test_invalid_coordinates(q):
    with pytest.raises(ValueError):run([0,0,0],q)


@pytest.mark.parametrize('field',[[1,2],[1,2,3,4],[sp.oo,0,0],[sp.Symbol('x'),0,0],[sp.Symbol('A',commutative=False),0,0]])
def test_invalid_fields(field):
    with pytest.raises(ValueError):run(field)


@pytest.mark.parametrize('source',["Add('x+1', Integer(2))","Derivative('x', Symbol('x'))","__import__('os')","Symbol('x').subs(x, 1)"])
def test_constructor_strings_and_code_rejected(source):
    with pytest.raises(ValueError):_checked_parse(source)
