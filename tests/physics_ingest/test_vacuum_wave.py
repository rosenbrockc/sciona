import pytest
import sympy as sp
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.vacuum_wave import vacuum_wave,_checked_parse


def inputs():
    return sp.symbols('x y z t',real=True)


def run(field,coordinates=None,mu=sp.Integer(1),eps=sp.Integer(1)):
    q=sp.Tuple(*(coordinates or inputs()))
    return tuple(_checked_parse(s) for s in vacuum_wave(sp.srepr(sp.Tuple(*field)),sp.srepr(q),sp.srepr(mu),sp.srepr(eps)))


def test_independent_polynomial_derivatives():
    x,y,z,t=inputs()
    field=[x*x*y+t*t*z,y*y*z+t**3*x,z*z*x+t**4*y]
    left,right,div=run(field,mu=sp.Integer(2),eps=sp.Integer(3))
    assert left.doit()==sp.Tuple(2*y,2*z,2*x)
    assert right.doit()==sp.Tuple(12*z,36*t*x,72*t*t*y)
    assert sp.expand(div.doit()-(2*x*y+2*y*z+2*z*x))==0


def test_arbitrary_composed_functions_keep_portable_derivatives():
    x,y,z,t=inputs();f=sp.Function('f')
    field=[f(x*y+t),f(z*z-t),sp.I*f(x+y+z+t)]
    left,right,div=run(field)
    assert all(e.has(sp.Derivative) for e in [left,right,div])
    assert not any(e.has(sp.Subs,sp.Dummy) for e in [left,right,div])
    assert left.doit()[0]==sp.diff(field[0],x,2)+sp.diff(field[0],y,2)+sp.diff(field[0],z,2)


def test_longitudinal_wave_is_not_certified_as_maxwell():
    x,y,z,t=inputs();left,right,div=run([0,0,sp.cos(z-t)])
    assert all(sp.simplify(a-b)==0 for a,b in zip(left.doit(),right.doit()))
    assert div.doit()!=0


def test_transverse_plane_wave_and_zero_field():
    x,y,z,t=inputs()
    for field in [[sp.cos(z-t),0,0],[0,0,0]]:
        left,right,div=run(field)
        assert all(sp.simplify(a-b)==0 for a,b in zip(left.doit(),right.doit()))
        assert div.doit()==0


@pytest.mark.parametrize('coefficient',[sp.Integer(0),sp.Integer(-1),sp.I,sp.oo,sp.Symbol('unknown'),sp.exp(inputs()[0])])
def test_invalid_coefficients(coefficient):
    with pytest.raises(ValueError):run([0,0,0],mu=coefficient)


@pytest.mark.parametrize('coordinates',[(1,2,3,4),sp.symbols('a b c d'),(*inputs()[:3],inputs()[0])])
def test_invalid_coordinates(coordinates):
    with pytest.raises(ValueError):run([0,0,0],coordinates=coordinates)


@pytest.mark.parametrize('field',[[1,2],[1,2,3,4],[sp.oo,0,0],[sp.Symbol('x'),0,0]])
def test_invalid_fields(field):
    with pytest.raises(ValueError):run(field)


@pytest.mark.parametrize('source',["Add('x+1', Integer(2))","Derivative('x', Symbol('x'))","__import__('os')","Symbol('x').subs(x, 1)"])
def test_constructor_strings_and_code_rejected(source):
    with pytest.raises(ValueError):_checked_parse(source)
