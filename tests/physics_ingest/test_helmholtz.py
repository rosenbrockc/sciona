import pytest
import sympy as sp
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.helmholtz import helmholtz,_checked_parse
x,y,z,t=sp.symbols('x y z t',real=True)
Q=sp.Tuple(x,y,z,t)


def invoke(U,w=2,c=1,q=Q):
    return tuple(_checked_parse(s) for s in helmholtz(*map(sp.srepr,[sp.Tuple(*U),q,sp.sympify(w),sp.sympify(c)])))


@pytest.mark.parametrize('U,w,c',[
    ((sp.exp(sp.I*2*z),0,0),2,1),((0,0,sp.exp(sp.I*2*z)),2,1),
    ((x*x,y*y,z*z),2,3),((x*x-y*y,0,1),0,3),
    ((sp.I*sp.sin(y),sp.exp(z),sp.cos(x)),-3,2),
    ((sp.Function('f')(x*y),sp.Function('f')(z*z),0),2,1),
])
def test_full_reduction(U,w,c):
    harmonic,lap,residual=invoke(U,w,c)
    phase=sp.exp(sp.I*w*t)
    for u,h,l,r in zip(U,harmonic,lap,residual):
        expected=sum(sp.diff(u,q,2) for q in [x,y,z])
        assert sp.simplify(h-u*phase)==0
        assert sp.simplify(l.doit()-expected)==0
        assert sp.simplify(r.doit()-expected-sp.Rational(w,c)**2*u)==0
        assert sp.simplify(sum(sp.diff(h,q,2) for q in [x,y,z])-sp.diff(h,t,2)/c**2-phase*r.doit())==0


def test_non_solution_preserved():
    assert invoke((x*x,0,0))[2][0].doit()==4*x*x+2


def test_longitudinal_is_not_gauss():
    h,_,r=invoke((sp.exp(sp.I*2*x),0,0))
    assert sp.simplify(r[0].doit())==0
    assert sp.diff(h[0],x)!=0


@pytest.mark.parametrize('U,w,c,q',[
    ((t,0,0),2,1,Q),((sp.Function('f')(t),0,0),2,1,Q),
    ((1,2),2,1,Q),((1,2,3),sp.I,1,Q),((1,2,3),2,0,Q),
    ((1,2,3),2,-1,Q),((1,2,3),x,1,Q),
    ((1,2,3),2,sp.exp(x),Q),((sp.oo,0,0),2,1,Q),
    ((sp.nan,0,0),2,1,Q),((1,2,3),2,1,sp.Tuple(x,x,z,t)),
    ((1,2,3),2,1,sp.Tuple(x,y,z,sp.Symbol('s'))),
    ((sp.Symbol('x'),0,0),2,1,Q),
])
def test_invalid(U,w,c,q):
    with pytest.raises(ValueError):invoke(U,w,c,q)


@pytest.mark.parametrize('payload',["Integral('1+1')","Add('1',Integer(1))","__import__('os')"])
def test_guard(payload):
    with pytest.raises(ValueError):_checked_parse(payload)
