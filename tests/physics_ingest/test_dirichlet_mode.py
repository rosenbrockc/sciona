"""Synthetic symbolic Dirichlet modes with independent integral checks."""
import json
import pytest
import sympy as sp
from sympy.simplify.fu import TR8
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.dirichlet_mode import dirichlet_mode
from sciona.physics_ingest.source_symbolic import parse_source_srepr


def exact(v):
    return v.xreplace({f:sp.Rational(f) for f in v.atoms(sp.Float)}).doit()


def cases():
    w=sp.Symbol('W',positive=True); n=sp.Symbol('j',integer=True,positive=True); p=sp.Symbol('phase',real=True)
    return [(sp.Integer(2),sp.Integer(1),sp.Integer(0)),(sp.Integer(3),sp.Integer(2),sp.pi),
            (sp.Rational(1,100),sp.Integer(3),sp.pi/2),(sp.sqrt(2),sp.Integer(7),sp.pi/7),
            (w,n,p),(w+1,2*n+1,sp.sin(p))]


def check_outputs(args, outputs):
    w,n,p=map(exact,args); x=sp.Symbol('x',real=True)
    mode,opposite,k,derivative,second=[parse_source_srepr(s).doit() for s in outputs[:5]]
    # Derive the normalization amplitude from a direct integral of the raw sine.
    raw=sp.sin(n*sp.pi*x/w)
    u=sp.Symbol('u',real=True)
    norm=w*sp.integrate(sp.sin(n*sp.pi*u)**2,(u,0,1))
    expected=sp.exp(sp.I*p)*raw/sp.sqrt(norm)
    for actual,ref in zip([mode,opposite,k,derivative,second],
                          [expected,-expected,n*sp.pi/w,sp.diff(expected,x),sp.diff(expected,x,2)]):
        assert sp.simplify(actual-ref)==0
    assert sp.simplify(mode.subs(x,0))==0 and sp.simplify(mode.subs(x,w))==0
    assert sp.simplify(sp.integrate(sp.simplify(w*(mode*sp.conjugate(mode)).subs(x,w*u)),(u,0,1)))==1
    c=json.loads(outputs[5])
    assert c['schema']=='sciona.normalized-dirichlet-mode.v1' and c['source_ast_parity'] is False
    assert len(c['identities'])==7
    for eq in c['identities']:
        assert sp.cancel(TR8(parse_source_srepr(eq['lhs_srepr']).doit()-parse_source_srepr(eq['rhs_srepr']).doit()))==0
    for name,value in zip(['width_srepr','mode_index_srepr','phase_srepr'],[w,n,p]):
        assert parse_source_srepr(c['inputs'][name]).doit()==value
    assert parse_source_srepr(c['coordinate_srepr']).doit()==x
    assert sp.simplify(parse_source_srepr(c['eigenvalue_srepr']).doit()-k*k)==0
    density=parse_source_srepr(c['density_srepr']).doit()
    primitive=parse_source_srepr(c['density_antiderivative_srepr']).doit()
    assert sp.cancel(density-mode*sp.conjugate(mode))==0
    assert sp.cancel(TR8(sp.diff(primitive,x)-density))==0
    assert sp.simplify(primitive.subs(x,w)-primitive.subs(x,0))==1


@pytest.mark.parametrize('args',cases())
def test_exact_and_symbolic_modes(args):
    check_outputs(args,dirichlet_mode(*(sp.srepr(v) for v in args)))


def test_parsed_floats_become_exact():
    args=(sp.Float('0.3'),sp.Float('2.0'),sp.Float('0.1'))
    outputs=dirichlet_mode(*(sp.srepr(v) for v in args))
    check_outputs(args,outputs)
    assert 'Float(' not in ''.join(outputs)


@pytest.mark.parametrize('index,value',[(0,sp.Integer(0)),(0,sp.Integer(-1)),(0,sp.oo),
    (0,sp.Symbol('unknown',real=True)),(1,sp.Integer(0)),(1,sp.Integer(-2)),
    (1,sp.Rational(3,2)),(1,sp.Symbol('j',positive=True)),(2,sp.I),(2,sp.oo),
    (2,sp.Symbol('unknown')),(0,sp.Symbol('x',positive=True)),(2,sp.sin(sp.Symbol('x',real=True)))])
def test_invalid_or_unproved_domains(index,value):
    args=list(cases()[0]);args[index]=value
    with pytest.raises(ValueError):dirichlet_mode(*(sp.srepr(v) for v in args))


def test_conflicting_symbol_assumptions():
    with pytest.raises(ValueError,match='Conflicting'):
        dirichlet_mode("Symbol('a', positive=True)",'Integer(1)',"Symbol('a', real=True)")


@pytest.mark.parametrize('source',["Add('1+2', Integer(1))","__import__('os')","Integral('x', Symbol('x'))"])
def test_constructor_text_rejected(source):
    with pytest.raises(ValueError):dirichlet_mode(source,'Integer(1)','Integer(0)')
