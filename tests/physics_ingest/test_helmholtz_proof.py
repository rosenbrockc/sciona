from dataclasses import replace
import pytest
import sympy as sp
from sciona.physics_ingest.helmholtz_proof import build_proof,verify_proof,eq,lap


def test_seven_step_residual_equivalence():
    r=verify_proof(build_proof())
    assert r['steps_verified']==7 and r['residual_equivalence'] and r['zero_frequency_included']


@pytest.mark.parametrize('i',range(7))
def test_corrupt_step(i):
    p=build_proof();s=list(p.steps);s[i]=eq(s[i].lhs,s[i].rhs+1)
    with pytest.raises(ValueError):verify_proof(replace(p,steps=tuple(s)))


def test_plane_wave_and_non_solution():
    x,y,z,t=sp.symbols('x y z t',real=True);q=(x,y,z)
    U=sp.exp(sp.I*3*x);phase=sp.exp(sp.I*6*t)
    assert sp.simplify(lap(U,q)+9*U)==0
    assert sp.simplify(lap(U*phase,q)-sp.diff(U*phase,t,2)/4)==0
    assert lap(x*x,q)+9*x*x!=0


def test_zero_frequency_is_laplace_equation():
    x,y,z=sp.symbols('x y z',real=True)
    assert lap(x*x-y*y,(x,y,z))==0


def test_longitudinal_wave_not_maxwell_certification():
    x=sp.Symbol('x',real=True);U=sp.exp(sp.I*x)
    assert sp.simplify(sp.diff(U,x,2)+U)==0
    assert sp.diff(U,x)!=0


def test_time_dependent_amplitude_breaks_reduction():
    t=sp.Symbol('t',real=True);U=t
    assert sp.simplify(sp.diff(U*sp.exp(sp.I*t),t,2)+U*sp.exp(sp.I*t))!=0
