from dataclasses import replace
import pytest
import sympy as sp
from sciona.physics_ingest.trig_exponential_proof import build_proof,verify_proof,eq


def test_complete():
    r=verify_proof(build_proof())
    assert r['steps_verified']==10 and r['independent_linear_system']


@pytest.mark.parametrize('i',range(10))
def test_mutation(i):
    p=build_proof();steps=list(p.steps);steps[i]=eq(steps[i].lhs,steps[i].rhs+1)
    with pytest.raises(ValueError):verify_proof(replace(p,steps=tuple(steps)))


@pytest.mark.parametrize('x',[0,sp.pi/2,sp.pi,-sp.pi/2,2*sp.pi])
def test_exact_angles(x):
    p,n=sp.exp(sp.I*x),sp.exp(-sp.I*x)
    assert sp.simplify((p+n)/2-sp.cos(x))==0
    assert sp.simplify((p-n)/(2*sp.I)-sp.sin(x))==0


def test_imaginary_unit_and_sign_matter():
    p,n=sp.I,-sp.I
    assert (p-n)/(2*sp.I)==1
    assert (p-n)/2!=1
    assert (n-p)/(2*sp.I)!=1
