from dataclasses import replace
import pytest
import sympy as sp
from sciona.physics_ingest.sine_squared_proof import build_proof,verify_proof,eq


def test_certificate():
    r=verify_proof(build_proof())
    assert r['steps_verified']==13 and r['execution_order'][:4]==[1,3,4,2]
    assert len(r['independent_differential_certificates'])==2


@pytest.mark.parametrize('i',range(13))
def test_corrupt_step(i):
    p=build_proof();steps=list(p.steps);steps[i]=eq(0,1)
    with pytest.raises(ValueError):verify_proof(replace(p,steps=tuple(steps)))


@pytest.mark.parametrize('angle',[0,sp.pi/2,sp.pi,-sp.pi/3])
def test_exact_angles(angle):
    x=sp.Symbol('x',real=True);answer=build_proof().steps[-1]
    assert sp.simplify((answer.lhs-answer.rhs).subs(x,angle))==0


def test_no_positive_sine_branch_inference():
    assert sp.sqrt(sp.sin(-sp.pi/2)**2)==1
    assert sp.sin(-sp.pi/2)==-1
