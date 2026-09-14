from dataclasses import replace
import sympy as sp
import pytest
from sciona.physics_ingest.work_energy_proof import build_proof,verify_proof


def test_seven_steps_and_zero_acceleration():
    assert all(verify_proof(build_proof())['checks'].values())


@pytest.mark.parametrize('i',range(7))
def test_corrupt_transition_rejected(i):
    p=build_proof();steps=list(p.steps);e=steps[i];steps[i]=sp.Eq(e.lhs,e.rhs+1,evaluate=False)
    with pytest.raises(ValueError):verify_proof(replace(p,steps=tuple(steps)))


@pytest.mark.parametrize('initial,acceleration,duration',[(3,2,4),(3,-2,1),(3,-2,3),(-3,2,3),(3,0,4)])
def test_constant_force_trajectory_including_reversal(initial,acceleration,duration):
    mass=sp.Rational(7,3);v2=initial+acceleration*duration
    x=initial*duration+sp.Rational(1,2)*acceleration*duration**2
    assert mass*acceleration*x==mass*(v2*v2-initial*initial)/2


def test_indefinite_work_reference_cannot_be_omitted():
    F,x,C=sp.symbols('F x C')
    antiderivative=F*x+C
    assert sp.diff(antiderivative,x)==F
    assert antiderivative.subs(x,0)==C
    assert antiderivative-antiderivative.subs(x,0)==F*x
