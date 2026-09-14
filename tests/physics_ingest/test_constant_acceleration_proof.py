import pytest
import sympy as sp
from sciona.physics_ingest.constant_acceleration_proof import build_proof,verify_proof


def test_all_steps_and_extensions():
    r=verify_proof(build_proof())
    assert r['reconstructed_steps_validated']==23
    assert len(r['integration_checks'])==8 and all(r['integration_checks'].values())


@pytest.mark.parametrize('index',range(23))
def test_mutated_steps_rejected(index):
    p=build_proof();e=p['steps'][index]
    p['steps'][index]=sp.Eq(e.lhs,e.rhs+1,evaluate=False)
    with pytest.raises(ValueError):verify_proof(p)


def test_variable_acceleration_endpoint_average_invalid():
    t=sp.Symbol('t',real=True)
    velocity=t**2
    assert sp.integrate(velocity,(t,0,1))==sp.Rational(1,3)
    assert (velocity.subs(t,0)+velocity.subs(t,1))/2==sp.Rational(1,2)


def test_displacement_is_not_total_distance():
    t=sp.Symbol('t',real=True)
    velocity=1-t
    assert sp.integrate(velocity,(t,0,2))==0
    assert sp.integrate(velocity,(t,0,1))-sp.integrate(velocity,(t,1,2))==1


def test_squared_velocity_loses_sign():
    v=sp.Symbol('v',real=True)
    assert sp.solve(v**2-4,v)==[-2,2]
