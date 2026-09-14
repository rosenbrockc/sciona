from dataclasses import replace
import pytest
import sympy as sp
from sciona.physics_ingest.vacuum_wave_proof import (
    build_proof,verify_proof,symbols,curl,divergence,gradient,laplacian,zero,residual,
)


def test_arbitrary_smooth_component_proof():
    assert all(verify_proof(build_proof())['checks'].values())


@pytest.mark.parametrize('i',range(6))
def test_corrupted_transition_rejected(i):
    proof=build_proof();steps=list(proof.steps);e=steps[i]
    delta=sp.ones(3,1) if isinstance(e.rhs,sp.MatrixBase) else 1
    steps[i]=sp.Eq(e.lhs,e.rhs+delta,evaluate=False)
    with pytest.raises(ValueError):verify_proof(replace(proof,steps=tuple(steps)))


def test_wrong_faraday_sign_rejected():
    proof=build_proof();premises=list(proof.premises);e=premises[1]
    premises[1]=sp.Eq(e.lhs,-e.rhs,evaluate=False)
    with pytest.raises(ValueError):verify_proof(replace(proof,premises=tuple(premises)))


def test_vector_identity_with_mixed_polynomial_components():
    (x,y,z),t,*_=symbols()
    field=sp.ImmutableMatrix([x*y*z+t*x*x,x*y*y+z*t,z*z*x+y*t*t])
    assert zero(curl(curl(field,(x,y,z)),(x,y,z))-gradient(divergence(field,(x,y,z)),(x,y,z))+laplacian(field,(x,y,z)))
    # This field is not divergence free; dropping grad(div E) would be wrong.
    assert not zero(curl(curl(field,(x,y,z)),(x,y,z))+laplacian(field,(x,y,z)))


def test_transverse_vacuum_plane_wave_satisfies_premises_and_endpoint():
    q,t,E,H,mu,eps,rho=symbols();x,y,z=q
    k=sp.Symbol('k',positive=True);omega=k/sp.sqrt(mu*eps)
    phase=k*z-omega*t
    electric=sp.ImmutableMatrix([sp.cos(phase),0,0])
    magnetic=sp.ImmutableMatrix([0,sp.sqrt(eps/mu)*sp.cos(phase),0])
    assert zero(curl(magnetic,q)-eps*electric.diff(t))
    assert zero(curl(electric,q)+mu*magnetic.diff(t))
    assert divergence(electric,q)==0
    assert zero(laplacian(electric,q)-mu*eps*electric.diff(t,2))


def test_wave_equation_alone_does_not_establish_maxwell():
    q,t,E,H,mu,eps,rho=symbols();x,y,z=q
    phase=z-t/sp.sqrt(mu*eps)
    longitudinal=sp.ImmutableMatrix([0,0,sp.cos(phase)])
    assert zero(laplacian(longitudinal,q)-mu*eps*longitudinal.diff(t,2))
    assert divergence(longitudinal,q)!=0
