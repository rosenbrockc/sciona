from dataclasses import replace
from itertools import product
import sympy as sp
import pytest
from sciona.physics_ingest.curl_curl_proof import build_proof,verify_proof,check_contraction,delta_contraction
from sciona.physics_ingest.vacuum_wave_proof import curl,gradient,divergence,laplacian,zero


def test_complete_component_and_index_proof():
    report=verify_proof(build_proof())
    assert report['contraction_cases']==81 and all(report['checks'].values())


@pytest.mark.parametrize('i',range(6))
def test_corrupt_stage_rejected(i):
    p=build_proof();stages=list(p.stages);stages[i]=stages[i]+sp.ones(3,1)
    with pytest.raises(ValueError):verify_proof(replace(p,stages=tuple(stages)))


def test_reversed_contraction_sign_rejected():
    with pytest.raises(ValueError):check_contraction(lambda i,j,m,n:-delta_contraction(i,j,m,n))


def test_source_double_contraction_is_not_the_needed_identity():
    # The source epsilon(i,j,k)*epsilon(n,j,k) contracts j AND k, yielding 2 delta(i,n).
    for i,n in product(range(3),repeat=2):
        actual=sum(sp.LeviCivita(i,j,k)*sp.LeviCivita(n,j,k) for j,k in product(range(3),repeat=2))
        assert actual==2*int(i==n)
    assert delta_contraction(0,0,0,0)==0


def test_nonzero_divergence_must_be_retained():
    x,y,z=sp.symbols('x y z');q=(x,y,z)
    field=sp.ImmutableMatrix([x*x*y,y*y*z,z*z*x])
    actual=curl(curl(field,q),q)
    assert actual==sp.ImmutableMatrix([2*z,2*x,2*y])
    assert zero(actual-gradient(divergence(field,q),q)+laplacian(field,q))
    assert not zero(actual+laplacian(field,q))
