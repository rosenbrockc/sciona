from dataclasses import replace
import sympy as sp
import pytest
from sciona.physics_ingest.wave_relations_proof import build_proof,verify_proof,symbols


def test_dependency_order_and_phase_cycles():
    result=verify_proof(build_proof())
    assert result['dependency_order']==[4,5,6,1,2,3]
    assert all(result['checks'].values())


@pytest.mark.parametrize('i',range(6))
def test_corrupted_transition_rejected(i):
    proof=build_proof();steps=list(proof.steps);e=steps[i]
    steps[i]=sp.Eq(e.lhs,e.rhs+1,evaluate=False)
    with pytest.raises(ValueError):verify_proof(replace(proof,steps=tuple(steps)))


def test_source_unrelated_f_cannot_cancel_frequency():
    f,T,L,v,w,k=symbols();other=sp.Symbol('unrelated_f')
    assert sp.simplify(other/f)!=1
    assert sp.simplify(f/f)==1


def test_angular_wavenumber_is_not_spatial_cycles():
    f=sp.Integer(3);v=sp.Integer(12);wavelength=v/f
    assert wavelength==4
    k=2*sp.pi/wavelength
    assert k*wavelength==2*sp.pi
    assert k!=1/wavelength
