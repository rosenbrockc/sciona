from dataclasses import replace
import sympy as sp
import pytest
from sciona.physics_ingest.ideal_gas_compressibility_proof import build_proof,verify_proof


def test_six_step_proof():
    assert all(verify_proof(build_proof())['checks'].values())


@pytest.mark.parametrize('i',range(6))
def test_corrupt_transition(i):
    proof=build_proof();steps=list(proof.steps);e=steps[i]
    steps[i]=sp.Eq(e.lhs,e.rhs+1,evaluate=False)
    with pytest.raises(ValueError):verify_proof(replace(proof,steps=tuple(steps)))


def test_isothermal_not_isentropic():
    P=sp.Symbol('P',positive=True)
    iso=3/P;adiabatic=3/P**sp.Rational(3,5)
    assert sp.simplify(-sp.diff(iso,P)/iso)==1/P
    assert sp.simplify(-sp.diff(adiabatic,P)/adiabatic)==sp.Rational(3,5)/P
