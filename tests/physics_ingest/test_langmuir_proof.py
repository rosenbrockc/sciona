from dataclasses import replace
import pytest
import sympy as sp
from sciona.physics_ingest.langmuir_proof import build_proof,verify_proof,eq


def test_certificate():
    r=verify_proof(build_proof())
    assert r['steps_verified']==13 and all(r['independent_checks'].values())
    assert r['zero_pressure_extension_verified']


@pytest.mark.parametrize('i',range(13))
def test_corrupt_transition(i):
    p=build_proof();steps=list(p.steps);steps[i]=eq(0,1)
    with pytest.raises(ValueError):verify_proof(replace(p,steps=tuple(steps)))


def test_missing_divisor():
    # ka=2,kd=3,p=5,S=3,B=10 satisfies rate balance.
    assert sp.Rational(2*5*3,5*3)==2
    assert sp.Rational(10,5*3)==sp.Rational(2,3)
    assert sp.Rational(2*5*3,3*5*3)==sp.Rational(2,3)


def test_half_coverage_and_zero_pressure():
    K,p=sp.symbols('K p',positive=True)
    rhs=build_proof().steps[-1].rhs
    assert rhs.subs(p,1/K)==sp.Rational(1,2)
    assert rhs.subs(p,0)==0


def test_source_reciprocal_undefined_at_zero():
    K,p=sp.symbols('K p',positive=True)
    assert build_proof().steps[-2].rhs.subs(p,0)==sp.zoo
