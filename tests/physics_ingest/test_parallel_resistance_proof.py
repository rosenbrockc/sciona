from dataclasses import replace
import pytest
import sympy as sp
from sciona.physics_ingest.parallel_resistance_proof import build_proof,verify_proof,eq


def test_source_and_zero_voltage_extension():
    r=verify_proof(build_proof())
    assert r['steps_verified']==8 and r['zero_voltage_constitutive_extension']


@pytest.mark.parametrize('i',range(8))
def test_bad_step(i):
    p=build_proof();steps=list(p.steps);steps[i]=eq(steps[i].lhs,steps[i].rhs+1)
    with pytest.raises(ValueError):verify_proof(replace(p,steps=tuple(steps)))


@pytest.mark.parametrize('voltage',[0,12,-12])
def test_independent_circuit(voltage):
    r1,r2=sp.Integer(3),sp.Integer(6)
    total=voltage/r1+voltage/r2
    assert total==sp.Rational(voltage,2)
    assert total*2==voltage


def test_series_is_different():
    assert 1/(sp.Rational(1,3)+sp.Rational(1,6))==2
    assert 3+6!=2


def test_zero_voltage_does_not_identify_resistance():
    assert all(0/r==0 for r in [1,2,3])
    assert sp.sympify(0)/sp.sympify(0) is sp.nan
