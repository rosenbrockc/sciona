from dataclasses import replace
import pytest
import sympy as sp
from sciona.physics_ingest.brewster_angle_proof import build_proof,verify_proof,eq


def test_proof():
    r=verify_proof(build_proof())
    assert r['steps_verified']==10 and r['snell_verified'] and r['fresnel_p_zero_verified']


@pytest.mark.parametrize('i',range(10))
def test_mutation(i):
    p=build_proof();s=list(p.steps);s[i]=eq(s[i].lhs,s[i].rhs+1)
    with pytest.raises(ValueError):verify_proof(replace(p,steps=tuple(s)))


def test_direction_ratio():
    forward=sp.atan(sp.Rational(3,2));reverse=sp.atan(sp.Rational(2,3))
    assert float(forward)>float(sp.pi/4)>float(reverse)
    assert abs(float(forward+reverse-sp.pi/2))<1e-15


def test_degrees_not_radians():
    assert sp.sin(sp.pi/2)==1 and sp.sin(90)!=1


def test_inverse_tangent_requires_branch():
    assert sp.atan(sp.tan(3*sp.pi/4))==-sp.pi/4


def test_identical_media_reflect_nothing():
    n=sp.Symbol('n',positive=True);a=sp.Symbol('a',real=True)
    assert n*sp.cos(a)-n*sp.cos(a)==0
