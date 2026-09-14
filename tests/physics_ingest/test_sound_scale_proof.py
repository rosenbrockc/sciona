from dataclasses import replace
import pytest
import sympy as sp
from sciona.physics_ingest.sound_scale_proof import build_proof,verify_proof,eq


def test_certificate_preserves_approximations():
    r=verify_proof(build_proof())
    assert r['steps_verified']==12 and all(r['independent_checks'].values())
    assert r['approximation_steps']==[1,5,7]
    assert r['physical_universal_bound_proven'] is False


@pytest.mark.parametrize('i',range(12))
def test_corrupt_transition(i):
    p=build_proof();steps=list(p.steps);steps[i]=eq(0,1)
    with pytest.raises(ValueError):verify_proof(replace(p,steps=tuple(steps)))


def test_missing_factor_regression():
    alpha,c,me,mp=sp.symbols('alpha c me mp',positive=True)
    actual=build_proof().steps[-1].rhs
    wrong=alpha*c*sp.sqrt(me/mp)
    assert sp.simplify(wrong/actual)==sp.sqrt(2)


def test_dropping_prefactor_is_not_equality():
    f=sp.Symbol('f',positive=True)
    p=build_proof()
    assert sp.simplify(p.steps[3].rhs.subs(f,4)/p.steps[4].rhs)==2


def test_maximum_requires_mass_lower_bound():
    A=sp.Symbol('A',positive=True)
    p=build_proof()
    assert sp.simplify(p.steps[10].rhs.subs(A,sp.Rational(1,4))/p.steps[11].rhs)==2


def test_shear_can_invalidate_bulk_only_assumption():
    K,rho=sp.symbols('K rho',positive=True)
    assert sp.simplify(sp.sqrt((K+4*K/3)/rho)/sp.sqrt(K/rho))==sp.sqrt(sp.Rational(7,3))
