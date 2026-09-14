import pytest
import sympy as sp
from sciona.physics_ingest.newton_force_proof import build_proof,verify_proof


def test_conditional_certificate_preserves_gaps():
    r=verify_proof(build_proof())
    assert not r['source_derivation_valid_without_added_assumptions']
    assert len(r['gaps'])==4 and all(r['conditional_checks'].values())


@pytest.mark.parametrize('key',['force','circular_force','period_squared','kepler_coefficient'])
def test_corrupt_conditional_identity(key):
    p=build_proof();p[key]+=1
    with pytest.raises(ValueError):verify_proof(p)


def test_newton_second_law_does_not_fix_mass_scaling():
    m=sp.Symbol('m',positive=True)
    force=m*m  # a=m obeys F=ma, but F is quadratic in m.
    assert force.subs(m,2*m)==4*force


def test_period_constraint_is_necessary():
    r,T,m=sp.symbols('r T m',positive=True)
    force=4*sp.pi**2*m*r/T**2
    assert sp.cancel(force.subs(r,2*r)/force)==2
    assert sp.cancel(force.subs({r:2*r,T:sp.sqrt(8)*T},simultaneous=True)/force)==sp.Rational(1,4)


def test_independent_mass_dependence_is_additional():
    a,b=sp.symbols('a b',positive=True)
    # Separate renamed equations F1=a and F2=b do not imply F=a*b.
    assert sp.diff(a*b,a)==b and sp.diff(a,a)==1
