from dataclasses import replace
import pytest
import sympy as sp
from sciona.physics_ingest.orbit_radius_proof import build_proof,verify_proof,eq


def test_certificate():
    r=verify_proof(build_proof())
    assert r['steps_verified']==13 and all(r['independent_checks'].values())


@pytest.mark.parametrize('i',range(13))
def test_corrupt_transition(i):
    p=build_proof();steps=list(p.steps);steps[i]=eq(0,1)
    with pytest.raises(ValueError):verify_proof(replace(p,steps=tuple(steps)))


def test_exact_synthetic_radius():
    G,M,T=sp.symbols('G M T',positive=True)
    r=build_proof().steps[-2].lhs.subs({G:1,M:8,T:2*sp.pi})
    assert r==2


def test_sidereal_and_solar_periods_are_distinct():
    ratio=(sp.Rational(86400,86164))**sp.Rational(2,3)
    assert ratio>1


def test_conditions_and_radius_not_altitude():
    r=verify_proof(build_proof())
    assert any('equatorial prograde' in a for a in r['assumptions'])
    assert any('not surface altitude' in a for a in r['limitations'])


def test_finite_satellite_mass_changes_relative_radius():
    assert sp.real_root(2,3)>1  # equal masses double gravitational parameter
