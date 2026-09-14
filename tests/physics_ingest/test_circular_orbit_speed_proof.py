from dataclasses import replace
import pytest
import sympy as sp
from sciona.physics_ingest.circular_orbit_speed_proof import build_proof,verify_proof,eq,SECONDS_365_DAYS,RADIUS_KM


def test_full_reconstruction():
    r=verify_proof(build_proof())
    assert r['steps_verified']==8
    assert r['exact_365_day_seconds']==31536000
    assert r['source_final_value_is_approximate']


@pytest.mark.parametrize('i',range(8))
def test_mutated_step(i):
    p=build_proof();s=list(p.steps);s[i]=eq(s[i].lhs,s[i].rhs+1)
    with pytest.raises(ValueError):verify_proof(replace(p,steps=tuple(s)))


def test_independent_decimal_rounding():
    import decimal
    with decimal.localcontext() as ctx:
        ctx.prec=60
        pi=decimal.Decimal('3.141592653589793238462643383279502884197169399375105820974944')
        full=2*pi*decimal.Decimal(RADIUS_KM)/decimal.Decimal(365*86400)
        early=2*pi*decimal.Decimal(RADIUS_KM)/decimal.Decimal(31600000)
        assert full.quantize(decimal.Decimal('0.1'))==decimal.Decimal('29.8')
        assert early.quantize(decimal.Decimal('0.1'))==decimal.Decimal('29.7')
        assert abs(full-decimal.Decimal(verify_proof(build_proof())['corrected_speed_km_s']))<decimal.Decimal('1e-18')


def test_speed_unit_conversion():
    km_s=2*sp.pi*RADIUS_KM/SECONDS_365_DAYS
    m_s=2*sp.pi*(RADIUS_KM*1000)/SECONDS_365_DAYS
    assert m_s==1000*km_s


def test_nonuniform_motion_average_vs_instantaneous():
    t=sp.Symbol('t',real=True)
    angle=t+sp.sin(t)/2
    assert sp.simplify(angle.subs(t,2*sp.pi)-angle.subs(t,0))==2*sp.pi
    assert sp.diff(angle,t).subs(t,0)!=1
