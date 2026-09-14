"""Circular-path speed and the source's explicitly approximate Earth example."""
from dataclasses import dataclass
import sympy as sp

SOURCE_VERSION='ec76168f-50f7-5a81-b2d7-0be499024080'
SOURCE_HASH='910ac2cd2cdce3f00953de162c6d396353b0d0761f96c7cc4c3258a01242ad9c'
SECONDS_365_DAYS=365*24*60*60
RADIUS_KM=149600000


def eq(a,b):return sp.Eq(a,b,evaluate=False)


@dataclass(frozen=True)
class CircularOrbitProof:
    steps:tuple


def build_proof():
    v,d,t,C,r,ve,te,Ce,re=sp.symbols('v d t C r v_orbit T_orbit C_orbit R_orbit',positive=True)
    return CircularOrbitProof((eq(v,C/t),eq(Ce,2*sp.pi*re),eq(ve,Ce/te),
        eq(ve,2*sp.pi*re/te),eq(te,sp.Integer(SECONDS_365_DAYS)),
        eq(ve,2*sp.pi*re/SECONDS_365_DAYS),
        eq(ve,2*sp.pi*RADIUS_KM/SECONDS_365_DAYS),
        eq(ve,sp.Rational(1870,1971)*10*sp.pi)))


def verify_proof(proof):
    v,d,t,C,r,ve,te,Ce,re=sp.symbols('v d t C r v_orbit T_orbit C_orbit R_orbit',positive=True)
    if len(proof.steps)!=8:raise ValueError('Eight steps required')
    expected=[]
    expected.append(eq(v,d/t).subs(d,C))
    expected.append(eq(C,2*sp.pi*r).subs({C:Ce,r:re},simultaneous=True))
    expected.append(expected[0].subs({v:ve,C:Ce,t:te},simultaneous=True))
    expected.append(expected[2].subs(Ce,expected[1].rhs))
    expected.append(eq(te,sp.Integer(365)*24*60*60))
    expected.append(expected[3].subs(te,expected[4].rhs))
    expected.append(expected[5].subs(re,RADIUS_KM))
    expected.append(eq(ve,sp.simplify(expected[6].rhs)))
    for i,(actual,wanted) in enumerate(zip(proof.steps,expected)):
        if actual!=wanted:raise ValueError(f'Reconstruction step {i+1} differs')
    corrected=expected[-1].rhs
    rounded_period=2*sp.pi*RADIUS_KM/31600000
    if not sp.Rational(595,20)<corrected<sp.Rational(597,20):
        raise ValueError('Corrected result does not round to29.8')
    if not sp.Rational(593,20)<rounded_period<sp.Rational(595,20):
        raise ValueError('Source rounded-period inconsistency not demonstrated')
    return dict(source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,steps_verified=8,
                exact_365_day_seconds=SECONDS_365_DAYS,example_radius_km=RADIUS_KM,
                corrected_speed_km_s=str(sp.N(corrected,20)),source_rounded_period_speed_km_s=str(sp.N(rounded_period,20)),
                source_final_value_is_approximate=True,source_ast_parity=False,
                steps=[sp.srepr(s) for s in proof.steps],
                scope='Average speed around a circular path from radius and period; instantaneous speed only for uniform circular motion.',
                limitations=['365-day year and stated radius are educational approximations, not an ephemeris.',
                             'Earth orbit is not exactly circular; no gravitational dynamics or eccentric-orbit instantaneous speed claim.',
                             'Source final29.8 is rounded, not an exact equality; retain exact unit conversion before final rounding.'])
