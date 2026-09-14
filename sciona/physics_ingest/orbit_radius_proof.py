"""Circular orbital radius from period, with conditional geostationary naming."""
from dataclasses import dataclass
import sympy as sp
SOURCE_VERSION='ac570cf4-7525-5ddb-8efb-63a438610c9e'
SOURCE_HASH='85901269fd506aada9589cc09b529eccdbf85d994559dd52babd2a5a988edf93'


def eq(a,b):return sp.Eq(a,b,evaluate=False)


@dataclass(frozen=True)
class OrbitRadiusProof:
    steps:tuple


def derive():
    G,M,m,r,v,T,Tg,rg,Fg,Fc,d,t=sp.symbols('G M m r v T Tg rg Fg Fc d t',positive=True)
    gravity=eq(Fg,G*M*m/r**2)
    centripetal=eq(Fc,m*v**2/r)
    balance=eq(centripetal.rhs,gravity.rhs)
    circumference=eq(d,2*sp.pi*r)
    speed=eq(v,circumference.rhs/t)
    periodic=eq(v,speed.rhs.subs(t,T))
    squared=eq(v**2,periodic.rhs**2)
    cancelled=eq(sp.cancel(balance.lhs/(m/r)),sp.cancel(balance.rhs/(m/r)))
    compare=eq(cancelled.rhs,squared.rhs)
    multiplied=eq(sp.cancel(compare.lhs*r*T**2),sp.cancel(compare.rhs*r*T**2))
    cubic=eq(multiplied.lhs/(4*sp.pi**2),sp.cancel(multiplied.rhs/(4*sp.pi**2)))
    radius=eq(cubic.lhs**sp.Rational(1,3),r)
    specialized=radius.subs({T:Tg,r:rg},simultaneous=True)
    return (gravity,centripetal,balance,circumference,speed,periodic,squared,
            cancelled,compare,multiplied,cubic,radius,specialized)


def build_proof():return OrbitRadiusProof(derive())


def verify_proof(proof):
    expected=derive()
    if len(proof.steps)!=13:raise ValueError('Thirteen transitions required')
    for i,(actual,wanted) in enumerate(zip(proof.steps,expected)):
        if actual!=wanted:raise ValueError(f'Step{i+1} differs')
    G,M,T=sp.symbols('G M T',positive=True)
    radius=proof.steps[-2].lhs
    speed=2*sp.pi*radius/T
    checks={
        'kepler_residual':sp.simplify(4*sp.pi**2*radius**3-G*M*T**2),
        'centripetal_gravity_residual':sp.simplify(speed**2/radius-G*M/radius**2),
        'period_inverse':sp.simplify(2*sp.pi*sp.sqrt(radius**3/(G*M))-T),
        'period_scaling':sp.simplify(radius.subs(T,8*T)-4*radius),
        'mass_scaling':sp.simplify(radius.subs(M,8*M)-2*radius),
    }
    if any(v!=0 for v in checks.values()):raise ValueError('Independent orbit certificate failed')
    return dict(source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,
                steps_verified=13,steps=[sp.srepr(s) for s in expected],
                independent_checks={k:True for k in checks},source_ast_parity=False,
                assumptions=['Positive G, central mass, orbital period and center-to-center radius; nonzero satellite test mass cancels.',
                             'Uniform circular orbit; gravity is the sole centripetal force, spherical exterior gravitational field.',
                             'Negligible satellite mass relative to central mass; fixed-central-body Newtonian model.',
                             'Geostationary interpretation additionally requires equatorial prograde circular motion at sidereal rotation period.'],
                limitations=['Computed radius is measured from center, not surface altitude.',
                             'Period matching alone does not certify geostationary motion or exterior clearance.',
                             'No oblateness, perturbations, relativistic or station-keeping model.',
                             'Finite two-body relative orbit uses G*(M+m); this source assumes negligible m.'])
