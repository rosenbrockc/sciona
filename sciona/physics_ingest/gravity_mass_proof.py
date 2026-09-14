"""Newtonian spherical-source mass from gravitational acceleration and radius."""
from dataclasses import dataclass
import sympy as sp
SOURCE_VERSION='8d6afa2c-187c-5852-a936-815d8416125d'
SOURCE_HASH='71e731440bf265275efa468cb30a8a2aa08b6cea2d6bff768476dd9a6cb6a60b'
G_VALUE=sp.Rational('6.67430e-11')
GRAVITY_VALUE=sp.Rational('9.80665')
RADIUS_VALUE=sp.Integer(6378100)


def eq(a,b):return sp.Eq(a,b,evaluate=False)


@dataclass(frozen=True)
class GravityMassProof:
    steps:tuple


def derive():
    F,m,a,g,ge,G,M,r,m1,m2,x=sp.symbols('F m a g ge G M r m1 m2 x',positive=True)
    steps=[eq(F,m*a).subs(a,g)]
    steps.append(steps[0].subs(g,ge))
    steps.append(eq(F,G*m1*m2/x**2).subs({m1:M,m2:m,x:r},simultaneous=True))
    steps.append(eq(steps[2].rhs,steps[1].rhs))
    steps.append(eq(sp.cancel(steps[3].lhs/m),sp.cancel(steps[3].rhs/m)))
    steps.append(eq(sp.cancel(steps[4].lhs*r**2/G),sp.cancel(steps[4].rhs*r**2/G)))
    steps.append(steps[5].subs(ge,GRAVITY_VALUE))
    steps.append(steps[6].subs(G,G_VALUE))
    steps.append(steps[7].subs(r,RADIUS_VALUE))
    steps.append(eq(M,sp.cancel(steps[8].rhs)))
    return tuple(steps)


def build_proof():return GravityMassProof(derive())


def verify_proof(proof):
    expected=derive()
    if len(proof.steps)!=10:raise ValueError('Ten source steps required')
    for i,(actual,wanted) in enumerate(zip(proof.steps,expected)):
        if actual!=wanted:raise ValueError(f'Source reconstruction step{i+1} differs')
    result=GRAVITY_VALUE*RADIUS_VALUE**2/G_VALUE
    M,r,G=sp.symbols('M r G',positive=True)
    if sp.cancel((G*M/r**2)*r**2/G-M)!=0:raise ValueError('Inverse law failed')
    return dict(source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,steps_verified=10,
                steps=[sp.srepr(s) for s in expected],computed_mass_kg=str(sp.N(result,25)),
                displayed_source_mass_kg='5.972e24',displayed_source_endpoint_matches=False,
                assumptions=['Newtonian gravitational acceleration magnitude at known positive distance from center.',
                             'Spherically symmetric source, exterior/surface point; positive G and mass.',
                             'Nonzero test mass cancels; no rotation, altitude/radius ambiguity or effective-gravity correction.'],
                limitations=['Source standard gravity and equatorial-radius constants do not constitute a precision Earth mass determination.',
                             'Correct computation is about5.9771974e24kg; source5.972e24 is not obtained from its displayed inputs.',
                             'No literal AST parity; malformed units, renaming and numeric expressions reconstructed.'])
