"""Sound-speed scaling model with explicit approximation boundaries."""
from dataclasses import dataclass
import sympy as sp

SOURCE_VERSION = 'e0b55979-db6a-51ed-aafb-d3c261bc288c'
SOURCE_HASH = '25e4a8c64922bb7f629c6e09f19696c4a12e0d41fd670eaaebd8a8ca89e4b13c'


def eq(a,b):
    return sp.Eq(a,b,evaluate=False)


@dataclass(frozen=True)
class SoundScaleProof:
    steps: tuple


def derive():
    K,rho,f,E,a,m,e,eps,hbar,me,mp,alpha,c,A = sp.symbols(
        'K rho f E a m e eps hbar me mp alpha c A', positive=True)
    v,vu = sp.symbols('v vu',positive=True)
    rydberg = me*e**4/(32*sp.pi**2*eps**2*hbar**2)
    return (
        eq(v,sp.sqrt(K/rho)),  # bulk-only approximation
        eq(v,sp.sqrt(f*E/(a**3*rho))),
        eq(a**3*rho,m),
        eq(v,sp.sqrt(f)*sp.sqrt(E/m)),
        eq(v,sp.sqrt(E/m)),  # unit-prefactor scaling convention
        eq(alpha*c,e**2/(4*sp.pi*eps*hbar)),
        eq(E,rydberg),  # Rydberg energy scale, not arbitrary bonding energy
        eq(v,sp.sqrt(rydberg/m)),
        eq(v,e**2/(4*sp.pi*eps*hbar)*sp.sqrt(me/(2*m))),
        eq(v,alpha*c*sp.sqrt(me/(2*m))),
        eq(v,alpha*c*sp.sqrt(me/(2*A*mp))),
        eq(vu,alpha*c*sp.sqrt(me/(2*mp))),
    )


def build_proof():
    return SoundScaleProof(derive())


def verify_proof(proof):
    expected=derive()
    if len(proof.steps)!=12:raise ValueError('Twelve transitions required')
    for i,(actual,wanted) in enumerate(zip(proof.steps,expected)):
        if actual!=wanted:raise ValueError(f'Step {i+1} differs')
    K,rho,f,E,a,m,e,eps,hbar,me,mp,alpha,c,A=sp.symbols(
        'K rho f E a m e eps hbar me mp alpha c A',positive=True)
    G=sp.Symbol('G',nonnegative=True)
    r=[s.rhs for s in expected]
    checks={
        'bulk_substitution':sp.simplify(r[0].subs(K,f*E/a**3)-r[1]),
        'density_substitution':sp.simplify(r[1].subs(rho,m/a**3)-r[3]),
        'unit_prefactor':sp.simplify(r[3].subs(f,1)-r[4]),
        'rydberg_substitution':sp.simplify(r[4].subs(E,r[6])-r[7]),
        'charge_factorization':sp.simplify(r[7]-r[8]),
        'alpha_definition':sp.simplify(r[9].subs(alpha,e**2/(4*sp.pi*eps*hbar*c))-r[8]),
        'mass_substitution':sp.simplify(r[9].subs(m,A*mp)-r[10]),
        'maximum_endpoint':sp.simplify(r[10].subs(A,1)-r[11]),
        'inverse_sqrt_mass':sp.simplify(r[10]*sp.sqrt(A)-r[11]),
        'shear_ratio':sp.simplify((sp.sqrt((K+4*G/3)/rho)/r[0])**2-(1+4*G/(3*K))),
    }
    if any(value!=0 for value in checks.values()):raise ValueError('Algebra check failed')
    if sp.diff(r[10],A).is_negative is not True:raise ValueError('Monotonicity unproved')
    return dict(source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,
                steps_verified=12,steps=[sp.srepr(s) for s in expected],
                independent_checks={k:True for k in checks},decreasing_in_atomic_mass=True,
                source_ast_parity=False,physical_universal_bound_proven=False,
                approximation_steps=[1,5,7],
                assumptions=['Positive moduli/density/energy/length/masses/constants; nonnegative shear modulus.',
                             'Bulk-only limit requires G/K small; full longitudinal speed differs by sqrt(1+4G/(3K)).',
                             'Dropping sqrt(f) is a unit-prefactor scaling convention, not an equality at sqrt(f)=2.',
                             'Bonding energy is modeled by the Rydberg scale.',
                             'Maximum of the scaling curve uses A>=1, positive constants and m=A*mp.'],
                limitations=['Conditional condensed-matter scaling estimate, not a universal material-speed certificate.',
                             'No guarantee for arbitrary elastic parameters, high-pressure matter or noncohesive fluids.',
                             'Exact algebra after model choices does not validate those choices for a material.'])
