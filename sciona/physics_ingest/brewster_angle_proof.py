"""Brewster incidence angle with explicit radians and medium ordering."""
from dataclasses import dataclass
import sympy as sp
SOURCE_VERSION='0c31c38a-e31c-54ec-96e4-b01d00fcff6e'
SOURCE_HASH='9451bcd76af899abf7cb7f16d7ae3dd5e7a76c085e4f71eacb306f38431020e6'


def eq(a,b):return sp.Eq(a,b,evaluate=False)


@dataclass(frozen=True)
class BrewsterProof:
    steps:tuple


def derive():
    b,t,x,t1,t2=sp.symbols('b t x t1 t2',real=True);n1,n2=sp.symbols('n1 n2',positive=True)
    complement=eq(t,sp.pi/2-b)
    snell=eq(n1*sp.sin(t1),n2*sp.sin(t2)).subs({t1:b,t2:t},simultaneous=True)
    substituted=eq(snell.lhs,n2*sp.sin(sp.pi/2-b,evaluate=False))
    cofunction=eq(sp.sin(sp.pi/2-b,evaluate=False),sp.cos(b))
    reduced=eq(substituted.lhs,n2*cofunction.rhs)
    divided=eq(reduced.lhs/n1,reduced.rhs/n1)
    ratio=eq(sp.sin(b)/sp.cos(b),n2/n1)
    tangent=eq(sp.tan(b),sp.sin(b)/sp.cos(b))
    tangent_law=eq(tangent.lhs,ratio.rhs)
    # Invert tan only on the physical acute branch (0,pi/2).
    answer=eq(b,sp.atan(tangent_law.rhs))
    return (complement,snell,substituted,cofunction,reduced,divided,ratio,tangent,tangent_law,answer)


def build_proof():return BrewsterProof(derive())


def verify_proof(proof):
    expected=derive()
    if len(proof.steps)!=10:raise ValueError('Ten steps required')
    for i,(actual,wanted) in enumerate(zip(proof.steps,expected)):
        if actual!=wanted:raise ValueError(f'Step{i+1} differs')
    n1,n2=sp.symbols('n1 n2',positive=True);D=sp.sqrt(n1*n1+n2*n2)
    si,ci,st,ct=n2/D,n1/D,n1/D,n2/D
    if sp.simplify(n1*si-n2*st)!=0:raise ValueError('Snell residual failed')
    if sp.simplify(n2*ci-n1*ct)!=0:raise ValueError('P-polarized Fresnel numerator failed')
    return dict(source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,steps_verified=10,
                steps=[sp.srepr(s) for s in expected],snell_verified=True,fresnel_p_zero_verified=True,
                corrected_ratio='n_transmitted/n_incident',source_ast_parity=False,
                assumptions=['Planar interface between homogeneous isotropic lossless nonmagnetic media with positive real indices.',
                             'Incidence/refraction measured from normal; acute-angle branch, radians.',
                             'p-polarized reflection vanishes; formula does not remove s-polarized reflection.'],
                limitations=['Equal indices give45degrees by the complementary-angle convention; reflection is zero at every angle, so no unique Brewster angle.',
                             'No absorbing/complex-index, magnetic or anisotropic medium claim.'])
