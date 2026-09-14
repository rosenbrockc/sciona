"""Explicit work reference and corrected constant-force work-energy derivation."""
from dataclasses import dataclass
import sympy as sp

SOURCE_VERSION='9bed89a4-8da7-59e4-a67a-1bdc0a1fffe0'
SOURCE_HASH='8f8447851adca1da1db0d209a0c61b3ac455d5c6bfe54e6e969b4f5cfd4dace5'


def symbols():
    return (*sp.symbols('W F x v1 v2 KE1 KE2',real=True),sp.Symbol('m',positive=True),sp.Symbol('a',real=True,nonzero=True))


@dataclass(frozen=True)
class WorkEnergyProof:
    steps: tuple


def build_proof():
    W,F,x,v1,v2,K1,K2,m,a=symbols();u,q=sp.symbols('work_dummy displacement_dummy',real=True)
    eq=lambda l,r:sp.Eq(l,r,evaluate=False)
    return WorkEnergyProof((
        eq(sp.Integral(1,(u,0,W)),F*sp.Integral(1,(q,0,x))),
        eq(W,F*x),eq(W,m*a*x),eq(x,(v2**2-v1**2)/(2*a)),
        eq(W,sp.Mul(m,a,(v2**2-v1**2)/(2*a),evaluate=False)),
        eq(W,m*v2**2/2-m*v1**2/2),eq(W,K2-K1),
    ))


def verify_proof(proof):
    expected=build_proof();W,F,x,v1,v2,K1,K2,m,a=symbols()
    if len(proof.steps)!=7 or proof.steps!=expected.steps:raise ValueError('Seven corrected source transitions required')
    s=proof.steps
    d,v,v0=sp.symbols('d v v0',real=True)
    renamed=sp.Eq(d,(v**2-v0**2)/(2*a),evaluate=False).xreplace({d:x,v:v2,v0:v1})
    checks=dict(
        work_reference=sp.diff(s[0].lhs.doit(),W)==1 and sp.diff(s[0].rhs.doit(),x)==F,
        definite_integrals=s[0].lhs.doit()==s[1].lhs and s[0].rhs.doit()==s[1].rhs,
        newton_substitution=sp.simplify(s[1].rhs.subs(F,m*a)-s[2].rhs)==0,
        distinct_velocity_renaming=renamed==s[3],
        displacement_substitution=sp.simplify(s[2].rhs.subs(x,s[3].rhs)-s[4].rhs)==0,
        acceleration_cancellation=sp.simplify(s[4].rhs-s[5].rhs)==0,
        kinetic_energy_definitions=sp.simplify(s[6].rhs.subs({K1:m*v1**2/2,K2:m*v2**2/2})-s[5].rhs)==0)
    # Independent trajectory check avoids dividing by acceleration, including a=0.
    alpha,t=sp.symbols('alpha t',real=True)
    displacement=v1*t+alpha*t*t/2;final_velocity=v1+alpha*t
    trajectory_residual=sp.expand(m*alpha*displacement-m*(final_velocity**2-v1**2)/2)
    checks['trajectory_without_division']=trajectory_residual==0
    checks['zero_acceleration_branch']=sp.simplify((m*(final_velocity**2-v1**2)/2).subs(alpha,0))==0
    if not all(checks.values()):raise ValueError('Work-energy inference failed')
    return dict(source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,source_ast_parity=False,
                steps=[sp.srepr(s) for s in proof.steps],checks=checks,
                domain='Classical point mass with positive constant mass, 1D signed velocities and displacement, constant net force/acceleration. Work accumulated from a reference endpoint set to zero.',
                division_branch='Source divided-acceleration steps require a nonzero; the independent trajectory identity covers a=0 with unchanged velocity and zero net work.',
                limitation='Net translational work and endpoint kinetic energies; no applied-force-only, relativistic, variable-mass, rotational or dissipated-heat claim.')
