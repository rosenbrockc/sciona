"""Componentwise reconstruction of the source vacuum electric-field wave proof."""
from dataclasses import dataclass
import sympy as sp

SOURCE_VERSION='8c5bbb6a-f2a9-50a2-8b52-1cc328beaa16'
SOURCE_HASH='08e3943f77707e8effa7b31bd8a2c64c076ace9d1baf010352e8d50c5b7deea2'


def symbols():
    x,y,z,t=sp.symbols('x y z t',real=True)
    E=sp.ImmutableMatrix([sp.Function('E'+k)(x,y,z,t) for k in 'xyz'])
    H=sp.ImmutableMatrix([sp.Function('H'+k)(x,y,z,t) for k in 'xyz'])
    mu,eps=sp.symbols('mu epsilon',positive=True)
    rho=sp.Function('rho')(x,y,z,t)
    return (x,y,z),t,E,H,mu,eps,rho


def curl(field,coordinates):
    x,y,z=coordinates
    a,b,c=field
    return sp.ImmutableMatrix([sp.diff(c,y)-sp.diff(b,z),sp.diff(a,z)-sp.diff(c,x),sp.diff(b,x)-sp.diff(a,y)])


def divergence(field,coordinates):
    return sum(sp.diff(v,q) for v,q in zip(field,coordinates))


def gradient(value,coordinates):
    return sp.ImmutableMatrix([sp.diff(value,q) for q in coordinates])


def laplacian(field,coordinates):
    return sp.ImmutableMatrix([sum(sp.diff(v,q,2) for q in coordinates) for v in field])


@dataclass(frozen=True)
class VacuumWaveProof:
    premises: tuple
    steps: tuple


def eq(a,b):
    return sp.Eq(a,b,evaluate=False)


def build_proof():
    q,t,E,H,mu,eps,rho=symbols()
    ampere=eq(curl(H,q),eps*E.diff(t))
    faraday=eq(curl(E,q),-mu*H.diff(t))
    gauss=eq(divergence(E,q),rho/eps)
    vacuum=eq(rho,0)
    vector_identity=eq(curl(curl(E,q),q),gradient(divergence(E,q),q)-laplacian(E,q))
    return VacuumWaveProof((ampere,faraday,gauss,vacuum,vector_identity),(
        eq(curl(H.diff(t),q),eps*E.diff(t,2)),
        eq(curl(curl(E,q),q),-mu*curl(H.diff(t),q)),
        eq(curl(curl(E,q),q),-mu*eps*E.diff(t,2)),
        eq(divergence(E,q),0),
        eq(curl(curl(E,q),q),-laplacian(E,q)),
        eq(laplacian(E,q),mu*eps*E.diff(t,2)),
    ))


def zero(value):
    items=list(value) if isinstance(value,sp.MatrixBase) else [value]
    return all(sp.simplify(v)==0 for v in items)


def residual(equation):
    return equation.lhs-equation.rhs


def verify_proof(proof):
    q,t,E,H,mu,eps,rho=symbols()
    expected=build_proof()
    if proof.premises!=expected.premises or len(proof.steps)!=6:
        raise ValueError('Five explicit premises and six transitions required')
    for i,(actual,wanted) in enumerate(zip(proof.steps,expected.steps)):
        if not isinstance(actual,sp.Equality) or not zero(actual.lhs-wanted.lhs) or not zero(actual.rhs-wanted.rhs):
            raise ValueError('Source transition differs: '+str(i+1))
    a,f,g,v,identity=proof.premises
    s1,s2,s3,s4,s5,s6=proof.steps
    checks={
        'curl_time_commutation':zero(residual(s1)-residual(a).diff(t)),
        'curl_faraday':zero(residual(s2)-curl(residual(f),q)),
        'maxwell_substitution':zero(residual(s3)-residual(s2)+mu*residual(s1)),
        'charge_free_gauss':zero(residual(s4)-residual(g).subs(rho,0)),
        'componentwise_curl_curl_identity':zero(residual(identity)),
        'divergence_gradient_elimination':zero(residual(s5)-gradient(residual(s4),q)),
        'final_sign_cancellation':zero(residual(s6)-residual(s5)+residual(s3)),
    }
    if not all(checks.values()):
        raise ValueError('Failed vector inference: '+str([k for k,v in checks.items() if not v]))
    return dict(source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,
                source_ast_parity=False,checks=checks,
                premises=[sp.srepr(e) for e in proof.premises],steps=[sp.srepr(e) for e in proof.steps],
                domain='C2 fields on a common open Cartesian space-time region; constant positive vacuum epsilon and mu, zero free charge and current; SI E in V/m and H in A/m.',
                limitation='Necessary wave-equation consequence of the stated Maxwell premises, not a proof that every wave-equation solution satisfies all Maxwell constraints or an initial/boundary-value solver.')
