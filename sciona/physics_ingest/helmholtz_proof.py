"""Componentwise time-harmonic wave to Helmholtz reconstruction."""
from dataclasses import dataclass
import sympy as sp

SOURCE_VERSION='ba802b17-88be-59f5-a399-a53a3c385c9c'
SOURCE_HASH='297cc23085554fb9ca3b51f3690593152bb3e141c598ad97abc92e4e7477420b'


def symbols():
    x,y,z,t=sp.symbols('x y z t',real=True)
    omega=sp.Symbol('omega',real=True)
    c,mu,eps=sp.symbols('c mu eps',positive=True)
    return x,y,z,t,omega,c,mu,eps


def lap(f,coords):return sum(sp.diff(f,q,2) for q in coords)
def eq(a,b):return sp.Eq(a,b,evaluate=False)


@dataclass(frozen=True)
class HelmholtzProof:
    steps: tuple


def build_proof():
    x,y,z,t,w,c,mu,eps=symbols();q=(x,y,z)
    U=sp.Function('U')(*q);E=sp.Function('E')(*q,t);phase=sp.exp(sp.I*w*t)
    lhs=lap(U,q)*phase
    return HelmholtzProof((eq(E,U*phase),
        eq(lap(E,q),mu*eps*sp.Derivative(E,(t,2),evaluate=False)),
        eq(lhs,mu*eps*U*sp.Derivative(phase,(t,2),evaluate=False)),
        eq(lhs,sp.I*w*mu*eps*U*sp.Derivative(phase,t,evaluate=False)),
        eq(lhs,-w*w*mu*eps*U*phase),
        eq(lhs,-w*w/c**2*U*phase),
        eq(lap(U,q),-w*w/c**2*U)))


def verify_proof(proof):
    x,y,z,t,w,c,mu,eps=symbols();q=(x,y,z)
    U=sp.Function('U')(*q);phase=sp.exp(sp.I*w*t)
    canonical=build_proof()
    if len(proof.steps)!=7 or any(not isinstance(s,sp.Equality) for s in proof.steps):
        raise ValueError('Seven unevaluated source reconstruction steps required')
    # Ansatz is an explicit restriction, not a consequence of the PDE.
    if proof.steps[:2]!=canonical.steps[:2]:raise ValueError('Ansatz or component wave equation differs')
    substituted=proof.steps[1].subs(proof.steps[0].lhs,proof.steps[0].rhs).doit()
    for i in [2,3,4]:
        step=proof.steps[i]
        if sp.simplify(step.lhs-substituted.lhs)!=0 or sp.simplify(step.rhs.doit()-substituted.rhs)!=0:
            raise ValueError('Time derivative evaluation differs')
        if step!=canonical.steps[i]:raise ValueError('Source evaluation stage differs')
    if proof.steps[5]!=eq(proof.steps[4].lhs,proof.steps[4].rhs.subs(mu*eps,1/c**2)):
        raise ValueError('Wave-speed substitution differs')
    last=eq(sp.simplify(proof.steps[5].lhs/phase),sp.simplify(proof.steps[5].rhs/phase))
    if proof.steps[6]!=last:raise ValueError('Nonzero exponential cancellation differs')
    # Equivalence for arbitrary amplitude, not an assertion that every U solves
    # either PDE. The nonzero harmonic factor maps residuals in both directions.
    wave_residual=lap(U*phase,q)-sp.diff(U*phase,t,2)/c**2
    helmholtz_residual=lap(U,q)+w**2*U/c**2
    if sp.simplify(wave_residual-phase*helmholtz_residual)!=0:
        raise ValueError('Residual equivalence failed')
    return dict(source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,steps_verified=7,
        steps=[sp.srepr(s) for s in proof.steps],residual_equivalence=True,zero_frequency_included=True,
        assumptions=['Cartesian C2 field components U independent of time.',
                     'Real constant omega including zero; positive constant c, mu, epsilon; mu*epsilon=1/c².',
                     'Complex harmonic convention exp(+i omega t); physical real part may be taken.'],
        limitations=['Harmonic ansatz restricts the wave-equation solution class; not every wave is monochromatic.',
                     'Each vector component obeys this reduction; Maxwell divergence constraint is separate.',
                     'Helmholtz residual is not necessarily zero for an arbitrary spatial amplitude.'],
        source_ast_parity=False)
