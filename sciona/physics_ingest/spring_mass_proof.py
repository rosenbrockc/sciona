"""Reconstructed undamped spring motion; source endpoint omits a square root."""
from dataclasses import dataclass
import sympy as sp

SOURCE_VERSION = 'cba4334a-0e0e-5a0e-855f-7aa7c41a5514'
SOURCE_HASH = '8d0506313ea4abd27f3a0ad7463b517bb7dd223769820bfa885b292ccc9db335'


def eq(a, b):
    return sp.Eq(a, b, evaluate=False)


@dataclass(frozen=True)
class SpringMassProof:
    steps: tuple


def derive():
    m, k = sp.symbols('m k', positive=True)
    t, A, omega, a = sp.symbols('t A omega a', real=True)
    x = sp.Function('x')(t)
    newton_hooke = eq(m*a, -k*x)
    acceleration = eq(a, newton_hooke.rhs/m)
    ode = eq(sp.diff(x, t, 2), acceleration.rhs)
    ansatz = eq(x, A*sp.cos(omega*t))
    velocity = eq(sp.diff(x, t), sp.diff(ansatz.rhs, t))
    second = eq(sp.diff(x, t, 2), sp.diff(velocity.rhs, t))
    comparison = eq(ode.rhs, second.rhs)
    substitution = eq(comparison.lhs.subs(x, ansatz.rhs), comparison.rhs)
    # For nonzero A evaluate the identity at t=0, then divide by -A.
    # This does not divide by cos(omega*t) at its zeros.
    frequency_squared = eq(k/m, omega**2)
    branches = sp.Tuple(eq(sp.sqrt(k/m), omega), eq(-sp.sqrt(k/m), omega))
    solution = eq(x, ansatz.rhs.subs(omega, sp.sqrt(k/m)))
    return (newton_hooke, acceleration, ode, ansatz, velocity, second,
            comparison, substitution, frequency_squared, branches, solution)


def build_proof():
    return SpringMassProof(derive())


def verify_proof(proof):
    expected = derive()
    if len(proof.steps) != 11:
        raise ValueError('Eleven source transitions required')
    for i, (actual, wanted) in enumerate(zip(proof.steps, expected)):
        if actual != wanted:
            raise ValueError(f'Step {i+1} differs')
    m, k = sp.symbols('m k', positive=True)
    t, A, omega = sp.symbols('t A omega', real=True)
    y = proof.steps[-1].rhs
    velocity, acceleration = sp.diff(y, t), sp.diff(y, t, 2)
    checks = {
        'newton_hooke_residual': sp.simplify(m*acceleration+k*y),
        'initial_displacement': sp.simplify(y.subs(t, 0)-A),
        'initial_velocity': sp.simplify(velocity.subs(t, 0)),
        'constant_energy': sp.simplify(m*velocity**2/2+k*y**2/2-k*A**2/2),
        'negative_branch_same_motion': sp.simplify(A*sp.cos(-sp.sqrt(k/m)*t)-y),
        'zero_amplitude_solution': sp.simplify(y.subs(A, 0)),
        'frequency_constraint_at_initial_time': sp.simplify(
            (proof.steps[7].lhs-proof.steps[7].rhs).subs(t, 0)
            - A*(omega**2-k/m)),
    }
    if any(value != 0 for value in checks.values()):
        raise ValueError('Independent motion certificate failed')
    return dict(source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH,
                steps_verified=11, steps=[sp.srepr(s) for s in expected],
                independent_checks={name: True for name in checks}, source_ast_parity=False,
                assumptions=['Constant positive mass and stiffness; displacement relative to equilibrium.',
                             'Net force equals linear spring restoring force; no damping or driving.',
                             'Real time and signed initial displacement A; initial velocity zero.',
                             'Positive frequency is a convention; both source square-root branches retained.'],
                limitations=['Cosine ansatz is not the general nonzero-initial-velocity solution.',
                             'For A=0 frequency cannot be inferred from motion; material parameters still define natural frequency.',
                             'Derivation uses the identity at t=0 for A nonzero; endpoint is verified separately for all A and times.'])
