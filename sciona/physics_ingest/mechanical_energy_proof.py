"""Reviewed constant-force mechanical energy derivation, with interval domains."""
import sympy as sp

SOURCE_VERSION = 'ab2e405e-eb52-54f6-b73b-f4a1450a78f9'
SOURCE_HASH = '49f52e25a02fe4e323beb9d7325542cbfbf4d6f78c6eb973886e31d1c040f624'


def build_proof():
    m = sp.Symbol('m', positive=True)
    F, t, x1, x2, v1, v2, mean, a = sp.symbols('F t x1 x2 v1 v2 mean a', real=True)
    K1, K2, U1, U2, E1, E2 = sp.symbols('K1 K2 U1 U2 E1 E2', real=True)
    eq = lambda lhs, rhs: sp.Eq(lhs, rhs, evaluate=False)
    steps = [eq(E2, K2+U2), eq(E1, K1+U1), eq(E2-E1, K2-K1+U2-U1),
             eq(K2, m*v2**2/2), eq(K1, m*v1**2/2),
             eq(K2-K1, m*(v2**2-v1**2)/2),
             eq((K2-K1)/t, m*(v2**2-v1**2)/(2*t)),
             eq(v2**2-v1**2, (v2+v1)*(v2-v1)),
             eq((K2-K1)/t, m*(v2+v1)*(v2-v1)/(2*t)),
             eq((K2-K1)/t, m*mean*(v2-v1)/t),
             eq((K2-K1)/t, m*mean*a), eq((K2-K1)/t, mean*F),
             eq(U2, -F*x2), eq(U1, -F*x1), eq(U2-U1, -F*(x2-x1)),
             eq((U2-U1)/t, -F*(x2-x1)/t), eq((U2-U1)/t, -F*mean),
             eq((E2-E1)/t, (K2-K1)/t+(U2-U1)/t),
             eq((E2-E1)/t, (K2-K1)/t-F*mean),
             eq((E2-E1)/t, sp.Add(mean*F, -F*mean, evaluate=False)),
             eq((E2-E1)/t, 0), eq(E2-E1, 0), eq(E2, E1)]
    velocity = v1+F*t/m
    position = x1+v1*t+F*t**2/(2*m)
    return dict(velocity=velocity, position=position, steps=steps)


def verify_proof(proof):
    if proof != build_proof():
        raise ValueError('Reviewed reconstruction changed')
    m = sp.Symbol('m', positive=True)
    F, t, x1, x2, v1, v2, mean, a = sp.symbols('F t x1 x2 v1 v2 mean a', real=True)
    K1, K2, U1, U2, E1, E2 = sp.symbols('K1 K2 U1 U2 E1 E2', real=True)
    V, X = proof['velocity'], proof['position']
    values = {v2: V, x2: X, mean: v1+F*t/(2*m), a: F/m,
              K1: m*v1**2/2, K2: m*V**2/2, U1: -F*x1, U2: -F*X,
              E1: m*v1**2/2-F*x1, E2: m*V**2/2-F*X}
    for i, equation in enumerate(proof['steps'], 1):
        residual = (equation.lhs-equation.rhs).subs(values, simultaneous=True)
        if sp.cancel(residual) != 0:
            raise ValueError('Energy identity failed at step '+str(i))
    s, z = sp.symbols('s z', real=True)
    energy = m*V**2/2-F*X
    checks = {'newton_law': m*sp.diff(V, t)-F,
              'position_derivative': sp.diff(X, t)-V,
              'velocity_integral': v1+sp.integrate(F/m, (s, 0, t))-V,
              'position_integral': x1+sp.integrate(v1+F*s/m, (s, 0, t))-X,
              'endpoint_mean_integral': sp.integrate(v1+F*s/m, (s, 0, t))-(v1+V)*t/2,
              'potential_gradient': -sp.diff(-F*z, z)-F,
              'work_energy': sp.integrate(F, (z, x1, X))-(m*V**2/2-m*v1**2/2),
              'conserved_energy_derivative': sp.diff(energy, t),
              'initial_energy': energy.subs(t, 0)-(m*v1**2/2-F*x1),
              'zero_duration_position': X.subs(t, 0)-x1,
              'zero_duration_velocity': V.subs(t, 0)-v1,
              'zero_force_energy': energy.subs(F, 0)-m*v1**2/2}
    if any(sp.cancel(r) != 0 for r in checks.values()):
        raise ValueError('Independent integral/derivative check failed')
    return dict(source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH,
                source_steps_reviewed=23, reconstructed_steps_validated=23,
                source_ast_parity=False, integration_checks={k: True for k in checks},
                assumptions=['Positive constant mass; finite real signed initial position, velocity and constant net force.',
                             'One-dimensional Newtonian motion with acceleration F/m and duration t>=0.',
                             'Time-independent potential U=-F*x; no additional work or dissipation.',
                             'The v in finite-interval power equations is average velocity, not endpoint velocity.',
                             'Quotient steps require t>0.'],
                endpoint_extensions='Polynomial trajectories prove t=0 independently; F=0 and velocity reversal are allowed.',
                limitations=['No variable-force, time-dependent-potential, dissipative or relativistic result.',
                             'The potential zero is chosen at x=0; an additive constant shifts both energies equally.',
                             'Conservation is an exact model identity; rounded numerical outputs need not preserve it exactly.'])
