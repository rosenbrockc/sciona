"""Corrected escape-energy threshold with outward-work and speed conventions."""
import sympy as sp

SOURCE_VERSION = '8d31398d-2174-589d-9624-c72abbf1a3d5'
SOURCE_HASH = '4acf6f6151a4f16c0aeecf3f9a32bb09d6661b36d45efebcac9cc67e91990960'


def build_proof():
    G, m, M, r, x, R = sp.symbols('G m M r x R', positive=True)
    return dict(external_force=G*m*M/x**2, antiderivative=-G*m*M/x,
                finite_external_work=G*m*M*(1/r-1/R),
                escape_work=G*m*M/r, surface_potential=-G*m*M/r,
                speed=sp.sqrt(2*G*M/r), negative_branch=-sp.sqrt(2*G*M/r),
                kinetic_energy=G*m*M/r)


def verify_proof(proof):
    if proof != build_proof():
        raise ValueError('Reviewed escape reconstruction changed')
    G, m, M, r, x, R = sp.symbols('G m M r x R', positive=True)
    F, A, finite, W, U, v, negative, K = [proof[k] for k in
        ['external_force','antiderivative','finite_external_work','escape_work',
         'surface_potential','speed','negative_branch','kinetic_energy']]
    checks = {
        'antiderivative_sign':sp.diff(A,x)-F,
        'finite_external_integral':sp.integrate(F,(x,r,R))-finite,
        'infinity_limit':sp.limit(finite,R,sp.oo)-W,
        'potential_reference':sp.limit(U,r,sp.oo),
        'potential_work':U+W,
        'opposing_gravity':-sp.diff(U,r)+F.subs(x,r),
        'zero_total_energy':K+U,
        'kinetic_speed':m*v**2/2-K,
        'mass_cancellation':2*K/m-v**2,
        'negative_branch_squared':negative**2-v**2,
        'positive_radial_acceleration':v*sp.diff(v,r)+G*M/r**2,
        'asymptotic_rest':sp.limit(v,r,sp.oo),
        'test_mass_independence':sp.diff(v,m),
    }
    if any(sp.simplify(value)!=0 for value in checks.values()):
        raise ValueError('Escape threshold check failed')
    if v.is_positive is not True or negative.is_negative is not True:
        raise ValueError('Branch signs not established')
    # Below the escape threshold, the conserved total energy is negative.
    fraction=sp.Symbol('fraction',positive=True)
    energy=sp.factor(m*(fraction*v)**2/2+U)
    if sp.simplify(energy-G*M*m*(fraction**2-1)/r)!=0:
        raise ValueError('Threshold energy factorization failed')
    return dict(source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,
                source_steps_reviewed=18,source_ast_parity=False,
                checks={key:True for key in checks},
                threshold_argument='For speed fraction a of threshold, total energy is GMm/r*(a²-1). For 0<a<1 it is negative, incompatible with reaching infinity where potential is zero and kinetic energy is nonnegative.',
                assumptions=['Positive G, central mass M, test mass m and center-based radius r.',
                             'Fixed exterior spherical/point-source Newtonian field with negligible test-mass backreaction and no dissipation.',
                             'Threshold has asymptotic zero terminal speed at infinity; use an outward trajectory without collision.'],
                work_semantics='Positive W is quasistatic external work against gravity. Gravitational work along outward motion is -W. Launch kinetic energy equals W.',
                branch_semantics='Positive speed magnitude; negative root is an alternative signed radial velocity, not a negative speed.',
                limitations=['No finite-time arrival at infinity, atmospheric loss, rotating-surface correction or trajectory-clearance certification.',
                             'No finite-mass relative two-body threshold or relativistic extension.',
                             'Final generic source m means central mass, distinct from cancelled test mass.'])
