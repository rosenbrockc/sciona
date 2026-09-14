"""Conditional Newtonian force law; source inference gaps are not certified."""
import sympy as sp
SOURCE_VERSION='abf1f483-647a-5634-8ddb-45924283770e'
SOURCE_HASH='5312a9c81cb0290bd4eed9101bf7c190121c2495c7f73784daa9180be7695409'


def build_proof():
    G,m1,m2,r,T,C=sp.symbols('G m1 m2 r T C',positive=True)
    return dict(force=G*m1*m2/r**2,
                circular_force=4*sp.pi**2*m2*r/T**2,
                period_squared=C*r**3,
                kepler_coefficient=4*sp.pi**2/(G*m1))


def verify_proof(proof):
    expected=build_proof()
    if proof!=expected:raise ValueError('Conditional reconstruction changed')
    G,m1,m2,r,T,C=sp.symbols('G m1 m2 r T C',positive=True)
    law=proof['force'];circular=proof['circular_force']
    conditional=sp.cancel(circular.subs(T**2,proof['period_squared']).subs(C,proof['kepler_coefficient']))
    checks={
        'conditional_kepler_reconstruction':sp.cancel(conditional-law),
        'mass_exchange_symmetry':sp.cancel(law.xreplace({m1:m2,m2:m1})-law),
        'test_mass_acceleration':sp.cancel(law/m2-G*m1/r**2),
        'inverse_square_scaling':sp.cancel(law.subs(r,2*r)-law/4),
        'potential_force':sp.cancel(-sp.diff(-G*m1*m2/r,r)+law),
    }
    if any(v!=0 for v in checks.values()):raise ValueError('Conditional force check failed')
    # Holding T fixed gives F(2r)=2F(r), contradicting inferred inverse-square.
    if sp.cancel(circular.subs(r,2*r)/circular)!=2:raise ValueError('Counterexample changed')
    return dict(source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,
                source_steps_reviewed=13,source_derivation_valid_without_added_assumptions=False,
                source_ast_parity=False,conditional_checks={k:True for k in checks},
                conditional_force_srepr=sp.srepr(law),
                gaps=['F=ma implies proportionality to m only when acceleration is held independent of that mass.',
                      'Renaming m to two different masses does not establish independent proportionality of the same force to both.',
                      'Circular kinematics requires an additional Kepler T^2 proportional to r^3 premise to infer inverse-square dependence.',
                      'Universal coefficient G and its independence from masses/radius are empirical/model assumptions.'],
                source_counterexample='At fixed positive period, circular force is proportional to radius, so doubling radius doubles force.',
                assumptions=['Newtonian point masses or exterior non-overlapping spherical bodies; positive masses, separation and G.',
                             'Gravitational force magnitude is G*m1*m2/r^2; this constitutive law is assumed, not proved from F=ma.',
                             'Circular comparison uses supplied Kepler coefficient4*pi^2/(G*m1) in the negligible-test-mass limit.'],
                limitations=['No proof of universal gravitation from Newton second law alone.',
                             'No general extended-body, relativistic or signed-vector force claim.'])
