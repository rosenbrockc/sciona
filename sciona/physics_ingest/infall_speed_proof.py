"""Corrected work-energy reconstruction for radial infall from rest at infinity."""
import sympy as sp

SOURCE_VERSION = 'e70acd5a-17a6-5e30-b307-78d427d5a207'
SOURCE_HASH = '79f180692b324066977be6ec88a4e526321e8526c3b437bbde5b31985ab643d6'


def build_proof():
    G, m, M, r, x, R = sp.symbols('G m M r x R', positive=True)
    speed = sp.sqrt(2*G*M/r)
    return dict(radial_force=-G*m*M/x**2,
                antiderivative=G*m*M/x,
                finite_work=G*m*M*(1/r-1/R),
                work_from_infinity=G*m*M/r,
                potential=-G*m*M/r,
                speed=speed, inward_radial_velocity=-speed,
                specific_kinetic_energy=G*M/r)


def verify_proof(proof):
    if proof != build_proof():
        raise ValueError('Reviewed infall reconstruction changed')
    G, m, M, r, x, R = sp.symbols('G m M r x R', positive=True)
    force, primitive, finite, work, potential, speed, inward, specific = (
        proof[k] for k in ['radial_force', 'antiderivative', 'finite_work',
                          'work_from_infinity', 'potential', 'speed',
                          'inward_radial_velocity', 'specific_kinetic_energy'])
    checks = {
        'antiderivative': sp.diff(primitive, x)-force,
        'finite_radial_integral': sp.integrate(force, (x, R, r))-finite,
        'improper_integral_limit': sp.limit(finite, R, sp.oo)-work,
        'potential_force_sign': -sp.diff(potential, r)-force.subs(x, r),
        'zero_potential_at_infinity': sp.limit(potential, r, sp.oo),
        'work_potential_difference': work+potential,
        'kinetic_work_equality': m*speed**2/2-work,
        'specific_kinetic_energy': speed**2/2-specific,
        'zero_total_energy': m*speed**2/2+potential,
        'inward_branch_squared': inward**2-speed**2,
        'radial_acceleration': inward*sp.diff(inward, r)+G*M/r**2,
        'speed_boundary_at_infinity': sp.limit(speed, r, sp.oo),
        'test_mass_independence': sp.diff(speed, m),
    }
    if any(sp.simplify(value) != 0 for value in checks.values()):
        raise ValueError('Infall proof check failed')
    if speed.is_positive is not True or inward.is_negative is not True:
        raise ValueError('Radial branch signs not established')
    return dict(source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH,
                source_steps_reviewed=17, source_ast_parity=False,
                checks={key: True for key in checks},
                assumptions=['Positive G, fixed central mass M, radius r and negligible positive test mass m.',
                             'Newtonian point source or exterior spherical source; inward radial motion, no other forces.',
                             'Rest at infinity means the asymptotic zero-total-energy boundary, not launch at infinity at finite time.'],
                branch_semantics='Positive speed sqrt(2GM/r); infall radial velocity is its negative. The two square roots are alternatives.',
                work_semantics='Positive gravitational work on the particle from infinity; potential energy is negative with zero at infinity.',
                limitations=['No finite-mass two-body relative speed, drag, general extended-body or relativistic claim.',
                             'Escape-speed magnitude uses time reversal in the same ideal field, not a collision-free trajectory guarantee.',
                             'Original force sign, malformed ASTs and work aliases require correction; literal source parity is false.'])
