"""Level-ground range reconstruction with explicit branch and maximum checks."""
import sympy as sp

SOURCE_VERSION = 'e416bcf7-e977-5da2-b136-b9927f3236ef'
SOURCE_HASH = '664efab7753078323d98d955c7c6ba727aae9622e948a31ecf1d4a5fcd5f549a'


def build_proof():
    v, g = sp.symbols('v g', positive=True)
    t, theta = sp.symbols('t theta', real=True)
    return dict(vertical=v*t*sp.sin(theta)-g*t**2/2,
                horizontal=v*t*sp.cos(theta),
                flight_time=2*v*sp.sin(theta)/g,
                range=v**2*sp.sin(2*theta)/g,
                maximizing_angle=sp.pi/4, maximum_range=v**2/g,
                maximum_gap=v**2/g*(sp.sin(theta)-sp.cos(theta))**2)


def verify_proof(proof):
    if proof != build_proof():
        raise ValueError('Reviewed reconstruction changed')
    v, g = sp.symbols('v g', positive=True)
    t, theta = sp.symbols('t theta', real=True)
    y, x, flight, distance = (proof[k] for k in
                             ('vertical', 'horizontal', 'flight_time', 'range'))
    residuals = {
        'vertical_acceleration': sp.diff(y, t, 2)+g,
        'horizontal_acceleration': sp.diff(x, t, 2),
        'initial_height': y.subs(t, 0),
        'initial_vertical_velocity': sp.diff(y, t).subs(t, 0)-v*sp.sin(theta),
        'initial_horizontal_velocity': sp.diff(x, t).subs(t, 0)-v*sp.cos(theta),
        'two_flight_roots': sp.expand(y+g*t*(t-flight)/2),
        'landing_height': y.subs(t, flight),
        'range_from_landing': x.subs(t, flight)-distance,
        'stationary_at_maximum': sp.diff(distance, theta).subs(theta, proof['maximizing_angle']),
        'maximum_attained': distance.subs(theta, proof['maximizing_angle'])-proof['maximum_range'],
        'global_square_gap': proof['maximum_range']-distance-proof['maximum_gap'],
        'horizontal_endpoint': distance.subs(theta, 0),
        'vertical_endpoint': distance.subs(theta, sp.pi/2),
    }
    if any(sp.trigsimp(sp.expand(r)) != 0 for r in residuals.values()):
        raise ValueError('Projectile reconstruction failed')
    return dict(source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH,
                source_steps_reviewed=17, source_ast_parity=False,
                checks={key: True for key in residuals},
                assumptions=['Fixed positive launch speed and positive constant gravity.',
                             'No drag; flat ground; landing height equals launch height.',
                             'Angle in radians, 0<theta<pi/2 for positive-flight derivation.'],
                maximum_justification='The gap is a positive coefficient times a real square; equality in the closed first quadrant occurs only at pi/4.',
                boundary_semantics='At theta=0 the positive flight root merges with launch; zero range is a continuous extension. At pi/2 positive flight remains but horizontal range is zero.',
                limitations=['No unequal-height, drag, curved-earth or variable-gravity optimization.',
                             'Source boundary conditions and reversed substitution feed require explicit correction.'])
