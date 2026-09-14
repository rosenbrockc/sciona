"""Time-based ideal projectile motion with explicit integration constants."""
import sympy as sp

SOURCE_VERSION = '8b6af03c-7244-559f-98a1-f41d23eda79e'
SOURCE_HASH = '3f705e752e04b90c50b873122ce1a2efedc797ea471707cd793b29747e8903bd'


def build_proof():
    x0, y0, theta, t = sp.symbols('x0 y0 theta t', real=True)
    speed, g = sp.symbols('speed g', nonnegative=True)
    ux, uy = speed*sp.cos(theta), speed*sp.sin(theta)
    return dict(x=x0+ux*t, y=y0+uy*t-g*t**2/2, vx=ux, vy=uy-g*t)


def verify_proof(proof):
    if proof != build_proof():
        raise ValueError('Reviewed reconstruction changed')
    x0, y0, theta, t = sp.symbols('x0 y0 theta t', real=True)
    speed, g = sp.symbols('speed g', nonnegative=True)
    ux, uy = speed*sp.cos(theta), speed*sp.sin(theta)
    x, y, vx, vy = (proof[k] for k in ['x', 'y', 'vx', 'vy'])
    s = sp.Symbol('s', real=True)
    checks = dict(horizontal_velocity_integral=ux+sp.integrate(0, (s, 0, t))-vx,
        vertical_velocity_integral=uy+sp.integrate(-g, (s, 0, t))-vy,
        horizontal_position_integral=x0+sp.integrate(vx.subs(t, s), (s, 0, t))-x,
        vertical_position_integral=y0+sp.integrate(vy.subs(t, s), (s, 0, t))-y,
        horizontal_position_derivative=sp.diff(x, t)-vx,
        vertical_position_derivative=sp.diff(y, t)-vy,
        horizontal_acceleration=sp.diff(vx, t), vertical_acceleration=sp.diff(vy, t)+g,
        initial_x=x.subs(t, 0)-x0, initial_y=y.subs(t, 0)-y0,
        initial_vx=vx.subs(t, 0)-ux, initial_vy=vy.subs(t, 0)-uy,
        initial_speed=ux**2+uy**2-speed**2,
        zero_gravity_height=y.subs(g, 0)-(y0+uy*t),
        zero_speed_height=y.subs(speed, 0)-(y0-g*t**2/2),
        vertical_launch_x=x.subs(theta, sp.pi/2)-x0,
        mechanical_energy=(vx**2+vy**2)/2+g*y-(speed**2/2+g*y0))
    if any(sp.simplify(value) != 0 for value in checks.values()):
        raise ValueError('Projectile integration or boundary identity failed')
    return dict(source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH,
                source_steps_reviewed=30, source_ast_parity=False,
                reconstructed_checks={k: True for k in checks},
                assumptions=['Fixed orthonormal Cartesian basis, x horizontal and y positive upward.',
                             'Finite initial position, nonnegative initial speed, real launch angle measured from positive x.',
                             'Constant nonnegative downward gravity and nonnegative elapsed time; no drag.',
                             'Initial position and velocity are specified at t=0.',
                             'Source sine/cosine quotient prerequisites require speed>0; component products extend to speed=0.'],
                limitations=['Ideal point-particle constant-gravity motion; no terrain or collision cutoff.',
                             'Zero gravity is the inertial extension; no variable gravity or drag model.',
                             'All launch directions and vertical motion included; no horizontal-velocity division.',
                             'Coordinates are signed positions, not path length; angle is irrelevant at zero speed.',
                             'Exact model energy identity does not guarantee equality after independent numerical rounding.'])
