"""Synthetic boundaries and error detection for full time-based projectiles."""
import pytest
import sympy as sp
from sciona.physics_ingest.projectile_motion_proof import build_proof, verify_proof


def test_integrals_derivatives_initial_conditions_and_energy():
    report = verify_proof(build_proof())
    assert len(report['reconstructed_checks']) == 17
    assert all(report['reconstructed_checks'].values())
    assert report['source_ast_parity'] is False


def test_missing_integration_constant_rejected():
    proof = build_proof()
    proof['y'] -= sp.Symbol('y0', real=True)
    with pytest.raises(ValueError, match='reconstruction changed'):
        verify_proof(proof)


def test_vertical_launch_needs_no_horizontal_division():
    proof = build_proof()
    theta, t, x0 = sp.symbols('theta t x0', real=True)
    speed, g = sp.symbols('speed g', nonnegative=True)
    assert sp.simplify(proof['x'].subs(theta, sp.pi/2)-x0) == 0
    assert proof['vx'].subs(theta, sp.pi/2) == 0
    assert proof['vy'].subs({theta: sp.pi/2, speed: 2, g: 1, t: 3}) == -1


def test_zero_speed_angle_independence():
    speed = sp.Symbol('speed', nonnegative=True)
    theta = sp.Symbol('theta', real=True)
    assert all(not v.subs(speed, 0).has(theta) for v in build_proof().values())


def test_backward_downward_launch():
    theta, t, x0, y0 = sp.symbols('theta t x0 y0', real=True)
    speed, g = sp.symbols('speed g', nonnegative=True)
    values = {theta: -3*sp.pi/4, speed: sp.sqrt(2), t: 2, g: 1, x0: 3, y0: 4}
    actual = {k: sp.simplify(v.subs(values)) for k, v in build_proof().items()}
    assert actual == dict(x=1, y=0, vx=-1, vy=-3)


def test_time_varying_gravity_not_constant_gravity_solution():
    t, y0, uy = sp.symbols('t y0 uy', real=True)
    # a_y=-t integrates to y0+uy*t-t^3/6, not the g=t substitution below.
    naive = y0+uy*t-t*t**2/2
    assert sp.diff(naive, t, 2) != -t


def test_inverted_source_trigonometric_quotient():
    speed, theta = sp.Integer(2), sp.pi/3
    vx = speed*sp.cos(theta)
    assert vx/speed == sp.cos(theta)
    assert speed/vx != sp.cos(theta)
