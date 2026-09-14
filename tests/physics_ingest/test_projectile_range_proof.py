import pytest
import sympy as sp
from sciona.physics_ingest.projectile_range_proof import build_proof, verify_proof


def test_reconstruction_and_global_bound():
    report = verify_proof(build_proof())
    assert len(report['checks']) == 13 and all(report['checks'].values())
    assert not report['source_ast_parity']


@pytest.mark.parametrize('key', list(build_proof()))
def test_mutated_certificate_rejected(key):
    proof = build_proof()
    proof[key] += 1
    with pytest.raises(ValueError):
        verify_proof(proof)


def test_division_loses_launch_root():
    t = sp.Symbol('t', real=True)
    assert sp.solve(2*t-t*t, t) == [0, 2]
    assert sp.solve(2-t, t) == [2]


def test_unequal_height_invalidates_level_ground_time():
    proof = build_proof()
    t = sp.Symbol('t', real=True)
    h = sp.Symbol('h', positive=True)
    assert sp.simplify((h+proof['vertical']).subs(t, proof['flight_time'])) == h


def test_angle_is_not_maximum_without_speed_constraint():
    theta = sp.Symbol('theta', real=True)
    # Choosing v(theta)^2=exp(theta) changes the stationary point.
    distance = sp.exp(theta)*sp.sin(2*theta)
    assert sp.diff(distance, theta).subs(theta, sp.pi/4) == sp.exp(sp.pi/4)


def test_complementary_angles_and_unique_square_zero():
    theta = sp.Symbol('theta', real=True)
    gap = (sp.sin(theta)-sp.cos(theta))**2
    assert sp.trigsimp(gap-(1-sp.sin(2*theta))) == 0
    assert sp.solveset(sp.sin(theta)-sp.cos(theta), theta,
                       domain=sp.Interval(0, sp.pi/2)) == sp.FiniteSet(sp.pi/4)
