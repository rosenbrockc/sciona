import pytest
import sympy as sp
from sciona.physics_ingest.infall_speed_proof import build_proof, verify_proof


def test_integral_energy_and_branch_checks():
    report = verify_proof(build_proof())
    assert len(report['checks']) == 13 and all(report['checks'].values())
    assert not report['source_ast_parity']


@pytest.mark.parametrize('key', list(build_proof()))
def test_mutated_certificate_rejected(key):
    proof = build_proof()
    proof[key] += 1
    with pytest.raises(ValueError):
        verify_proof(proof)


def test_outward_source_force_gives_wrong_work_sign():
    x = sp.Symbol('x', positive=True)
    assert sp.integrate(1/x**2, (x, sp.oo, 1)) == -1


def test_finite_release_changes_speed():
    proof = build_proof()
    G, m, M, r, R = sp.symbols('G m M r R', positive=True)
    assert sp.simplify(2*proof['finite_work']/m-proof['speed']**2) == -2*G*M/R


def test_both_radial_branches_are_not_simultaneous():
    proof = build_proof()
    assert proof['speed'].is_positive
    assert proof['inward_radial_velocity'].is_negative
    assert sp.simplify(proof['speed']+proof['inward_radial_velocity']) == 0


def test_finite_two_body_relative_speed_is_different():
    G, M, m, r = sp.symbols('G M m r', positive=True)
    relative_squared = 2*G*(M+m)/r
    assert sp.simplify(relative_squared-build_proof()['speed']**2) == 2*G*m/r
