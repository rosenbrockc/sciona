import pytest
import sympy as sp
from sciona.physics_ingest.hyperbolic_identities_proof import build_proof, verify_proof


def test_all_steps_and_complex_definitions():
    report = verify_proof(build_proof())
    assert report['reconstructed_steps_validated'] == 18
    assert report['complex_definition_checks'] == 6 and report['denominator_positive']
    assert report['replay_order'].index(8) < report['replay_order'].index(7)


@pytest.mark.parametrize('index', range(18))
def test_each_mutated_step_rejected(index):
    proof = build_proof()
    e = proof[index]
    proof[index] = sp.Eq(e.lhs, e.rhs+1, evaluate=False)
    with pytest.raises(ValueError):
        verify_proof(proof)


def test_generic_named_i_does_not_square_to_minus_one():
    i = sp.Symbol('i')
    assert sp.simplify(i**2+1) != 0
    assert sp.I**2+1 == 0


def test_complex_quotient_pole_excluded():
    z = sp.I*sp.pi/2
    assert sp.cosh(z) == 0
    assert sp.sech(z) == sp.zoo


def test_zero_argument_boundary():
    x = sp.Symbol('x', real=True)
    for e in build_proof():
        assert sp.simplify((e.lhs-e.rhs).subs(x, 0)) == 0
