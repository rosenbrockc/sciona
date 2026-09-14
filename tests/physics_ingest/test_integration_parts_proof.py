from dataclasses import replace
import sympy as sp
import pytest
from sciona.physics_ingest.integration_parts_proof import build_proof,verify_proof,symbols


def test_parametrized_product_rule_and_antiderivative():
    assert all(verify_proof(build_proof())['checks'].values())


@pytest.mark.parametrize('index',range(3))
def test_corrupted_step(index):
    proof=build_proof();steps=list(proof.steps);steps[index]=sp.Eq(sp.Symbol('bad'),0,evaluate=False)
    with pytest.raises(ValueError):verify_proof(replace(proof,steps=tuple(steps)))


def test_product_rule_must_be_a_premise():
    proof=build_proof()
    with pytest.raises(ValueError):verify_proof(replace(proof,premise=sp.Eq(0,0,evaluate=False)))


def test_variable_dependent_integration_constant_rejected():
    proof=build_proof();x,u,v,c=symbols();steps=list(proof.steps)
    steps[2]=sp.Eq(steps[2].lhs,steps[2].rhs.subs(c,x),evaluate=False)
    with pytest.raises(ValueError):verify_proof(replace(proof,steps=tuple(steps)))
