from dataclasses import replace

import pytest
import sympy as sp

from sciona.physics_ingest.two_body_corrected_proof import build_proof, symbols, verify_proof


def test_composed_proof():
    result = verify_proof(build_proof())
    assert all(result['checks'].values())
    assert not result['source_parity_claim']
    assert not result['pi_interpretation']


@pytest.mark.parametrize('index', range(13))
def test_each_intermediate_is_checked(index):
    proof = build_proof()
    steps = list(proof.steps)
    steps[index] = sp.Eq(steps[index].lhs, 2*steps[index].rhs, evaluate=False)
    with pytest.raises(ValueError, match='Invalid corrected proof'):
        verify_proof(replace(proof, steps=tuple(steps)))


@pytest.mark.parametrize('index', range(6))
def test_each_premise_is_checked(index):
    proof = build_proof()
    premises = list(proof.premises)
    premises[index] = sp.Eq(premises[index].lhs, 2*premises[index].rhs, evaluate=False)
    with pytest.raises(ValueError, match='premise'):
        verify_proof(replace(proof, premises=tuple(premises)))


def test_canceled_unproved_denominator_is_rejected():
    proof = build_proof()
    s = symbols()
    factor = s['r']-s['d1']
    rhs = sp.Mul(proof.steps[-1].rhs, factor, sp.Pow(factor, -1, evaluate=False), evaluate=False)
    steps = (*proof.steps[:-1], sp.Eq(proof.steps[-1].lhs, rhs, evaluate=False))
    with pytest.raises(ValueError, match='positivity'):
        verify_proof(replace(proof, steps=steps))


@pytest.mark.parametrize('m1,m2,d2,G,p', [(2, 3, 4, 5, 1), (7, 1, 2, 3, 2), (1, 1, 1, 1, 3)])
def test_exact_positive_witness_satisfies_every_equation(m1, m2, d2, G, p):
    proof = build_proof()
    s = symbols()
    m1, m2, d2, G, p = map(sp.Rational, (m1, m2, d2, G, p))
    d1 = m2*d2/m1
    r = d1+d2
    period = sp.sqrt(4*p**2*r**3/(G*(m1+m2)))
    omega = 2*p/period
    force = G*m1*m2/r**2
    values = dict(m1=m1, m2=m2, d1=d1, d2=d2, r=r, G=G, p=p, T=period,
                  w=omega, F=force, Fc=force, m=m2*d2/r)
    assignment = {s[k]: v for k, v in values.items()}
    for equation in (*proof.premises, *proof.steps):
        assert sp.simplify((equation.lhs-equation.rhs).subs(assignment)) == 0
