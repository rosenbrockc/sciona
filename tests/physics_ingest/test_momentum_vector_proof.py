from dataclasses import replace
import sympy as sp
import pytest
from sciona.physics_ingest.momentum_vector_proof import build_proof, verify_proof, vectors


def test_proof_and_nonorthogonal_cross_term():
    proof = build_proof()
    assert all(verify_proof(proof)['checks'].values())
    v = vectors()
    mapping = dict(zip(v['p1'], [3, 4, 0])) | dict(zip(v['p2'], [1, 2, 2])) | dict(zip(v['electron'], [2, 2, -2]))
    assert proof.norm_squared.lhs.subs(mapping) == proof.norm_squared.rhs.subs(mapping) == 12


@pytest.mark.parametrize('step', range(4))
@pytest.mark.parametrize('component', range(3))
def test_each_vector_component_checked(step, component):
    proof = build_proof()
    steps = [list(e) for e in proof.steps]
    e = steps[step][component]
    steps[step][component] = sp.Eq(e.lhs, e.rhs+1, evaluate=False)
    with pytest.raises(ValueError, match='Invalid vector proof'):
        verify_proof(replace(proof, steps=tuple(tuple(e) for e in steps)))


@pytest.mark.parametrize('index', range(3))
def test_each_premise_checked(index):
    proof = build_proof()
    premises = list(proof.premises)
    premises[index] = tuple(sp.Eq(e.lhs, e.rhs+1, evaluate=False) for e in premises[index])
    with pytest.raises(ValueError):
        verify_proof(replace(proof, premises=tuple(premises)))


def test_missing_cross_term_rejected():
    proof = build_proof()
    v = vectors()
    wrong = sp.Eq(proof.norm_squared.lhs, v['p1'].dot(v['p1'])+v['p2'].dot(v['p2']), evaluate=False)
    with pytest.raises(ValueError, match='dot-product'):
        verify_proof(replace(proof, norm_squared=wrong))


def test_nonreal_or_unknown_component_rejected():
    proof = build_proof()
    wrong = sp.Eq(sp.Symbol('unknown'), proof.norm_squared.rhs, evaluate=False)
    with pytest.raises(ValueError, match='Real scalar'):
        verify_proof(replace(proof, norm_squared=wrong))


@pytest.mark.parametrize('p1,p2', [([0, 0, 0], [0, 0, 0]), ([3, 4, 0], [1, 2, 2]), ([-1, 2, -3], [4, -5, 6])])
def test_exact_witnesses(p1, p2):
    proof, v = build_proof(), vectors()
    p1, p2 = sp.Matrix(p1), sp.Matrix(p2)
    mapping = {s: value for name, vector in dict(p1=p1, p2=p2, electron=p1-p2, before=p1, after=p1).items() for s, value in zip(v[name], vector)}
    equations = [e for eqs in (*proof.premises, *proof.steps) for e in eqs]+[proof.norm_squared]
    assert all(sp.expand((e.lhs-e.rhs).subs(mapping)) == 0 for e in equations)
