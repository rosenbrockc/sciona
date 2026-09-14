from dataclasses import replace
import pytest
import sympy as sp
from sciona.physics_ingest.quadratic_corrected_proof import build_proof,verify_proof,symbols


def test_complete_corrected_proof():
    report=verify_proof(build_proof())
    assert all(report['checks'].values())
    assert report['source_parity_claim'] is False


@pytest.mark.parametrize('coefficients',[(2,3,1),(-2,3,1),(1,2,1),(-1,2,-1),(3,0,-12),(2,3,0)])
def test_both_coefficient_signs_repeated_and_zero_root(coefficients):
    p=build_proof();a,b,c,x=symbols();values=dict(zip((a,b,c),coefficients))
    lo=sp.simplify(p.lower.rhs.subs(values));hi=sp.simplify(p.upper.rhs.subs(values))
    assert sp.simplify(hi-lo).is_nonnegative
    for root in [lo,hi]:assert sp.simplify(p.original.lhs.subs(values).subs(x,root))==0


def test_source_missing_linear_coefficient_rejected():
    p=build_proof();a,b,c,x=symbols()
    with pytest.raises(ValueError,match='division'):
        verify_proof(replace(p,divided=sp.Eq(x*x+x+c/a,0,evaluate=False)))


def test_source_swapped_branch_rejected():
    p=build_proof()
    with pytest.raises(ValueError,match='negative branch offset'):
        verify_proof(replace(p,lower=p.upper))


def test_dropping_absolute_value_rejected():
    p=build_proof();a,b,c,x=symbols()
    bad=sp.Eq(x,(-b-sp.sqrt(b*b-4*a*c))/(2*a),evaluate=False)
    with pytest.raises(ValueError):verify_proof(replace(p,lower=bad))
