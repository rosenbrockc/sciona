import pytest
import sympy as sp
from sciona.physics_ingest.lorentz_boost_proof import build_proof,verify_proof


def test_metric_and_all_coefficients():
    r=verify_proof(build_proof())
    assert len(r['checks'])==10 and all(r['checks'].values())
    assert not r['source_ast_parity']


@pytest.mark.parametrize('key',list(build_proof()))
def test_certificate_mutations(key):
    p=build_proof()
    if key=='boost':p[key][0,1]=-p[key][0,1]
    else:p[key]+=1
    with pytest.raises(ValueError):verify_proof(p)


def test_x_coefficient_alone_has_extraneous_branch():
    p=build_proof();S=sp.Symbol('gamma_squared',positive=True)
    b=sp.Symbol('beta',real=True);c=sp.Symbol('c',positive=True)
    assert p['source_x_coefficient'].subs(S,1)==1
    assert sp.simplify(p['source_xt_coefficient'].subs(S,1))==-2*b*c
    assert sp.simplify(p['source_time_coefficient'].subs(S,1)-c*c)==-b*b*c*c


def test_negative_gamma_not_identity_at_rest():
    p=build_proof();b,g=sp.symbols('beta gamma',real=True)
    assert p['boost'].subs({b:0,g:-1})!=sp.eye(4)


def test_subluminal_domain_is_necessary():
    b=sp.Symbol('beta',real=True);q=build_proof()['gamma_squared']
    assert q.subs(b,2)<0 and q.subs(b,1)==sp.zoo


def test_mixed_sign_boost_breaks_interval():
    b,g=sp.symbols('beta gamma',real=True);B=build_proof()['boost']
    B[0,1]*=-1
    residual=B.T*sp.diag(1,-1,-1,-1)*B-sp.diag(1,-1,-1,-1)
    assert sp.expand(residual[0,1])==2*b*g*g
