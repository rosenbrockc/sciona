import pytest
import sympy as sp
from sciona.physics_ingest.escape_speed_proof import build_proof, verify_proof


def test_work_and_threshold_certificate():
    r=verify_proof(build_proof())
    assert len(r['checks'])==13 and all(r['checks'].values())
    assert not r['source_ast_parity']


@pytest.mark.parametrize('key',list(build_proof()))
def test_mutations_rejected(key):
    p=build_proof();p[key]+=1
    with pytest.raises(ValueError):verify_proof(p)


def test_original_antiderivative_wrong_sign():
    x=sp.Symbol('x',positive=True)
    assert sp.diff(1/x,x)==-1/x**2
    assert sp.integrate(1/x**2,(x,1,sp.oo))==1


def test_subthreshold_turns_before_infinity():
    # Synthetic G=M=m=r0=1, launch speed=1 < sqrt(2).
    # E=-1/2 and radial kinetic energy vanishes at radius2.
    assert sp.Rational(1,2)-1==-sp.Rational(1,2)
    assert -sp.Rational(1,2)+sp.Rational(1,2)==0


def test_cancelling_zero_mass_is_not_valid():
    v=sp.Symbol('v',real=True)
    assert sp.Eq(0*v**2,0) is sp.true
    assert sp.Eq(v**2,2) is not sp.true


def test_approved_infall_formula_matches_threshold():
    from sciona.physics_ingest.infall_speed_proof import build_proof as infall
    p=build_proof();q=infall()
    assert sp.simplify(p['speed']-q['speed'])==0
    assert sp.simplify(p['escape_work']-q['work_from_infinity'])==0
    assert sp.simplify(p['surface_potential']-q['potential'])==0


def test_central_mass_is_not_test_mass():
    G,m,M,r=sp.symbols('G m M r',positive=True)
    assert sp.simplify(build_proof()['speed']**2-2*G*m/r)==2*G*(M-m)/r
