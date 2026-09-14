import pytest
import sympy as sp
from sciona.physics_ingest.residual_equivalence import equivalent_residual_mapping


def test_renamed_ohmic_residuals_share_arithmetic():
    v,i,r,w,j,s=sp.symbols('v i r w j s')
    mapping=equivalent_residual_mapping(sp.Eq(v,i*r),sp.Eq(w,j*s),
        {'v':'M1L2T-3I-1','i':'I1','r':'M1L2T-3I-2'},
        {'w':'M1L2T-3I-1','j':'I1','s':'M1L2T-3I-2'})
    assert mapping=={'v':'w','i':'j','r':'s'}


def test_equivalence_does_not_confuse_units_scaling_or_sign():
    a,b,c=sp.symbols('a b c')
    dims={str(x):'L1' for x in [a,b,c]}
    original=sp.Eq(a,b+c)
    assert equivalent_residual_mapping(original,original,dims,{str(x):'T1' for x in [a,b,c]}) is None
    assert equivalent_residual_mapping(original,sp.Eq(a,2*b+c),dims,dims) is None
    assert equivalent_residual_mapping(original,sp.Eq(b+c,a),dims,dims) is None
    with pytest.raises(ValueError,match='complete dimensions'):
        equivalent_residual_mapping(original,original,{},dims)
