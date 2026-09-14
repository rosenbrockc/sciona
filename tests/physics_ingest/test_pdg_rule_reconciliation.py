import pytest
import sympy as sp
from sciona.physics_ingest.pdg_rule_reconciliation import prove_variable_renaming


def test_three_pair_renaming_preserves_radius_and_units():
    f,m,v,r,g,n,w=sp.symbols('f m v r g n w')
    dims={'f':'M1L1T-2','g':'M1L1T-2','m':'M1','n':'M1','v':'L1T-1','w':'L1T-1','r':'L1'}
    assert prove_variable_renaming(sp.Eq(f,m*v**2/r),sp.Eq(g,n*w**2/r),[f,g,m,n,v,w],dims)=={'f':'g','m':'n','v':'w'}
    with pytest.raises(ValueError,match='exactly match'):
        prove_variable_renaming(sp.Eq(f,m*v**2/r),sp.Eq(g,n*w/r),[f,g,m,n,v,w],dims)
    dims['w']='L1'
    with pytest.raises(ValueError,match='dimensions'):
        prove_variable_renaming(sp.Eq(f,m*v**2/r),sp.Eq(g,n*w**2/r),[f,g,m,n,v,w],dims)


def test_renaming_is_simultaneous_and_requires_complete_pairs():
    x,y,z=sp.symbols('x y z');dims={str(v):'L1' for v in [x,y,z]}
    assert prove_variable_renaming(sp.Eq(x,y),sp.Eq(y,z),[x,y,y,z],dims)=={'x':'y','y':'z'}
    with pytest.raises(ValueError,match='ordered atomic'):
        prove_variable_renaming(sp.Eq(x,y),sp.Eq(y,z),[x,y,z],dims)
