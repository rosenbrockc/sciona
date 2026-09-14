import pytest
import sympy as sp
from sciona.ghost.symbolic import serialize_expr
from sciona.ghost.dimensions import DimensionalSignature
from sciona.physics_ingest.pdg_symbols import PdgScalarDefinition
from sciona.physics_ingest.source_identity_normalization import normalize_source_identities


def test_composite_labels_are_metadata_and_identities_remain_distinct():
    first,second=sp.symbols('pdg0000001 pdg0000002')
    source=serialize_expr(sp.Eq(first,second,evaluate=False))
    dimension=DimensionalSignature(M=1,L=2,T=-2)
    defs={str(s):PdgScalarDefinition(str(i), 'KE_1',dimension) for i,s in enumerate([first,second])}
    normalized,labels=normalize_source_identities(source,defs)
    assert normalized.srepr_str==source
    assert set(normalized.variables)=={str(first),str(second)}
    assert list(labels.values())==['KE_1','KE_1']


@pytest.mark.parametrize('case',['missing','unknown_dimension','dimension_mismatch'])
def test_incomplete_or_inconsistent_source_fails(case):
    a,b=sp.symbols('pdg0000001 pdg0000002')
    defs={str(s):PdgScalarDefinition(str(i),'label',DimensionalSignature(L=1)) for i,s in enumerate([a,b])}
    if case=='missing':del defs[str(b)]
    elif case=='unknown_dimension':defs[str(b)]=PdgScalarDefinition('2','label',None)
    else:defs[str(b)]=PdgScalarDefinition('2','label',DimensionalSignature(T=1))
    with pytest.raises(ValueError):normalize_source_identities(serialize_expr(sp.Eq(a,b,evaluate=False)),defs)


def test_singular_denominator_is_retained():
    a,b=sp.symbols('pdg0000001 pdg0000002')
    with sp.evaluate(False):expr=sp.Eq(a/a,b)
    source=serialize_expr(expr)
    defs={str(s):PdgScalarDefinition(str(i),'label',DimensionalSignature()) for i,s in enumerate([a,b])}
    normalized,_=normalize_source_identities(source,defs)
    assert normalized.srepr_str==source
