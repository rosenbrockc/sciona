import pytest
import sympy as sp
from sciona.physics_ingest.pdg_symbols import PdgScalarDefinition
from sciona.physics_ingest.source_normalization import normalize_pinned_source_ast
from sciona.ghost.dimensions import DimensionalSignature


def definitions():
    from types import SimpleNamespace
    return {name:SimpleNamespace(latex=latex,dimension=DimensionalSignature.from_compact(dim)) for name,latex,dim in [('a','F','M1L1T-2'),('b','m','M1'),('c','a','L1T-2')]}


def test_source_ast_normalization_uses_source_dimensions():
    # Names do not overlap mapped labels: source-symbol collisions are rejected.
    d={'src_'+k:v for k,v in definitions().items()}
    a,b,c=sp.symbols('src_a src_b src_c')
    normalized,mapping=normalize_pinned_source_ast(sp.srepr(sp.Eq(a,b*c)),d)
    assert normalized.parse_status=='parsed' and not normalized.review_tasks
    assert mapping=={'F':'src_a','m':'src_b','a':'src_c'}
    assert normalized.variables['F'].dim_signature.to_compact()=='M1L1T-2'


def test_source_normalization_rejects_missing_and_inconsistent_dimensions():
    d={'src_'+k:v for k,v in definitions().items()}
    a,b,c=sp.symbols('src_a src_b src_c')
    d['src_a'].dimension=None
    with pytest.raises(ValueError,match='complete source dimensions'):
        normalize_pinned_source_ast(sp.srepr(sp.Eq(a,b*c)),d)
    d['src_a'].dimension=DimensionalSignature(L=1)
    with pytest.raises(ValueError,match='dimensional check'):
        normalize_pinned_source_ast(sp.srepr(sp.Eq(a,b*c)),d)
