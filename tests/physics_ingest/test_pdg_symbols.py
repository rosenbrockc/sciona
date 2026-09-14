"""Synthetic scalar registry and equations only."""
import hashlib
import pytest
import sympy as sp
from sciona.physics_ingest.pdg_symbols import load_pinned_pdg_scalars, map_pdg_scalar_names


def scalar(identity='42', latex='x', missing=False):
    dimensions = {'time': -1, 'electric_charge': 1, 'luminous_intensity': 0,
                  'length': 0, 'amount_of_substance': 0, 'mass': 0, 'temperature': 0}
    if missing:
        dimensions.pop('temperature')
    props = ','.join('dimension_' + k + ':' + str(v) for k,v in dimensions.items())
    return ('UNWIND [{id:"' + identity + '", properties:{latex:"' + latex + '",' + props + '}}] AS row\n'
            'CREATE (n:scalar{id: row.id}) SET n += row.properties SET n:symbol;\n').encode()


def load(content):
    return load_pinned_pdg_scalars(content, hashlib.sha256(content).hexdigest())


def test_pin_and_dimension_basis_are_checked():
    with pytest.raises(ValueError, match='pin'):
        load_pinned_pdg_scalars(scalar(), 'wrong')
    item = load(scalar())['pdg42']
    assert item.dimension.I == 1
    assert item.dimension.T == 0  # charge/time is current
    assert load(scalar(missing=True))['pdg42'].dimension is None


def test_duplicate_source_id_rejected():
    with pytest.raises(ValueError, match='identity'):
        load(scalar() + scalar())


def test_mapping_uses_source_identity_and_rejects_composite_notation():
    definitions = load(scalar() + scalar('43', 'a b'))
    x, y = sp.symbols('pdg42 pdg43')
    replacements, unresolved = map_pdg_scalar_names(x + y, definitions)
    assert replacements == {x: sp.Symbol('x')}
    assert unresolved == ['pdg43']


def test_collision_rejects_all_affected_source_symbols():
    definitions = load(scalar() + scalar('43') + scalar('44'))
    expression = sp.Add(*sp.symbols('pdg42 pdg43 pdg44'))
    replacements, unresolved = map_pdg_scalar_names(expression, definitions)
    assert replacements == {}
    assert unresolved == ['pdg42','pdg43','pdg44']


def test_mapping_cannot_merge_with_an_unmapped_literal():
    expression = sp.Symbol('pdg42') + sp.Symbol('x')
    replacements, unresolved = map_pdg_scalar_names(expression, load(scalar()))
    assert not replacements
    assert unresolved == ['pdg42', 'x']


def test_correspondence_checks_dimensions_separately():
    from sciona.ghost.symbolic import serialize_expr
    from sciona.physics_ingest.pdg_symbols import inspect_pdg_correspondence
    definitions = load(scalar())
    source = serialize_expr(sp.Eq(sp.Symbol('pdg42'), 1, evaluate=False))
    stored = serialize_expr(sp.Eq(sp.Symbol('x'), 1, evaluate=False))
    result = inspect_pdg_correspondence(source, stored, definitions)
    assert result['correspondence'] == 'exact_ast_match'
    assert result['dimension_status'] == 'checker_errors'  # current != dimensionless
    assert 'approved' not in result


def test_partial_source_mapping_cannot_claim_correspondence():
    from sciona.ghost.symbolic import serialize_expr
    from sciona.physics_ingest.pdg_symbols import inspect_pdg_correspondence
    source = serialize_expr(sp.Eq(sp.Symbol('unknown'), 1, evaluate=False))
    result = inspect_pdg_correspondence(source, source, load(scalar()))
    assert result['mapping_status'] == 'partial'
    assert result['correspondence'] == 'unchecked'
    assert result['dimension_status'] == 'incomplete'
