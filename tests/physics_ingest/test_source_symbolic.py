"""Synthetic source forms only; never use imported source records as fixtures."""
import pytest
import sympy as sp

from sciona.ghost.symbolic import serialize_expr
from sciona.physics_ingest.source_symbolic import (
    SourceSymbolicError, inspect_source_symbolic, parse_source_srepr,
)


def payload(lhs="Symbol('x')", rhs="Integer(1)", relation="="):
    return {"raw_payload": {"sympy_lhs": lhs, "sympy_rhs": rhs, "latex_relation": relation}}


@pytest.mark.parametrize("source", [
    "__import__('os').getcwd()", "Symbol.__class__", "[x for x in ()]",
    "open('/tmp/forbidden', 'w')", "Function('f')(**{'x': 1})",
    "Symbol('x', **{})", "lambda: 1", "1", "Symbol('x') + Integer(1)",
])
def test_rejects_non_grammar_inputs(source):
    with pytest.raises(SourceSymbolicError):
        parse_source_srepr(source)


def test_preserves_singularity_and_unevaluated_equality():
    p = payload("Mul(Symbol('x'), Pow(Symbol('x'), Integer(-1)))", "Integer(1)")
    result = inspect_source_symbolic(p)
    assert result['status'] == 'roundtrip_passed'
    assert 'Pow' in result['sympy_srepr']
    assert 'Equality' in result['sympy_srepr']
    assert result['correspondence'] == 'no_stored_expression'


def test_exact_and_different_ast_are_distinguished_without_simplification():
    p = payload()
    exact = serialize_expr(sp.Eq(sp.Symbol('x'), 1, evaluate=False))
    assert inspect_source_symbolic(p, exact)['correspondence'] == 'exact_ast_match'
    different = serialize_expr(sp.Eq(1, sp.Symbol('x'), evaluate=False))
    result = inspect_source_symbolic(p, different)
    assert result['correspondence'] == 'different_ast_requires_review'
    assert 'approval' not in result


def test_missing_rhs_cannot_be_invented():
    assert inspect_source_symbolic(payload(rhs=''))['reason'] == 'missing_rhs'
    result = inspect_source_symbolic(payload("Equality(Symbol('x'), Integer(1))", ''))
    assert result['status'] == 'roundtrip_passed'


def test_relations_must_be_explicit_and_not_nested():
    assert inspect_source_symbolic(payload(relation='approx'))['reason'] == 'unsupported_relation'
    assert inspect_source_symbolic(payload("Equality(Symbol('x'), Integer(1))"))['reason'] == 'nested_relation'


def test_integral_and_undefined_function_roundtrip():
    result = inspect_source_symbolic(payload("Integral(Function('f')(Symbol('x')), Tuple(Symbol('x')))"))
    assert result['status'] == 'roundtrip_passed'
