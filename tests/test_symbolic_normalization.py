"""Tests for source-neutral symbolic equation normalization."""

from __future__ import annotations

from fractions import Fraction

import sympy as sp

from sciona.ghost.dimensions import (
    JOULE,
    KELVIN,
    METER,
    MOLE,
    PASCAL,
    SECOND,
)
from sciona.ghost.symbolic import SymbolicExpression, serialize_expr
from sciona.ghost.symbolic_normalization import normalize_symbolic_candidate


def test_normalizes_equation_candidate_and_preserves_aliases():
    result = normalize_symbolic_candidate(
        {
            "candidate_id": "ideal-gas-law",
            "source_id": "fixture",
            "source_version": "v1",
            "formula": "P*V = n*R*T",
            "variable_hints": {
                "P": {
                    "aliases": ["pressure", "p"],
                    "role": "input",
                    "dim_signature": PASCAL,
                    "quantity_kind": "Pressure",
                    "qudt_uri": "http://qudt.org/vocab/quantitykind/Pressure",
                },
                "V": {"aliases": ["volume"], "dim_signature": "L3"},
                "n": {"aliases": ["amount"], "dim_signature": MOLE},
                "R": {
                    "aliases": ["gas constant"],
                    "role": "constant",
                    "dim_signature": JOULE.divide(MOLE).divide(KELVIN),
                },
                "T": {"aliases": ["temperature"], "dim_signature": KELVIN},
            },
            "constants": {"R": 8.31446261815324},
            "validity_bounds": {"T": {"min": 0.0, "max": None}},
            "references": ["fixture:ideal-gas"],
        }
    )

    assert result.parse_status == "parsed"
    assert result.srepr_str is not None
    assert result.expression_hash is not None
    assert result.topology_hash is not None
    assert result.dimensional_hash is not None
    assert result.variables["P"].aliases == ["P", "pressure", "p"]
    assert result.variables["P"].quantity_kind == "Pressure"
    assert result.variables["R"].role == "constant"
    assert result.validity_bounds == {"T": (0.0, None)}
    assert result.bibliography == ["fixture:ideal-gas"]
    assert result.review_tasks == []

    symbolic = result.to_symbolic_expression()
    assert isinstance(symbolic, SymbolicExpression)
    assert symbolic.variables["P"] == "input"
    assert symbolic.variables["R"] == "constant"
    assert symbolic.check_dimensional_consistency() == []


def test_accepts_live_sympy_expression_and_emits_symbolic_kwargs():
    x, t = sp.symbols("x t")
    result = normalize_symbolic_candidate(
        {
            "id": "velocity",
            "sympy_expr": sp.Eq(sp.Symbol("v"), x / t),
            "variables": {
                "v": {"role": "output", "dim_signature": METER.divide(SECOND)},
                "x": {"role": "input", "dim_signature": METER},
                "t": {"role": "input", "dim_signature": SECOND},
            },
        }
    )

    kwargs = result.symbolic_expression_kwargs()
    assert kwargs["variables"] == {"t": "input", "v": "output", "x": "input"}
    assert kwargs["dim_map"]["v"] == METER.divide(SECOND)
    assert SymbolicExpression(**kwargs).check_dimensional_consistency() == []


def test_expression_hash_is_exact_but_topology_hash_ignores_symbol_names():
    left = normalize_symbolic_candidate({"formula": "x + y"})
    right = normalize_symbolic_candidate({"formula": "a + b"})

    assert left.parse_status == "parsed"
    assert right.parse_status == "parsed"
    assert left.expression_hash != right.expression_hash
    assert left.topology_hash == right.topology_hash


def test_dimensional_hash_uses_fractional_compact_dimensions():
    first = normalize_symbolic_candidate(
        {
            "formula": "y = sqrt(x)",
            "variables": {
                "x": {"dim_signature": "L1"},
                "y": {"dim_signature": {"L": Fraction(1, 2)}},
            },
        }
    )
    second = normalize_symbolic_candidate(
        {
            "formula": "y = sqrt(x)",
            "variables": {
                "x": {"dim_signature": "L1"},
                "y": {"dim_signature": "L1/2"},
            },
        }
    )

    assert first.parse_status == "parsed"
    assert second.parse_status == "parsed"
    assert first.topology_hash == second.topology_hash
    assert first.dimensional_hash == second.dimensional_hash
    assert first.variables["y"].dim_signature.to_compact() == "L1/2"


def test_srepr_input_round_trips():
    x = sp.Symbol("x")
    srepr_str = serialize_expr(sp.exp(-x))
    result = normalize_symbolic_candidate(
        {
            "candidate_id": "srepr-fixture",
            "srepr_str": srepr_str,
            "variables": {"x": {"dim_signature": "1"}},
        }
    )

    assert result.parse_status == "parsed"
    assert result.srepr_str == srepr_str
    assert result.to_symbolic_expression().check_dimensional_consistency() == []


def test_unknown_dimensions_are_review_tasks_not_dimensionless_defaults():
    result = normalize_symbolic_candidate(
        {
            "formula": "F = m*a",
            "variables": {
                "F": {"role": "output"},
                "m": {"dim_signature": "M1"},
                "a": {"dim_signature": "L1T-2"},
            },
        },
        require_dimensions=True,
    )

    assert result.parse_status == "parsed"
    assert [task.code for task in result.review_tasks] == [
        "missing_required_dimension"
    ]
    assert result.review_tasks[0].symbol == "F"
    kwargs = result.symbolic_expression_kwargs()
    assert "F" not in kwargs["dim_map"]


def test_parse_failure_returns_reviewable_candidate():
    result = normalize_symbolic_candidate(
        {
            "candidate_id": "latex-only",
            "raw_formula": r"E = mc^2",
            "raw_formula_format": "latex",
        }
    )

    assert result.parse_status == "failed"
    assert result.srepr_str is None
    assert result.review_tasks[0].code == "parse_failed"
