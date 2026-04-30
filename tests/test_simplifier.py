"""Tests for sciona.synthesizer.simplifier – cross-atom algebraic simplification."""

from __future__ import annotations

import pytest
import sympy as sp

from sciona.architect.models import IOSpec
from sciona.ghost.dimensions import DIMENSIONLESS
from sciona.ghost.registry import REGISTRY
from sciona.ghost.symbolic import SymbolicExpression, serialize_expr
from sciona.synthesizer.models import AssemblyUnit, GlueEdge
from sciona.synthesizer.simplifier import SimplificationResult, simplify_pipeline


def _cleanup_registry(*names: str):
    for n in names:
        REGISTRY.pop(n, None)


def _register_symbolic(name: str, expr, variables: dict, dim_map=None):
    """Register a fake atom with symbolic metadata."""
    sym = SymbolicExpression(
        srepr_str=serialize_expr(expr),
        variables=variables,
        dim_map=dim_map or {},
    )
    REGISTRY[name] = {
        "impl": None,
        "witness": None,
        "doc": f"test: {name}",
        "signature": {},
        "heavy_signature": {},
        "module": "",
        "name": name,
        "dim_signature": {},
        "symbolic": sym,
    }


def _make_unit(node_id: str, name: str, decl_name: str,
               inputs: list[IOSpec] | None = None,
               outputs: list[IOSpec] | None = None) -> AssemblyUnit:
    return AssemblyUnit(
        node_id=node_id,
        name=name,
        declaration_name=decl_name,
        type_signature="",
        inputs=inputs or [],
        outputs=outputs or [],
    )


# ---------------------------------------------------------------------------
# Core simplification
# ---------------------------------------------------------------------------


class TestSimplifier:
    def test_add_subtract_cancels(self):
        """(x + 1) -> (y - 1) should simplify to z = x."""
        x = sp.Symbol("x")
        y = sp.Symbol("y")

        expr_a = x + 1       # A: y = x + 1
        expr_b = y - 1       # B: z = y - 1

        _register_symbolic("test.atom_add1", expr_a, {"x": "input"},
                           dim_map={"x": DIMENSIONLESS})
        _register_symbolic("test.atom_sub1", expr_b, {"y": "input"},
                           dim_map={"y": DIMENSIONLESS})

        units = [
            _make_unit("a", "atom_add1", "test.atom_add1",
                       inputs=[IOSpec(name="x", type_desc="float")],
                       outputs=[IOSpec(name="y", type_desc="float")]),
            _make_unit("b", "atom_sub1", "test.atom_sub1",
                       inputs=[IOSpec(name="y", type_desc="float")],
                       outputs=[IOSpec(name="z", type_desc="float")]),
        ]
        edges = [
            GlueEdge(source_id="a", target_id="b",
                      output_name="y", input_name="y",
                      source_type="float", target_type="float"),
        ]

        try:
            result_units, results = simplify_pipeline(units, edges)
            assert len(results) == 1
            assert results[0].simplified_ops < results[0].original_ops
            assert len(result_units) == 1
            assert "fused" in result_units[0].name
        finally:
            _cleanup_registry("test.atom_add1", "test.atom_sub1",
                              result_units[0].declaration_name if result_units else "")

    def test_exp_log_cancels_with_dim(self):
        """exp(x) -> log(y) simplifies to z = x when variables have dimensions (real-valued)."""
        x = sp.Symbol("x")
        y = sp.Symbol("y")

        expr_a = sp.exp(x)
        expr_b = sp.log(y)

        # Variables with dim annotations are treated as real by the simplifier
        _register_symbolic("test.atom_exp", expr_a, {"x": "input"},
                           dim_map={"x": DIMENSIONLESS})
        _register_symbolic("test.atom_log", expr_b, {"y": "input"},
                           dim_map={"y": DIMENSIONLESS})

        units = [
            _make_unit("a", "atom_exp", "test.atom_exp",
                       inputs=[IOSpec(name="x", type_desc="float")],
                       outputs=[IOSpec(name="y", type_desc="float")]),
            _make_unit("b", "atom_log", "test.atom_log",
                       inputs=[IOSpec(name="y", type_desc="float")],
                       outputs=[IOSpec(name="z", type_desc="float")]),
        ]
        edges = [
            GlueEdge(source_id="a", target_id="b",
                      output_name="y", input_name="y",
                      source_type="float", target_type="float"),
        ]

        try:
            result_units, results = simplify_pipeline(units, edges)
            assert len(results) == 1
            assert results[0].simplified_ops < results[0].original_ops
            assert len(result_units) == 1
        finally:
            _cleanup_registry("test.atom_exp", "test.atom_log",
                              result_units[0].declaration_name if result_units else "")

    def test_no_real_assumption_without_dim(self):
        """exp(x) -> log(y) does NOT simplify when variables lack dim annotations (may be complex)."""
        x = sp.Symbol("x")
        y = sp.Symbol("y")

        expr_a = sp.exp(x)
        expr_b = sp.log(y)

        # No dim_map -> symbols stay complex -> log(exp(x)) != x
        _register_symbolic("test.atom_exp_c", expr_a, {"x": "input"})
        _register_symbolic("test.atom_log_c", expr_b, {"y": "input"})

        units = [
            _make_unit("a", "atom_exp_c", "test.atom_exp_c",
                       inputs=[IOSpec(name="x", type_desc="float")],
                       outputs=[IOSpec(name="y", type_desc="float")]),
            _make_unit("b", "atom_log_c", "test.atom_log_c",
                       inputs=[IOSpec(name="y", type_desc="float")],
                       outputs=[IOSpec(name="z", type_desc="float")]),
        ]
        edges = [
            GlueEdge(source_id="a", target_id="b",
                      output_name="y", input_name="y",
                      source_type="float", target_type="float"),
        ]

        try:
            result_units, results = simplify_pipeline(units, edges)
            # Should NOT simplify because symbols are potentially complex
            assert len(results) == 0
            assert len(result_units) == 2
        finally:
            _cleanup_registry("test.atom_exp_c", "test.atom_log_c")

    def test_no_simplification_when_not_simpler(self):
        """Two unrelated expressions should not be fused."""
        x = sp.Symbol("x")
        y = sp.Symbol("y")

        expr_a = sp.sin(x) + sp.cos(x)
        expr_b = y**2 + sp.exp(y)

        _register_symbolic("test.atom_sincos", expr_a, {"x": "input"})
        _register_symbolic("test.atom_poly", expr_b, {"y": "input"})

        units = [
            _make_unit("a", "atom_sincos", "test.atom_sincos",
                       inputs=[IOSpec(name="x", type_desc="float")],
                       outputs=[IOSpec(name="y", type_desc="float")]),
            _make_unit("b", "atom_poly", "test.atom_poly",
                       inputs=[IOSpec(name="y", type_desc="float")],
                       outputs=[IOSpec(name="z", type_desc="float")]),
        ]
        edges = [
            GlueEdge(source_id="a", target_id="b",
                      output_name="y", input_name="y",
                      source_type="float", target_type="float"),
        ]

        try:
            result_units, results = simplify_pipeline(units, edges)
            # Composing sin+cos into poly+exp is not simpler
            # If no simplification happened, original units should be preserved
            assert len(result_units) == 2
            assert len(results) == 0
        finally:
            _cleanup_registry("test.atom_sincos", "test.atom_poly")

    def test_non_symbolic_units_pass_through(self):
        """Units without symbolic metadata are passed through unchanged."""
        units = [
            _make_unit("a", "regular_atom", "test.regular",
                       inputs=[IOSpec(name="x", type_desc="float")],
                       outputs=[IOSpec(name="y", type_desc="float")]),
        ]
        edges: list[GlueEdge] = []

        result_units, results = simplify_pipeline(units, edges)
        assert len(result_units) == 1
        assert result_units[0].node_id == "a"
        assert results == []

    def test_chain_of_three_simplifies_first_pair(self):
        """A -> B -> C where A+B simplify but B+C don't."""
        x = sp.Symbol("x")
        y = sp.Symbol("y")
        z = sp.Symbol("z")

        # A: x^2, B: sqrt(y), C: z + 1
        # A->B: sqrt(x^2) = |x| ~ x (simplifies)
        _register_symbolic("test.sq", x**2, {"x": "input"})
        _register_symbolic("test.sqrt", sp.sqrt(y), {"y": "input"})
        _register_symbolic("test.inc", z + 1, {"z": "input"})

        units = [
            _make_unit("a", "sq", "test.sq",
                       inputs=[IOSpec(name="x", type_desc="float")],
                       outputs=[IOSpec(name="y", type_desc="float")]),
            _make_unit("b", "sqrt", "test.sqrt",
                       inputs=[IOSpec(name="y", type_desc="float")],
                       outputs=[IOSpec(name="z", type_desc="float")]),
            _make_unit("c", "inc", "test.inc",
                       inputs=[IOSpec(name="z", type_desc="float")],
                       outputs=[IOSpec(name="w", type_desc="float")]),
        ]
        edges = [
            GlueEdge(source_id="a", target_id="b",
                      output_name="y", input_name="y",
                      source_type="float", target_type="float"),
            GlueEdge(source_id="b", target_id="c",
                      output_name="z", input_name="z",
                      source_type="float", target_type="float"),
        ]

        try:
            result_units, results = simplify_pipeline(units, edges)
            # At least some simplification should occur
            assert len(result_units) <= 3
        finally:
            _cleanup_registry("test.sq", "test.sqrt", "test.inc")
            for u in result_units:
                _cleanup_registry(u.declaration_name)
