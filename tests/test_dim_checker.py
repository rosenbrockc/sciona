"""Tests for sciona.synthesizer.dim_checker – dimensional analysis in compilation."""

from __future__ import annotations

import pytest

from sciona.architect.models import IOSpec
from sciona.ghost.dimensions import (
    DIMENSIONLESS,
    METER,
    SECOND,
    VOLT,
    WATT,
    DimensionalSignature,
)
from sciona.synthesizer.dim_checker import (
    DimCheckResult,
    check_dimensional_consistency,
)
from sciona.synthesizer.models import AssemblyUnit, GlueEdge


def _make_unit(
    node_id: str,
    name: str,
    inputs: list[IOSpec] | None = None,
    outputs: list[IOSpec] | None = None,
) -> AssemblyUnit:
    return AssemblyUnit(
        node_id=node_id,
        name=name,
        declaration_name=f"test.{name}",
        type_signature="",
        inputs=inputs or [],
        outputs=outputs or [],
    )


def _make_edge(
    source_id: str,
    target_id: str,
    output_name: str = "out",
    input_name: str = "in",
    source_dim: str = "",
    target_dim: str = "",
) -> GlueEdge:
    return GlueEdge(
        source_id=source_id,
        target_id=target_id,
        output_name=output_name,
        input_name=input_name,
        source_type="np.ndarray",
        target_type="np.ndarray",
        source_dim=source_dim,
        target_dim=target_dim,
    )


# ---------------------------------------------------------------------------
# Core rules
# ---------------------------------------------------------------------------


class TestDimChecker:
    def test_both_annotated_compatible_passes(self):
        """Both sides have same dim -> pass."""
        units = [
            _make_unit("a", "atom_a"),
            _make_unit("b", "atom_b"),
        ]
        edges = [_make_edge("a", "b", source_dim="M1L2T-3", target_dim="M1L2T-3")]
        result = check_dimensional_consistency(units, edges)
        assert result.passed
        assert result.errors == []
        assert result.warnings == []

    def test_both_annotated_incompatible_fails(self):
        """Power != Voltage -> error."""
        units = [
            _make_unit("a", "atom_a"),
            _make_unit("b", "atom_b"),
        ]
        edges = [_make_edge(
            "a", "b",
            source_dim=WATT.to_compact(),
            target_dim=VOLT.to_compact(),
        )]
        result = check_dimensional_consistency(units, edges)
        assert not result.passed
        assert len(result.errors) == 1
        assert "mismatch" in result.errors[0].message.lower()

    def test_one_annotated_warns(self):
        """Source has dim, target doesn't -> warning (not error)."""
        units = [
            _make_unit("a", "atom_a"),
            _make_unit("b", "atom_b"),
        ]
        edges = [_make_edge("a", "b", source_dim=WATT.to_compact())]
        result = check_dimensional_consistency(units, edges)
        assert result.passed  # NOT an error
        assert len(result.warnings) == 1
        assert "partial" in result.warnings[0].lower()

    def test_both_unannotated_silent_pass(self):
        """Neither side has dim -> silent pass."""
        units = [
            _make_unit("a", "atom_a"),
            _make_unit("b", "atom_b"),
        ]
        edges = [_make_edge("a", "b")]
        result = check_dimensional_consistency(units, edges)
        assert result.passed
        assert result.errors == []
        assert result.warnings == []

    def test_dimensionless_compatible_with_dimensionless(self):
        """Explicit DIMENSIONLESS on both sides passes."""
        units = [
            _make_unit("a", "atom_a"),
            _make_unit("b", "atom_b"),
        ]
        edges = [_make_edge(
            "a", "b",
            source_dim=DIMENSIONLESS.to_compact(),
            target_dim=DIMENSIONLESS.to_compact(),
        )]
        result = check_dimensional_consistency(units, edges)
        assert result.passed

    def test_multiple_edges_mixed(self):
        """Multiple edges: one bad + one good -> overall fails."""
        units = [
            _make_unit("a", "atom_a"),
            _make_unit("b", "atom_b"),
            _make_unit("c", "atom_c"),
        ]
        edges = [
            _make_edge("a", "b", source_dim=WATT.to_compact(), target_dim=WATT.to_compact()),  # OK
            _make_edge("b", "c", source_dim=WATT.to_compact(), target_dim=METER.to_compact()),  # BAD
        ]
        result = check_dimensional_consistency(units, edges)
        assert not result.passed
        assert len(result.errors) == 1

    def test_lookup_from_iospec(self):
        """Dim info from IOSpec on unit inputs/outputs is used when GlueEdge has no dim."""
        units = [
            _make_unit("a", "atom_a", outputs=[
                IOSpec(name="signal", type_desc="np.ndarray", dim_signature=WATT.to_compact()),
            ]),
            _make_unit("b", "atom_b", inputs=[
                IOSpec(name="signal", type_desc="np.ndarray", dim_signature=VOLT.to_compact()),
            ]),
        ]
        edges = [_make_edge("a", "b", output_name="signal", input_name="signal")]
        result = check_dimensional_consistency(units, edges)
        assert not result.passed
        assert len(result.errors) == 1

    def test_iospec_dims_match(self):
        """Matching IOSpec dims pass."""
        units = [
            _make_unit("a", "atom_a", outputs=[
                IOSpec(name="out", type_desc="float", dim_signature=SECOND.to_compact()),
            ]),
            _make_unit("b", "atom_b", inputs=[
                IOSpec(name="in", type_desc="float", dim_signature=SECOND.to_compact()),
            ]),
        ]
        edges = [_make_edge("a", "b")]
        result = check_dimensional_consistency(units, edges)
        assert result.passed


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestDimCheckerEdgeCases:
    def test_missing_unit_skipped(self):
        """Edge referencing non-existent unit is silently skipped."""
        units = [_make_unit("a", "atom_a")]
        edges = [_make_edge("a", "nonexistent", source_dim=WATT.to_compact(), target_dim=VOLT.to_compact())]
        result = check_dimensional_consistency(units, edges)
        assert result.passed  # skipped, not failed

    def test_empty_edges(self):
        """No edges -> trivially passes."""
        result = check_dimensional_consistency([], [])
        assert result.passed

    def test_invalid_compact_string_parsed_as_dimensionless(self):
        """Malformed dim string with no valid tokens parses as DIMENSIONLESS."""
        units = [
            _make_unit("a", "atom_a"),
            _make_unit("b", "atom_b"),
        ]
        # "INVALID!!!" has no valid dimension tokens -> parses as DIMENSIONLESS ("1")
        # DIMENSIONLESS != WATT -> error
        edges = [_make_edge("a", "b", source_dim="INVALID!!!", target_dim=WATT.to_compact())]
        result = check_dimensional_consistency(units, edges)
        assert not result.passed
        assert len(result.errors) == 1
