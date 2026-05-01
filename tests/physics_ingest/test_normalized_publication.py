"""Tests for normalized expression draft publication rows."""

from __future__ import annotations

from uuid import UUID

from sciona.physics_ingest import (
    build_publication_load_result_from_normalized_drafts,
)
from sciona.physics_ingest.normalization import (
    NormalizedExpressionDraft,
    normalize_candidate_expression_draft,
)


ARTIFACT_ID = "00000000-0000-0000-0000-000000000201"
VERSION_ID = "00000000-0000-0000-0000-000000000202"


def test_normalized_draft_publication_rows_include_variables_and_bounds() -> None:
    draft = normalize_candidate_expression_draft(
        {
            "source_candidate_id": "force-law",
            "raw_formula": "F = m a",
            "raw_formula_format": "plain_text",
            "variables": {
                "F": {
                    "role": "output",
                    "aliases": ["force"],
                    "dim_signature": "M1L1T-2",
                    "quantity_kind": "force",
                    "qudt_uri": "http://qudt.org/vocab/quantitykind/Force",
                    "unit_uri": "http://qudt.org/vocab/unit/N",
                    "assumptions": {"frame": "inertial"},
                },
                "m": {"role": "input", "dim_signature": "M1"},
                "a": {"role": "input", "dim_signature": "L1T-2"},
            },
            "validity_bounds": {"m": {"min": 0, "max": None}},
        },
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
    )

    result = build_publication_load_result_from_normalized_drafts([draft])
    rows = result.to_insert_rows()

    assert result.diagnostics == ()
    assert len(rows["artifact_symbolic_expressions"]) == 1
    assert rows["artifact_symbolic_expressions"][0]["source_expression_id"] == "force-law"
    assert UUID(rows["artifact_symbolic_expressions"][0]["expression_id"])

    variables = {
        row["symbol_name"]: row for row in rows["artifact_symbolic_variables"]
    }
    assert set(variables) == {"F", "a", "m"}
    assert variables["F"]["aliases"] == ["F", "force"]
    assert variables["F"]["variable_role"] == "output"
    assert variables["F"]["quantity_kind_uri"] == "http://qudt.org/vocab/quantitykind/Force"
    assert variables["F"]["quantity_kind_label"] == "force"
    assert variables["F"]["unit_uri"] == "http://qudt.org/vocab/unit/N"
    assert variables["F"]["dim_signature"] == "M1L1T-2"
    assert variables["F"]["assumptions_json"] == {"frame": "inertial"}
    assert UUID(variables["F"]["variable_id"])

    bounds = rows["artifact_validity_bounds"]
    assert len(bounds) == 1
    assert UUID(bounds[0]["bound_id"])
    assert bounds[0]["variable_id"] == variables["m"]["variable_id"]
    assert bounds[0]["variable_name"] == "m"
    assert bounds[0]["lower_value"] == 0.0
    assert "upper_value" not in bounds[0]
    assert bounds[0]["validity_statement"] == "m >= 0.0"


def test_normalized_draft_publication_preserves_failed_expression_drafts() -> None:
    draft = normalize_candidate_expression_draft(
        {
            "source_candidate_id": "bad-expression",
            "raw_formula": "F =",
            "raw_formula_format": "plain_text",
        },
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
    )

    result = build_publication_load_result_from_normalized_drafts([draft])
    rows = result.to_insert_rows()

    assert result.diagnostics == ()
    assert rows["artifact_symbolic_expressions"][0]["parse_status"] == "parse_failed"
    assert rows["artifact_symbolic_expressions"][0]["review_status"] == "needs_human"
    assert rows["artifact_symbolic_expressions"][0]["raw_formula"] == "F ="
    assert rows["artifact_symbolic_variables"] == []
    assert rows["artifact_validity_bounds"] == []


def test_normalized_draft_publication_handles_partial_mapping_metadata() -> None:
    base = normalize_candidate_expression_draft(
        {
            "source_candidate_id": "mapping-expression",
            "raw_formula": "x = y",
            "raw_formula_format": "plain_text",
        },
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
    )
    draft = NormalizedExpressionDraft(
        row=base.row,
        normalized_candidate={
            "variables": {
                "x": {
                    "role": "observable",
                    "aliases": ["position"],
                    "evidence_json": {"source": "fixture"},
                }
            },
            "validity_bounds": {
                "x": {
                    "lower": 1,
                    "upper": 0,
                    "evidence": {"source": "fixture-bound"},
                }
            },
        },
        diagnostics=base.diagnostics,
    )

    result = build_publication_load_result_from_normalized_drafts([draft])
    rows = result.to_insert_rows()

    assert rows["artifact_symbolic_expressions"][0]["source_expression_id"] == (
        "mapping-expression"
    )
    assert rows["artifact_symbolic_variables"][0]["variable_role"] == "output"
    assert rows["artifact_symbolic_variables"][0]["evidence_json"]["source"] == "fixture"
    assert rows["artifact_validity_bounds"] == []
    assert [
        (diagnostic.table, diagnostic.reason, diagnostic.severity)
        for diagnostic in result.diagnostics
    ] == [("artifact_validity_bounds", "validation_error", "error")]
