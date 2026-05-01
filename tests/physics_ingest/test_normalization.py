"""Tests for physics ingest symbolic normalization drafts."""

from __future__ import annotations

import sympy as sp

from sciona.physics_ingest.normalization import (
    normalize_candidate_expression_draft,
    normalize_candidate_expression_drafts,
)


ARTIFACT_ID = "00000000-0000-0000-0000-000000000101"
VERSION_ID = "00000000-0000-0000-0000-000000000102"
CANDIDATE_ID = "00000000-0000-0000-0000-000000000103"


def test_plain_text_candidate_normalizes_to_expression_row_draft() -> None:
    draft = normalize_candidate_expression_draft(
        {
            "candidate_id": CANDIDATE_ID,
            "source_candidate_id": "fixture-force",
            "raw_formula": "F = m a",
            "raw_formula_format": "plain_text",
            "variables": {
                "F": {"role": "output", "dim_signature": "M1L1T-2"},
                "m": {"role": "input", "dim_signature": "M1"},
                "a": {"role": "input", "dim_signature": "L1T-2"},
            },
            "mechanism_tags": ["newtonian"],
        },
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
        require_dimensions=True,
    )

    row = draft.row

    assert row.candidate_id == CANDIDATE_ID
    assert row.source_expression_id == "fixture-force"
    assert row.parse_status == "normalized"
    assert row.parse_confidence == 0.95
    assert row.review_status == "automated_pass"
    assert row.sympy_srepr.startswith("Equality(")
    assert row.canonical_expr_hash
    assert row.topology_hash
    assert row.dimensional_hash
    assert row.mechanism_tags == ["newtonian"]
    assert row.evidence_json["parse_roundtrip"]["status"] == "passed"
    assert row.evidence_json["normalization"]["review_tasks"] == []


def test_latex_candidate_uses_local_parser_and_preserves_raw_formula() -> None:
    draft = normalize_candidate_expression_draft(
        {
            "source_candidate_id": "fixture-latex-energy",
            "raw_formula": r"E = m c^2",
            "raw_formula_format": "latex",
        },
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
    )

    assert draft.row.parse_status == "normalized"
    assert draft.row.raw_formula == r"E = m c^2"
    assert draft.row.raw_formula_format == "latex"
    assert draft.row.evidence_json["parse_roundtrip"]["status"] == "passed"
    assert "latex_parsed_locally" in {diagnostic.code for diagnostic in draft.diagnostics}


def test_sympy_expression_candidate_normalizes_without_text_preparse() -> None:
    draft = normalize_candidate_expression_draft(
        {
            "source_candidate_id": "fixture-sympy",
            "sympy_expr": sp.Eq(sp.Symbol("v"), sp.Symbol("x") / sp.Symbol("t")),
            "raw_formula": "Eq(v, x/t)",
            "raw_formula_format": "sympy",
        },
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
    )

    assert draft.row.parse_status == "normalized"
    assert draft.row.review_status == "needs_human"
    assert draft.row.canonical_expr_hash
    assert draft.row.topology_hash
    assert {diagnostic.code for diagnostic in draft.diagnostics} == {
        "missing_dimension"
    }


def test_parse_failure_returns_needs_human_row_without_hashes() -> None:
    draft = normalize_candidate_expression_draft(
        {
            "candidate_id": "not-a-uuid-source-id",
            "raw_formula": "F =",
            "raw_formula_format": "plain_text",
        },
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
    )

    row = draft.row

    assert row.candidate_id is None
    assert row.source_expression_id == "not-a-uuid-source-id"
    assert row.parse_status == "parse_failed"
    assert row.parse_confidence == 0.0
    assert row.review_status == "needs_human"
    assert row.raw_formula == "F ="
    assert row.sympy_srepr == ""
    assert row.canonical_expr_hash == ""
    assert row.topology_hash == ""
    assert row.evidence_json["parse_roundtrip"]["status"] == "failed"
    assert "parse_failed" in {diagnostic.code for diagnostic in draft.diagnostics}


def test_batch_normalization_keeps_failed_candidates() -> None:
    drafts = normalize_candidate_expression_drafts(
        [
            {"source_candidate_id": "ok", "raw_formula": "x = y", "raw_formula_format": "plain_text"},
            {"source_candidate_id": "bad", "raw_formula": "x =", "raw_formula_format": "plain_text"},
        ],
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
    )

    assert [draft.row.source_expression_id for draft in drafts] == ["ok", "bad"]
    assert [draft.row.parse_status for draft in drafts] == [
        "normalized",
        "parse_failed",
    ]
