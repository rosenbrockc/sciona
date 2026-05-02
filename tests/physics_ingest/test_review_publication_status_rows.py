from __future__ import annotations

from sciona.physics_ingest import (
    assess_publishability,
    build_review_publication_status_rows,
)


CANDIDATE_ID = "11111111-1111-4111-8111-111111111111"
EXPRESSION_ID = "22222222-2222-4222-8222-222222222222"


def test_review_publication_rows_mark_automated_pass_before_human_review() -> None:
    assessment = assess_publishability(**_review_bundle(review_status="automated_pass"))

    result = build_review_publication_status_rows(
        assessment,
        candidate={"candidate_id": CANDIDATE_ID},
        expression={"expression_id": EXPRESSION_ID},
    )

    assert result.to_upsert_rows() == {
        "physics_equation_candidates": [
            {
                "candidate_id": CANDIDATE_ID,
                "candidate_status": "source_verified",
            }
        ],
        "artifact_symbolic_expressions": [
            {
                "expression_id": EXPRESSION_ID,
                "review_status": "automated_pass",
            }
        ],
    }
    assert result.diagnostics == ()


def test_review_publication_rows_accept_review_reports_and_human_status() -> None:
    assessment = assess_publishability(
        **_review_bundle(
            review_status="human_reviewed",
            human_review={
                "reviewer_id": "reviewer-1",
                "reviewed_at": "2026-04-30T12:00:00Z",
            },
            bound_review_status="human_reviewed",
            candidate_status="human_reviewed",
        )
    )

    result = build_review_publication_status_rows(
        assessment.to_report().to_dict(),
        candidate={"candidate_id": CANDIDATE_ID},
        expression={"expression_id": EXPRESSION_ID},
    )

    assert result.artifact_symbolic_expressions == (
        {
            "expression_id": EXPRESSION_ID,
            "review_status": "human_reviewed",
        },
    )
    assert result.physics_equation_candidates == (
        {
            "candidate_id": CANDIDATE_ID,
            "candidate_status": "human_reviewed",
        },
    )


def test_review_publication_rows_report_missing_ids_without_guessing() -> None:
    assessment = assess_publishability(
        **_review_bundle(
            review_status="blocked",
            candidate_status="blocked",
        )
    )

    result = build_review_publication_status_rows(
        assessment,
        candidate={"candidate_id": CANDIDATE_ID},
        expression={"raw_formula": "F = m a"},
    )

    assert result.physics_equation_candidates == (
        {
            "candidate_id": CANDIDATE_ID,
            "candidate_status": "blocked",
        },
    )
    assert result.artifact_symbolic_expressions == ()
    assert [diagnostic.to_dict() for diagnostic in result.diagnostics] == [
        {
            "table": "artifact_symbolic_expressions",
            "reason": "missing_expression_id",
            "row_id": "",
            "severity": "skipped",
            "detail": "review_status patch requires expression_id",
        }
    ]


def test_review_publication_rows_skip_non_actionable_unreviewed_reports() -> None:
    result = build_review_publication_status_rows(
        {"trust_status": "unreviewed"},
        candidate={"candidate_id": CANDIDATE_ID},
        expression={"expression_id": EXPRESSION_ID},
    )

    assert result.to_upsert_rows() == {}
    assert [diagnostic.reason for diagnostic in result.diagnostics] == [
        "non_actionable_assessment"
    ]


def _review_bundle(
    *,
    review_status: str,
    human_review: dict[str, object] | None = None,
    bound_review_status: str = "automated_pass",
    candidate_status: str = "source_verified",
) -> dict[str, object]:
    evidence_json = {
        "parse_roundtrip": {"status": "passed"},
        "dimensional_analysis": {"status": "passed"},
        "numpy_runtime": {
            "no_sympy_runtime": True,
            "runtime_imports": ["numpy"],
            "tests_passed": True,
            "source": "import numpy as np\n\ndef force(m, a):\n    return m * a\n",
        },
    }
    if human_review is not None:
        evidence_json["human_review"] = human_review

    return {
        "candidate": {
            "raw_formula": "F = m a",
            "candidate_status": candidate_status,
            "parse_confidence": 0.96,
        },
        "expression": {
            "raw_formula": "F = m a",
            "sympy_srepr": "Equality(Symbol('F'), Mul(Symbol('m'), Symbol('a')))",
            "parse_status": "normalized",
            "parse_confidence": 0.96,
            "canonical_expr_hash": "a" * 64,
            "topology_hash": "b" * 64,
            "dimensional_hash": "c" * 64,
            "validation_status": "passed",
            "review_status": review_status,
            "mechanism_tags": ["force_balance", "classical_mechanics"],
            "behavioral_archetypes": ["linear_proportionality"],
            "evidence_json": evidence_json,
        },
        "variables": [
            {
                "symbol_name": "F",
                "variable_role": "output",
                "dim_signature": "M1L1T-2",
                "dimension_source": "qudt",
            }
        ],
        "references": [{"title": "Newton fixture", "verified": True}],
        "validity_bounds": [
            {
                "variable_name": "m",
                "lower_value": 0.0,
                "validity_statement": "mass is non-negative",
                "review_status": bound_review_status,
            }
        ],
    }
