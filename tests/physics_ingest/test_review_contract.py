from __future__ import annotations

import json

import pytest

from sciona.physics_ingest.review import (
    REVIEW_STATUSES,
    WORKFLOW_STATUSES,
    assess_publishability,
    build_review_trust_report,
    require_publishable,
    summarize_review_assessments,
)


def test_phase5_review_marks_fully_evidenced_bundle_publishable() -> None:
    assessment = assess_publishability(**_publishable_bundle())

    assert WORKFLOW_STATUSES == (
        "raw_imported",
        "parsed",
        "dimension_resolved",
        "symbolically_validated",
        "source_verified",
        "human_reviewed",
        "published",
    )
    assert assessment.publishable is True
    assert assessment.achieved_status == "published"
    assert [gate.status for gate in assessment.gates] == list(WORKFLOW_STATUSES)
    assert all(gate.passed for gate in assessment.gates)
    assert REVIEW_STATUSES == (
        "unreviewed",
        "automated_pass",
        "needs_human",
        "human_reviewed",
        "blocked",
    )
    assert assessment.trust_status == "human_reviewed"
    assert assessment.human_reviewed is True
    assert assessment.needs_human is False
    assert assessment.blocked is False
    assert assessment.gate("symbolically_validated").evidence == {
        "numpy_runtime_checked": True,
        "has_mechanism_classification": True,
        "mechanism_tag_count": 2,
        "behavioral_archetype_count": 1,
        "classification_detail_count": 0,
        "mechanism_evidence_sources": [
            "expression.mechanism_tags",
            "expression.behavioral_archetypes",
        ],
    }


def test_phase5_review_blocks_parse_roundtrip_and_low_confidence() -> None:
    bundle = _publishable_bundle()
    bundle["candidate"] = {**bundle["candidate"], "parse_confidence": 0.79}
    bundle["expression"] = {
        **bundle["expression"],
        "parse_confidence": 0.79,
        "evidence_json": {
            **bundle["expression"]["evidence_json"],
            "parse_roundtrip": {"status": "failed"},
        },
    }

    assessment = assess_publishability(**bundle)

    assert assessment.publishable is False
    assert assessment.achieved_status == "raw_imported"
    assert assessment.gate("parsed").passed is False
    assert "parse_confidence must be >= 0.8" in assessment.blockers
    assert "parse roundtrip evidence must pass" in assessment.blockers


def test_phase5_review_blocks_dimension_gaps_and_missing_bounds() -> None:
    bundle = _publishable_bundle()
    bundle["variables"] = [
        {**bundle["variables"][0], "dim_signature": ""},
        {**bundle["variables"][1], "dimension_source": "unknown"},
    ]
    bundle["expression"] = {
        **bundle["expression"],
        "dimensional_hash": "",
        "evidence_json": {
            **bundle["expression"]["evidence_json"],
            "dimensional_analysis": {"status": "failed"},
        },
    }
    bundle["validity_bounds"] = []

    assessment = assess_publishability(**bundle)

    assert assessment.publishable is False
    assert assessment.achieved_status == "parsed"
    assert assessment.gate("dimension_resolved").passed is False
    assert "dimensional_hash is missing" in assessment.blockers
    assert "variables missing dim_signature: F" in assessment.blockers
    assert "variables missing dimension_source: m" in assessment.blockers
    assert "dimensional consistency evidence must pass" in assessment.blockers
    assert "validity bounds are required or must be explicitly waived" in assessment.blockers


def test_phase5_review_blocks_explicit_unknown_dimension_signatures() -> None:
    bundle = _publishable_bundle()
    bundle["variables"] = [
        {**bundle["variables"][0], "dim_signature": "unknown"},
        {**bundle["variables"][1], "dim_signature": "?"},
        bundle["variables"][2],
    ]
    bundle["io_specs"] = [
        {**bundle["io_specs"][0], "dim_signature": "unresolved"},
        *bundle["io_specs"][1:],
    ]

    assessment = assess_publishability(**bundle)

    assert assessment.publishable is False
    assert assessment.needs_human is True
    assert "variables missing dim_signature: F, m" in assessment.blockers
    assert "io_specs missing dim_signature: m" in assessment.blockers


def test_phase5_review_blocks_unverified_sources_and_dependencies() -> None:
    bundle = _publishable_bundle()
    bundle["references"] = [{"title": "fixture", "verified": False}]
    bundle["relationships"] = [
        {
            **bundle["relationships"][0],
            "relationship_label": "speed of light",
            "verified": False,
        }
    ]

    assessment = assess_publishability(**bundle)

    assert assessment.publishable is False
    assert assessment.achieved_status == "symbolically_validated"
    assert "at least one verified reference is required" in assessment.blockers
    assert "speed of light relationship must be verified" in assessment.blockers


def test_phase5_review_requires_mechanism_classification_for_source_verification() -> None:
    bundle = _publishable_bundle()
    bundle["expression"] = {
        key: value
        for key, value in bundle["expression"].items()
        if key not in {"mechanism_tags", "behavioral_archetypes"}
    }

    assessment = assess_publishability(**bundle)

    assert assessment.publishable is False
    assert assessment.achieved_status == "symbolically_validated"
    assert assessment.gate("symbolically_validated").passed is True
    assert assessment.gate("symbolically_validated").evidence[
        "has_mechanism_classification"
    ] is False
    assert assessment.gate("source_verified").passed is False
    assert "mechanism classification evidence is required" in assessment.blockers


def test_phase5_review_accepts_nested_mechanism_classification_evidence() -> None:
    bundle = _publishable_bundle()
    expression = {
        key: value
        for key, value in bundle["expression"].items()
        if key not in {"mechanism_tags", "behavioral_archetypes"}
    }
    bundle["expression"] = {
        **expression,
        "evidence_json": {
            **expression["evidence_json"],
            "classification": {
                "mechanism_class": "force_balance",
                "basis": "source annotation classifies the relation as inertial force balance",
            },
        },
    }

    assessment = assess_publishability(**bundle)

    assert assessment.publishable is True
    assert assessment.gate("source_verified").evidence[
        "classification_detail_count"
    ] == 2
    assert assessment.gate("source_verified").evidence[
        "mechanism_evidence_sources"
    ] == ["evidence_json.classification"]


def test_phase5_review_checks_numpy_evidence_only_when_present() -> None:
    bundle = _publishable_bundle()
    bundle["expression"] = {
        **bundle["expression"],
        "evidence_json": {
            **bundle["expression"]["evidence_json"],
            "numpy_runtime": {
                "no_sympy_runtime": False,
                "runtime_imports": ["numpy", "sympy"],
                "tests_passed": False,
                "source": "import sympy\nimport numpy as np",
            },
        },
    }

    assessment = assess_publishability(**bundle)

    assert assessment.publishable is False
    assert assessment.achieved_status == "dimension_resolved"
    assert "generated NumPy evidence must assert no_sympy_runtime" in assessment.blockers
    assert "generated NumPy runtime evidence imports SymPy" in assessment.blockers
    assert "generated NumPy runtime tests did not pass" in assessment.blockers

    no_codegen_bundle = _publishable_bundle()
    no_codegen_bundle["expression"] = {
        **no_codegen_bundle["expression"],
        "evidence_json": {
            key: value
            for key, value in no_codegen_bundle["expression"]["evidence_json"].items()
            if key != "numpy_runtime"
        },
    }
    assert assess_publishability(**no_codegen_bundle).publishable is True


def test_require_publishable_raises_with_ordered_blockers() -> None:
    bundle = _publishable_bundle()
    bundle["expression"] = {**bundle["expression"], "review_status": "automated_pass"}

    with pytest.raises(ValueError, match="review_status must be human_reviewed"):
        require_publishable(**bundle)


def test_phase5_review_reports_needs_human_and_blocked_trust_states() -> None:
    needs_human_bundle = _publishable_bundle()
    needs_human_bundle["expression"] = {
        **needs_human_bundle["expression"],
        "review_status": "needs_human",
        "evidence_json": {
            key: value
            for key, value in needs_human_bundle["expression"]["evidence_json"].items()
            if key != "human_review"
        },
    }

    report = build_review_trust_report(**needs_human_bundle)

    assert report["publishable"] is False
    assert report["trust_status"] == "needs_human"
    assert report["needs_human"] is True
    assert "expression review_status needs human review" in report["blockers"]

    blocked_bundle = _publishable_bundle()
    blocked_bundle["candidate"] = {
        **blocked_bundle["candidate"],
        "candidate_status": "blocked",
    }
    blocked_bundle["expression"] = {
        **blocked_bundle["expression"],
        "review_status": "blocked",
    }

    blocked = assess_publishability(**blocked_bundle)

    assert blocked.publishable is False
    assert blocked.trust_status == "blocked"
    assert blocked.blocked is True
    assert "candidate_status is blocked" in blocked.blockers
    assert "expression review_status is blocked" in blocked.blockers


def test_phase5_review_summarizes_dashboard_rollups_json_safe() -> None:
    publishable = assess_publishability(**_publishable_bundle())

    needs_human_bundle = _publishable_bundle()
    needs_human_bundle["expression"] = {
        **needs_human_bundle["expression"],
        "review_status": "automated_pass",
        "evidence_json": {
            key: value
            for key, value in needs_human_bundle["expression"]["evidence_json"].items()
            if key != "human_review"
        },
    }
    needs_human = assess_publishability(**needs_human_bundle).to_report()

    blocked_bundle = _publishable_bundle()
    blocked_bundle["candidate"] = {
        **blocked_bundle["candidate"],
        "candidate_status": "blocked",
    }
    blocked = assess_publishability(**blocked_bundle).to_report().to_dict()

    summary = summarize_review_assessments([publishable, needs_human, blocked])

    assert json.loads(json.dumps(summary)) == summary
    assert summary == {
        "assessment_count": 3,
        "achieved_status_counts": {
            "raw_imported": 1,
            "source_verified": 1,
            "published": 1,
        },
        "trust_status_counts": {
            "needs_human": 1,
            "human_reviewed": 1,
            "blocked": 1,
        },
        "publishable_counts": {
            "publishable": 1,
            "not_publishable": 2,
            "unknown": 0,
        },
        "blocker_counts": {
            "candidate_status is blocked": 1,
            "expression review_status must be human_reviewed": 1,
            "human review evidence must identify a reviewer": 1,
            "human review evidence must include a review timestamp": 1,
        },
        "gate_counts": {
            "raw_imported": {"passed": 2, "failed": 1},
            "parsed": {"passed": 3, "failed": 0},
            "dimension_resolved": {"passed": 3, "failed": 0},
            "symbolically_validated": {"passed": 3, "failed": 0},
            "source_verified": {"passed": 3, "failed": 0},
            "human_reviewed": {"passed": 2, "failed": 1},
            "published": {"passed": 1, "failed": 2},
        },
    }


def _publishable_bundle() -> dict[str, object]:
    expression = {
        "raw_formula": "F = m a",
        "sympy_srepr": "Equality(Symbol('F'), Mul(Symbol('m'), Symbol('a')))",
        "parse_status": "normalized",
        "parse_confidence": 0.96,
        "canonical_expr_hash": "a" * 64,
        "topology_hash": "b" * 64,
        "dimensional_hash": "c" * 64,
        "validation_status": "passed",
        "review_status": "human_reviewed",
        "mechanism_tags": ["force_balance", "classical_mechanics"],
        "behavioral_archetypes": ["linear_proportionality"],
        "evidence_json": {
            "parse_roundtrip": {"status": "passed"},
            "dimensional_analysis": {"status": "passed"},
            "required_constants": ["standard_gravity"],
            "data_dependencies": ["calibration_table"],
            "numpy_runtime": {
                "no_sympy_runtime": True,
                "runtime_imports": ["numpy"],
                "tests_passed": True,
                "source": "import numpy as np\n\ndef force(m, a):\n    return m * a\n",
            },
            "human_review": {
                "reviewer_id": "phys-reviewer-1",
                "reviewed_at": "2026-04-30T12:00:00Z",
            },
        },
    }
    return {
        "candidate": {
            "raw_formula": "F = m a",
            "candidate_status": "human_reviewed",
            "parse_confidence": 0.96,
        },
        "expression": expression,
        "variables": [
            {
                "symbol_name": "F",
                "variable_role": "output",
                "dim_signature": "M1L1T-2",
                "dimension_source": "qudt",
            },
            {
                "symbol_name": "m",
                "variable_role": "input",
                "dim_signature": "M1",
                "dimension_source": "source",
            },
            {
                "symbol_name": "a",
                "variable_role": "input",
                "dim_signature": "L1T-2",
                "dimension_source": "inferred",
            },
        ],
        "io_specs": [
            {"name": "m", "direction": "input", "dim_signature": "M1"},
            {"name": "a", "direction": "input", "dim_signature": "L1T-2"},
            {"name": "F", "direction": "output", "dim_signature": "M1L1T-2"},
        ],
        "references": [
            {
                "title": "Newton fixture",
                "doi": "10.0000/fixture",
                "verified": True,
            }
        ],
        "validity_bounds": [
            {
                "variable_name": "m",
                "bound_kind": "domain",
                "lower_value": 0.0,
                "lower_inclusive": False,
                "validity_statement": "mass is positive",
                "review_status": "human_reviewed",
            }
        ],
        "relationships": [
            {
                "relationship_kind": "uses_constant",
                "relationship_label": "standard_gravity",
                "target_node_id": "constant:g0",
                "verified": True,
            },
            {
                "relationship_kind": "uses_data_artifact",
                "relationship_label": "calibration_table",
                "target_node_id": "data:calibration",
                "verified": True,
            },
        ],
    }
