from __future__ import annotations

from dataclasses import dataclass
import json

from sciona.physics_ingest import (
    build_phase7_coverage_summary,
    build_phase7_coverage_summary_dict,
)


def test_phase7_coverage_summary_groups_source_and_physics_family() -> None:
    rows = [
        {
            "candidate_status": "raw_imported",
            "source_payload": {
                "source_system": "wikidata",
                "source_family": "knowledge_graph",
            },
            "mechanism_tags": ["classical_mechanics"],
        },
        {
            "candidate_status": "dimension_resolved",
            "source_payload": {
                "source_system": "nist_codata",
                "source_family": "reference_data",
            },
            "mechanism_tags": ["metrology"],
        },
        {
            "parse_status": "normalized",
            "dimensional_hash": "dimension-hash",
            "review_status": "human_reviewed",
            "validation_status": "passed",
            "publication_status": "published",
            "source_system": "manual",
            "source_family": "curated_seed",
            "physics_family": "thermodynamics",
        },
        {
            "parse_status": "parse_failed",
            "review_status": "blocked",
            "source_payload": {
                "source": "pdg",
                "source_family": "derivation_graph",
            },
            "mechanism_tags": ["dynamics", "force_balance"],
        },
    ]

    report = build_phase7_coverage_summary(rows).to_dict()

    assert report["summary"] == {
        "total_rows": 4,
        "discovered": 4,
        "parsed": 2,
        "dimensioned": 2,
        "reviewed": 1,
        "published": 1,
        "blocked": 1,
        "metrics": {
            "parsed_rate": 0.5,
            "dimensioned_rate": 0.5,
            "reviewed_rate": 0.25,
            "published_rate": 0.25,
            "blocked_rate": 0.25,
            "discovered_to_parsed_loss": 2,
            "parsed_to_dimensioned_loss": 0,
            "dimensioned_to_reviewed_loss": 1,
            "reviewed_to_published_loss": 0,
        },
    }
    assert report["by_source"] == [
        {
            "key": {"source_system": "manual", "source_family": "curated_seed"},
            "counts": {
                "discovered": 1,
                "parsed": 1,
                "dimensioned": 1,
                "reviewed": 1,
                "published": 1,
                "blocked": 0,
            },
            "metrics": {
                "parsed_rate": 1.0,
                "dimensioned_rate": 1.0,
                "reviewed_rate": 1.0,
                "published_rate": 1.0,
                "blocked_rate": 0.0,
                "discovered_to_parsed_loss": 0,
                "parsed_to_dimensioned_loss": 0,
                "dimensioned_to_reviewed_loss": 0,
                "reviewed_to_published_loss": 0,
            },
        },
        {
            "key": {"source_system": "nist_codata", "source_family": "reference_data"},
            "counts": {
                "discovered": 1,
                "parsed": 1,
                "dimensioned": 1,
                "reviewed": 0,
                "published": 0,
                "blocked": 0,
            },
            "metrics": {
                "parsed_rate": 1.0,
                "dimensioned_rate": 1.0,
                "reviewed_rate": 0.0,
                "published_rate": 0.0,
                "blocked_rate": 0.0,
                "discovered_to_parsed_loss": 0,
                "parsed_to_dimensioned_loss": 0,
                "dimensioned_to_reviewed_loss": 1,
                "reviewed_to_published_loss": 0,
            },
        },
        {
            "key": {"source_system": "pdg", "source_family": "derivation_graph"},
            "counts": {
                "discovered": 1,
                "parsed": 0,
                "dimensioned": 0,
                "reviewed": 0,
                "published": 0,
                "blocked": 1,
            },
            "metrics": {
                "parsed_rate": 0.0,
                "dimensioned_rate": 0.0,
                "reviewed_rate": 0.0,
                "published_rate": 0.0,
                "blocked_rate": 1.0,
                "discovered_to_parsed_loss": 1,
                "parsed_to_dimensioned_loss": 0,
                "dimensioned_to_reviewed_loss": 0,
                "reviewed_to_published_loss": 0,
            },
        },
        {
            "key": {"source_system": "wikidata", "source_family": "knowledge_graph"},
            "counts": {
                "discovered": 1,
                "parsed": 0,
                "dimensioned": 0,
                "reviewed": 0,
                "published": 0,
                "blocked": 0,
            },
            "metrics": {
                "parsed_rate": 0.0,
                "dimensioned_rate": 0.0,
                "reviewed_rate": 0.0,
                "published_rate": 0.0,
                "blocked_rate": 0.0,
                "discovered_to_parsed_loss": 1,
                "parsed_to_dimensioned_loss": 0,
                "dimensioned_to_reviewed_loss": 0,
                "reviewed_to_published_loss": 0,
            },
        },
    ]

    family_counts = {
        bucket["key"]["physics_family"]: bucket["counts"]
        for bucket in report["by_physics_family"]
    }
    assert family_counts["thermodynamics"]["published"] == 1
    assert family_counts["metrology"]["dimensioned"] == 1
    assert family_counts["classical_mechanics"]["discovered"] == 1
    assert family_counts["dynamics"]["blocked"] == 1
    assert family_counts["force_balance"]["blocked"] == 1


def test_phase7_coverage_summary_accepts_row_like_objects_and_nested_evidence() -> None:
    @dataclass(frozen=True)
    class RowObject:
        parse_status: str
        sympy_srepr: str
        evidence_json: dict[str, object]
        review_status: str
        source_payload: dict[str, str]
        mechanism_tags: tuple[str, ...]

    rows = [
        RowObject(
            parse_status="parsed",
            sympy_srepr="Equality(Symbol('F'), Mul(Symbol('m'), Symbol('a')))",
            evidence_json={"dimensional_analysis": {"status": "passed"}},
            review_status="automated_pass",
            source_payload={
                "source_system": "foundational_manual_seed",
                "source_family": "manual",
            },
            mechanism_tags=("force_balance",),
        )
    ]

    report = build_phase7_coverage_summary(rows)

    assert report.summary == {
        "total_rows": 1,
        "discovered": 1,
        "parsed": 1,
        "dimensioned": 1,
        "reviewed": 1,
        "published": 0,
        "blocked": 0,
        "metrics": {
            "parsed_rate": 1.0,
            "dimensioned_rate": 1.0,
            "reviewed_rate": 1.0,
            "published_rate": 0.0,
            "blocked_rate": 0.0,
            "discovered_to_parsed_loss": 0,
            "parsed_to_dimensioned_loss": 0,
            "dimensioned_to_reviewed_loss": 0,
            "reviewed_to_published_loss": 1,
        },
    }
    assert report.by_source[0].key == {
        "source_system": "foundational_manual_seed",
        "source_family": "manual",
    }


def test_phase7_coverage_summary_dict_is_json_serializable() -> None:
    report_dict = build_phase7_coverage_summary_dict(
        [
            {
                "candidate_status": "published",
                "source_payload": {"source_family": "knowledge_graph"},
                "mechanism_tags": ["wave_propagation"],
            }
        ]
    )

    encoded = json.dumps(report_dict, sort_keys=True)
    decoded = json.loads(encoded)

    assert decoded == report_dict
    assert decoded["report_version"] == "physics-phase7-coverage-summary.v1"
    assert decoded["summary"]["published"] == 1
    assert decoded["summary"]["metrics"]["published_rate"] == 1.0


def test_phase7_coverage_summary_zero_rows_has_stable_metrics() -> None:
    report = build_phase7_coverage_summary_dict([])

    assert report["summary"] == {
        "total_rows": 0,
        "discovered": 0,
        "parsed": 0,
        "dimensioned": 0,
        "reviewed": 0,
        "published": 0,
        "blocked": 0,
        "metrics": {
            "parsed_rate": 0.0,
            "dimensioned_rate": 0.0,
            "reviewed_rate": 0.0,
            "published_rate": 0.0,
            "blocked_rate": 0.0,
            "discovered_to_parsed_loss": 0,
            "parsed_to_dimensioned_loss": 0,
            "dimensioned_to_reviewed_loss": 0,
            "reviewed_to_published_loss": 0,
        },
    }
    assert report["by_source"] == []
    assert report["by_physics_family"] == []
    assert report["by_source_and_physics_family"] == []
