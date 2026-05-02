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

    ring_counts = {
        bucket["key"]["phase7_ring"]: bucket["counts"]
        for bucket in report["by_phase7_ring"]
    }
    assert ring_counts["ring_1_foundational_physics"]["published"] == 1
    assert ring_counts["ring_3_wikidata_equations"]["discovered"] == 1
    assert ring_counts["ring_4_pdg_derivations"]["blocked"] == 1
    assert ring_counts["ring_5_reference_datasets"]["dimensioned"] == 1

    ring_family_counts = {
        (
            bucket["key"]["phase7_ring"],
            bucket["key"]["physics_family"],
        ): bucket["counts"]
        for bucket in report["by_phase7_ring_and_physics_family"]
    }
    assert ring_family_counts[
        ("ring_1_foundational_physics", "thermodynamics")
    ]["published"] == 1
    assert ring_family_counts[
        ("ring_3_wikidata_equations", "classical_mechanics")
    ]["discovered"] == 1
    assert ring_family_counts[("ring_4_pdg_derivations", "force_balance")][
        "blocked"
    ] == 1


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


def test_phase7_coverage_summary_groups_explicit_and_inferred_backfill_rings() -> None:
    report = build_phase7_coverage_summary_dict(
        [
            {
                "candidate_status": "published",
                "source_payload": {"phase7_ring": "Ring 2"},
                "physics_family": "imaging",
            },
            {
                "candidate_status": "parsed",
                "source_system": "wikidata",
                "physics_family": "electromagnetism",
            },
            {
                "candidate_status": "dimension_resolved",
                "source_family": "reference_data",
                "physics_family": "materials",
            },
            {
                "candidate_status": "raw_imported",
                "source_family": "lower_metadata_quality",
                "physics_family": "unclassified",
            },
            {
                "candidate_status": "raw_imported",
                "source_system": "manual",
                "physics_family": "unclassified",
            },
        ]
    )

    buckets = {
        bucket["key"]["phase7_ring"]: bucket for bucket in report["by_phase7_ring"]
    }

    assert list(buckets) == [
        "ring_2_existing_sciona_domains",
        "ring_3_wikidata_equations",
        "ring_5_reference_datasets",
        "ring_6_long_tail",
        "unknown",
    ]
    assert buckets["ring_2_existing_sciona_domains"]["counts"]["published"] == 1
    assert buckets["ring_3_wikidata_equations"]["counts"]["parsed"] == 1
    assert buckets["ring_5_reference_datasets"]["counts"]["dimensioned"] == 1
    assert buckets["ring_6_long_tail"]["counts"]["discovered"] == 1
    assert buckets["unknown"]["key"]["phase7_ring_label"] == "Unknown or unassigned"


def test_phase7_coverage_summary_drills_down_ring_by_physics_family() -> None:
    report = build_phase7_coverage_summary_dict(
        [
            {
                "candidate_status": "published",
                "source_payload": {"phase7_ring": "Ring 1"},
                "mechanism_tags": ["thermodynamics", "transport"],
            },
            {
                "parse_status": "parsed",
                "source_system": "wikidata",
                "physics_family": "electromagnetism",
            },
            {
                "parse_status": "parse_failed",
                "review_status": "blocked",
                "source_payload": {"source": "pdg"},
                "physics_family": "mechanics",
            },
        ]
    )

    buckets = report["by_phase7_ring_and_physics_family"]

    assert [
        (bucket["key"]["phase7_ring"], bucket["key"]["physics_family"])
        for bucket in buckets
    ] == [
        ("ring_1_foundational_physics", "thermodynamics"),
        ("ring_1_foundational_physics", "transport"),
        ("ring_3_wikidata_equations", "electromagnetism"),
        ("ring_4_pdg_derivations", "mechanics"),
    ]
    assert buckets[0]["key"]["phase7_ring_label"] == (
        "Foundational mechanics, thermodynamics, electromagnetism, waves, and transport"
    )
    assert buckets[0]["counts"]["published"] == 1
    assert buckets[1]["counts"]["published"] == 1
    assert buckets[2]["counts"]["parsed"] == 1
    assert buckets[3]["counts"]["blocked"] == 1


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
    assert report["by_phase7_ring"] == []
    assert report["by_physics_family"] == []
    assert report["by_source_and_physics_family"] == []
    assert report["by_phase7_ring_and_physics_family"] == []
