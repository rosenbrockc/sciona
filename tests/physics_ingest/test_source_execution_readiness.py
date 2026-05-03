from __future__ import annotations

import json
from dataclasses import replace

from sciona.physics_ingest import (
    build_physics_source_retrieval_run_plan,
    build_source_execution_readiness_report,
)
from sciona.physics_ingest.sources import (
    SourceRetrievalManifest,
    build_physics_source_retrieval_manifest,
    build_source_execution_readiness_report_dict,
)


def test_source_execution_readiness_reports_valid_plan_steps() -> None:
    plan = build_physics_source_retrieval_run_plan(max_jobs=3, limit=5)

    report = build_source_execution_readiness_report(plan)

    assert report.report_version == "physics-source-execution-readiness.v1"
    assert report.dry_run is True
    assert report.summary == {
        "total_steps": 3,
        "executable": 2,
        "offline_blocked": 0,
        "manual": 1,
        "diagnostic_count": 0,
        "by_source_family": {
            "curated_foundational": 1,
            "ontology": 1,
            "standards_reference": 1,
        },
        "by_source_system": {
            "manual": 1,
            "nist_codata": 1,
            "qudt": 1,
        },
        "by_phase7_ring": {"ring_1_foundational": 3},
        "by_status": {
            "executable": 2,
            "manual": 1,
        },
        "payload_requirements": {
            "offline_payload_available": 1,
            "payload_required": 2,
        },
        "storage_requirements": {
            "side_effect_free": 3,
            "storage_required": 2,
            "write_required": 0,
        },
    }
    manual = report.steps[0]
    assert manual.status == "manual"
    assert manual.required_adapter == {
        "name": "sciona.physics_ingest.sources.foundational_physics",
        "version": "wave1.foundational_physics_backfill.v1",
        "target_input": "curated_seed_records",
    }
    assert manual.endpoint["url"] == "manual://sciona.physics_ingest/foundational_physics/v1"
    assert manual.replay_key.startswith("physics-source-retrieval:")
    assert manual.payload_expectation["payload_required"] is False
    assert manual.storage_expectation["side_effect_free"] is True

    nist = report.steps[1]
    assert nist.status == "executable"
    assert nist.endpoint["method"] == "GET"
    assert nist.endpoint["paging"]["limit"] == 5
    assert nist.payload_expectation["target_adapter_input"] == "raw_document"
    assert nist.storage_expectation["storage_required"] is True


def test_source_execution_readiness_accepts_dict_input() -> None:
    plan = build_physics_source_retrieval_run_plan(
        job_id="wikidata_equation_candidates.backfill",
    )

    report = build_source_execution_readiness_report(plan.to_dict())

    assert report.summary["total_steps"] == 1
    assert report.steps[0].status == "executable"
    assert report.steps[0].required_adapter["name"] == "sciona.physics_ingest.sources.wikidata"
    assert report.steps[0].endpoint["url"] == "https://query.wikidata.org/sparql"
    assert report.steps[0].payload_expectation["payload_required"] is True


def test_source_execution_readiness_reports_diagnostics_deterministically() -> None:
    manifest = build_physics_source_retrieval_manifest()
    endpoint = manifest.endpoints[0]
    job = manifest.jobs[0]
    broken_manifest = SourceRetrievalManifest(
        manifest_version=manifest.manifest_version,
        snapshot_key_prefix=manifest.snapshot_key_prefix,
        endpoints=(
            replace(
                endpoint,
                adapter_name="",
                adapter_version="",
                url="",
            ),
        ),
        jobs=(
            replace(
                job,
                job_id="unsupported.backfill",
                adapter_name="",
                adapter_version="",
            ),
        ),
    )
    plan = build_physics_source_retrieval_run_plan(
        manifest=broken_manifest,
        dry_run=False,
    )

    report = build_source_execution_readiness_report(plan)

    assert report.summary == {
        "total_steps": 1,
        "executable": 0,
        "offline_blocked": 1,
        "manual": 0,
        "diagnostic_count": 6,
        "by_source_family": {"knowledge_graph": 1},
        "by_source_system": {"wikidata": 1},
        "by_phase7_ring": {"ring_3_wikidata_physical_equations": 1},
        "by_status": {"offline_blocked": 1},
        "payload_requirements": {
            "offline_payload_available": 0,
            "payload_required": 1,
        },
        "storage_requirements": {
            "side_effect_free": 1,
            "storage_required": 1,
            "write_required": 0,
        },
    }
    assert report.steps[0].status == "offline_blocked"
    assert [diagnostic.code for diagnostic in report.diagnostics] == [
        "non_dry_run_plan",
        "non_dry_run_plan",
        "unsupported_job_id",
        "missing_adapter_name",
        "missing_adapter_version",
        "missing_endpoint_url",
    ]
    assert [diagnostic.code for diagnostic in report.steps[0].diagnostics] == [
        "non_dry_run_plan",
        "unsupported_job_id",
        "missing_adapter_name",
        "missing_adapter_version",
        "missing_endpoint_url",
    ]


def test_source_execution_readiness_report_dict_is_json_serializable() -> None:
    plan = build_physics_source_retrieval_run_plan(max_jobs=2)

    report_dict = build_source_execution_readiness_report_dict(plan)
    encoded = json.dumps(report_dict, sort_keys=True)
    decoded = json.loads(encoded)

    assert decoded == report_dict
    assert decoded["summary"]["total_steps"] == 2
    assert decoded["steps"][0]["storage_expectation"]["write_required"] is False


def test_source_execution_readiness_report_dict_includes_phase7_metadata() -> None:
    plan = build_physics_source_retrieval_run_plan(
        phase7_ring="ring_5_reference_datasets",
    )

    report_dict = build_source_execution_readiness_report_dict(plan)
    encoded = json.dumps(report_dict, sort_keys=True)
    decoded = json.loads(encoded)

    assert decoded == report_dict
    assert decoded["summary"]["total_steps"] == 4
    assert all(
        "ring_5_reference_datasets" in step["phase7_rings"]
        for step in decoded["steps"]
    )
    assert decoded["steps"][0]["phase7_ring"] == "ring_1_foundational"
    assert decoded["steps"][0]["phase7_ring_order"] == 1
    assert decoded["steps"][2]["phase7_ring"] == "ring_2_existing_sciona_domains"
    assert decoded["steps"][2]["phase7_ring_order"] == 2
    assert decoded["summary"]["by_phase7_ring"] == {
        "ring_1_foundational": 2,
        "ring_2_existing_sciona_domains": 2,
    }
    assert decoded["summary"]["by_source_family"] == {
        "materials": 1,
        "ontology": 1,
        "spectroscopy": 1,
        "standards_reference": 1,
    }


def test_source_execution_readiness_rolls_up_phase7_rings_by_source_family() -> None:
    plan = build_physics_source_retrieval_run_plan(source_family="benchmark")

    report = build_source_execution_readiness_report(plan)

    assert report.summary["total_steps"] == 3
    assert report.summary["by_source_family"] == {"benchmark": 3}
    assert report.summary["by_source_system"] == {
        "opb": 1,
        "phy_srbench": 1,
        "theoria": 1,
    }
    assert report.summary["by_phase7_ring"] == {
        "ring_2_existing_sciona_domains": 1,
        "ring_6_long_tail": 2,
    }
