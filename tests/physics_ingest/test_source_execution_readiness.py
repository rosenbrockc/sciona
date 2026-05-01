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
