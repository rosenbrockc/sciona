from __future__ import annotations

from dataclasses import replace
import json
import sys
import types

from sciona.physics_ingest.sources import (
    SourceRetrievalManifest,
    build_physics_source_retrieval_manifest,
    build_source_adapter_coverage_report,
    build_source_adapter_coverage_report_dict,
)
from sciona.physics_ingest.sources import adapter_coverage


def test_source_adapter_coverage_reports_current_manifest_contracts() -> None:
    report = build_source_adapter_coverage_report()

    assert report.report_version == "physics-source-adapter-coverage.v1"
    assert report.summary == {
        "total_jobs": 11,
        "covered": 11,
        "metadata_only": 0,
        "blocked": 0,
        "manual": 1,
        "offline": 11,
        "diagnostic_count": 0,
    }

    jobs = {job.job_id: job for job in report.jobs}
    manual = jobs["foundational_manual_seed.backfill"]
    assert manual.manual is True
    assert manual.offline is True
    assert manual.supported is True
    assert manual.adapter_module == "sciona.physics_ingest.sources.foundational_physics"
    assert manual.adapter_version == "wave1.foundational_physics_backfill.v1"
    assert manual.target_adapter_input == "curated_seed_records"
    assert manual.offline_contract["available_contract"] == [
        "build_foundational_physics_backfill_bundle"
    ]

    pdg = jobs["pdg_derivation_graph.backfill"]
    assert pdg.adapter_metadata["adapter_version"] == "wave1.pdg_scaffold.v1"
    assert pdg.offline_contract["available_contract"] == ["parse_pdg_document"]

    wikidata = jobs["wikidata_equation_candidates.backfill"]
    assert wikidata.target_adapter_input == "json_records"
    assert wikidata.offline_contract["available_contract"] == [
        "build_snapshot_record",
        "build_wave0_candidate_records",
    ]


def test_source_adapter_coverage_reports_missing_module_and_contract() -> None:
    manifest = build_physics_source_retrieval_manifest()
    broken_job = replace(
        manifest.jobs[0],
        job_id="synthetic_missing_module.backfill",
        adapter_name="sciona.physics_ingest.sources.does_not_exist",
        adapter_version="0.0.1",
    )
    broken_manifest = SourceRetrievalManifest(
        manifest_version=manifest.manifest_version,
        snapshot_key_prefix=manifest.snapshot_key_prefix,
        endpoints=manifest.endpoints,
        jobs=(broken_job,),
    )

    report = build_source_adapter_coverage_report(broken_manifest)

    assert report.summary["blocked"] == 1
    assert report.jobs[0].supported is False
    assert report.jobs[0].offline is False
    assert report.jobs[0].adapter_metadata["importable"] is False
    assert [diagnostic.code for diagnostic in report.diagnostics] == [
        "missing_adapter_module",
        "missing_builder_readiness_contract",
    ]


def test_source_adapter_coverage_reports_missing_version_and_unknown_target(
    monkeypatch,
) -> None:
    module_name = "synthetic_adapter_without_version"
    module = types.ModuleType(module_name)

    def build_synthetic_bundle() -> None:
        return None

    module.build_synthetic_bundle = build_synthetic_bundle
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setitem(
        adapter_coverage._JOB_CONTRACTS,
        "synthetic_missing_version.backfill",
        ("build_synthetic_bundle",),
    )

    manifest = build_physics_source_retrieval_manifest()
    broken_job = replace(
        manifest.jobs[0],
        job_id="synthetic_missing_version.backfill",
        adapter_name=module_name,
        adapter_version="0.0.1",
        target_adapter_input="mystery_payload",
    )
    broken_manifest = SourceRetrievalManifest(
        manifest_version=manifest.manifest_version,
        snapshot_key_prefix=manifest.snapshot_key_prefix,
        endpoints=manifest.endpoints,
        jobs=(broken_job,),
    )

    report = build_source_adapter_coverage_report(broken_manifest)

    assert report.summary["blocked"] == 1
    assert report.jobs[0].adapter_metadata["importable"] is True
    assert report.jobs[0].offline is True
    assert [diagnostic.code for diagnostic in report.diagnostics] == [
        "missing_adapter_version",
        "unknown_target_adapter_input",
    ]


def test_source_adapter_coverage_report_dict_is_json_serializable() -> None:
    report_dict = build_source_adapter_coverage_report_dict()

    encoded = json.dumps(report_dict, sort_keys=True)
    decoded = json.loads(encoded)

    assert decoded == report_dict
    assert decoded["summary"]["total_jobs"] == 11
    assert all(isinstance(job["supported"], bool) for job in decoded["jobs"])
