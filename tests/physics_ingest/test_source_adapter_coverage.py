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
        "by_source_family": {
            "benchmark": 3,
            "curated_foundational": 1,
            "derivation_graph": 1,
            "knowledge_graph": 1,
            "materials": 1,
            "ontology": 1,
            "spectroscopy": 1,
            "standards_reference": 2,
        },
        "by_target_adapter_input": {
            "curated_seed_records": 1,
            "json_records": 3,
            "raw_document": 2,
            "raw_records": 5,
        },
        "by_status": {
            "blocked": 0,
            "covered": 11,
            "metadata_only": 0,
        },
        "by_supported": {
            "false": 0,
            "true": 11,
        },
        "by_offline": {
            "false": 0,
            "true": 11,
        },
        "by_manual": {
            "false": 10,
            "true": 1,
        },
        "by_adapter_module": {
            "sciona.physics_ingest.sources.foundational_physics": 1,
            "sciona.physics_ingest.sources.hitran": 1,
            "sciona.physics_ingest.sources.materials_project": 1,
            "sciona.physics_ingest.sources.nist": 2,
            "sciona.physics_ingest.sources.opb": 1,
            "sciona.physics_ingest.sources.pdg": 1,
            "sciona.physics_ingest.sources.phy_srbench": 1,
            "sciona.physics_ingest.sources.qudt": 1,
            "sciona.physics_ingest.sources.theoria": 1,
            "sciona.physics_ingest.sources.wikidata": 1,
        },
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


def test_source_adapter_coverage_rollups_include_blocked_and_metadata_only(
    monkeypatch,
) -> None:
    module_name = "synthetic_metadata_only_adapter"
    module = types.ModuleType(module_name)
    module.ADAPTER_NAME = "synthetic-metadata-only"
    module.ADAPTER_VERSION = "1.2.3"
    monkeypatch.setitem(sys.modules, module_name, module)

    manifest = build_physics_source_retrieval_manifest()
    blocked_job = replace(
        manifest.jobs[0],
        job_id="synthetic_missing_module.backfill",
        source_family="synthetic_family",
        adapter_name="sciona.physics_ingest.sources.does_not_exist",
        adapter_version="0.0.1",
        target_adapter_input="raw_records",
    )
    metadata_only_job = replace(
        manifest.jobs[1],
        job_id="synthetic_metadata_only.backfill",
        source_family="synthetic_family",
        adapter_name=module_name,
        adapter_version="1.2.3",
        target_adapter_input="raw_document",
    )
    synthetic_manifest = SourceRetrievalManifest(
        manifest_version=manifest.manifest_version,
        snapshot_key_prefix=manifest.snapshot_key_prefix,
        endpoints=manifest.endpoints,
        jobs=(blocked_job, metadata_only_job),
    )

    report = build_source_adapter_coverage_report(synthetic_manifest)

    assert report.summary == {
        "total_jobs": 2,
        "covered": 0,
        "metadata_only": 1,
        "blocked": 1,
        "manual": 0,
        "offline": 0,
        "diagnostic_count": 3,
        "by_source_family": {"synthetic_family": 2},
        "by_target_adapter_input": {
            "raw_document": 1,
            "raw_records": 1,
        },
        "by_status": {
            "blocked": 1,
            "covered": 0,
            "metadata_only": 1,
        },
        "by_supported": {
            "false": 1,
            "true": 1,
        },
        "by_offline": {
            "false": 2,
            "true": 0,
        },
        "by_manual": {
            "false": 2,
            "true": 0,
        },
        "by_adapter_module": {
            "sciona.physics_ingest.sources.does_not_exist": 1,
            "synthetic_metadata_only_adapter": 1,
        },
    }
    assert [job.status for job in report.jobs] == ["metadata_only", "blocked"]
    assert [diagnostic.code for diagnostic in report.diagnostics] == [
        "missing_builder_readiness_contract",
        "missing_adapter_module",
        "missing_builder_readiness_contract",
    ]


def test_source_adapter_coverage_report_dict_is_json_serializable() -> None:
    report_dict = build_source_adapter_coverage_report_dict()

    encoded = json.dumps(report_dict, sort_keys=True)
    decoded = json.loads(encoded)

    assert decoded == report_dict
    assert decoded["summary"]["total_jobs"] == 11
    assert all(isinstance(job["supported"], bool) for job in decoded["jobs"])
