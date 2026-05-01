from __future__ import annotations

import json

from sciona.physics_ingest.sources import build_physics_source_retrieval_manifest
from sciona.physics_ingest.sources.retrieval_plan import (
    build_physics_source_retrieval_manifest_dict,
)


def test_retrieval_manifest_covers_current_physics_source_set() -> None:
    manifest = build_physics_source_retrieval_manifest()

    assert {endpoint.source_system for endpoint in manifest.endpoints} == {
        "wikidata",
        "qudt",
        "physics_derivation_graph",
        "nist_codata",
        "nist_dlmf",
        "hitran",
        "materials_project",
        "opb",
        "theoria",
        "phy_srbench",
        "manual",
    }
    assert len(manifest.jobs) == len(manifest.endpoints)
    assert all(endpoint.snapshot_key.startswith("physics-ingest/") for endpoint in manifest.endpoints)


def test_retrieval_manifest_declares_pagination_rate_limits_and_retries() -> None:
    manifest = build_physics_source_retrieval_manifest()
    endpoints = manifest.endpoint_by_id()

    wikidata = endpoints["wikidata_equation_candidates"]
    assert wikidata.method == "POST"
    assert wikidata.pagination.strategy == "offset_limit"
    assert wikidata.pagination.cursor_parameter == "OFFSET"
    assert wikidata.pagination.limit_parameter == "LIMIT"
    assert wikidata.pagination.default_limit == 500
    assert wikidata.rate_limit.requests_per_second <= 0.1
    assert 429 in wikidata.retry_policy.retry_on_statuses

    materials = endpoints["materials_project_documents"]
    assert materials.requires_auth is True
    assert materials.pagination.cursor_parameter == "_skip"
    assert materials.pagination.limit_parameter == "_limit"
    assert materials.auth_hint

    manual = endpoints["foundational_manual_seed"]
    assert manual.method == "MANUAL"
    assert manual.pagination.strategy == "in_memory_seed_set"
    assert manual.retry_policy.max_attempts == 1
    assert manual.retry_policy.retry_on_statuses == ()


def test_retrieval_manifest_jobs_link_to_endpoints_and_adapter_inputs() -> None:
    manifest = build_physics_source_retrieval_manifest(snapshot_key_prefix="wave-a")
    endpoints = manifest.endpoint_by_id()

    for job in manifest.jobs:
        endpoint = endpoints[job.endpoint_id]
        assert job.snapshot_key == endpoint.snapshot_key
        assert job.adapter_name == endpoint.adapter_name
        assert job.adapter_version == endpoint.adapter_version
        assert job.provenance["license_expression"] == endpoint.license_expression
        assert job.provenance["provenance_summary"] == endpoint.provenance_summary

    jobs = manifest.job_by_id()
    assert jobs["nist_codata_constants.backfill"].target_adapter_input == "raw_document"
    assert jobs["qudt_units_quantity_kinds.backfill"].target_adapter_input == "raw_records"
    assert jobs["foundational_manual_seed.backfill"].target_adapter_input == "curated_seed_records"
    assert jobs["wikidata_equation_candidates.backfill"].limit == 500
    assert jobs["wikidata_equation_candidates.backfill"].snapshot_key == "wave-a/wikidata"


def test_retrieval_manifest_dict_is_json_safe_and_side_effect_free() -> None:
    manifest_dict = build_physics_source_retrieval_manifest_dict()

    encoded = json.dumps(manifest_dict, sort_keys=True)
    decoded = json.loads(encoded)

    assert decoded == manifest_dict
    assert decoded["manifest_version"] == "physics-source-retrieval-plan.v1"
    assert "endpoints" in decoded
    assert "jobs" in decoded
    assert all("url" in endpoint for endpoint in decoded["endpoints"])
    assert all("retry_policy" in endpoint for endpoint in decoded["endpoints"])
