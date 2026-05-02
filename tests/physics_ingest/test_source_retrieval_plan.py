from __future__ import annotations

import json

from sciona.physics_ingest.sources import build_physics_source_retrieval_manifest
from sciona.physics_ingest.sources.retrieval_plan import (
    PHASE7_RING_ORDER,
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


def test_retrieval_manifest_exposes_phase7_ring_metadata() -> None:
    manifest = build_physics_source_retrieval_manifest()
    endpoints = manifest.endpoint_by_id()
    jobs = manifest.job_by_id()

    assert endpoints["foundational_manual_seed"].phase7_ring == "ring_1_foundational"
    assert endpoints["nist_codata_constants"].phase7_ring == "ring_1_foundational"
    assert endpoints["qudt_units_quantity_kinds"].phase7_ring == "ring_1_foundational"
    assert endpoints["nist_dlmf_equations"].phase7_ring == "ring_1_foundational"
    assert endpoints["hitran_lines"].phase7_ring == "ring_2_existing_sciona_domains"
    assert endpoints["materials_project_documents"].phase7_ring == "ring_2_existing_sciona_domains"
    assert endpoints["opb_problem_payloads"].phase7_ring == "ring_2_existing_sciona_domains"
    assert endpoints["wikidata_equation_candidates"].phase7_ring == "ring_3_wikidata_physical_equations"
    assert endpoints["pdg_derivation_graph"].phase7_ring == "ring_4_pdg_derivations"
    assert endpoints["theoria_payloads"].phase7_ring == "ring_6_long_tail"
    assert endpoints["phy_srbench_payloads"].phase7_ring == "ring_6_long_tail"

    reference_ring = "ring_5_reference_datasets"
    assert reference_ring in endpoints["nist_codata_constants"].phase7_rings
    assert reference_ring in endpoints["qudt_units_quantity_kinds"].phase7_rings
    assert reference_ring in endpoints["hitran_lines"].phase7_rings
    assert reference_ring in endpoints["materials_project_documents"].phase7_rings
    assert reference_ring not in endpoints["nist_dlmf_equations"].phase7_rings

    for endpoint in manifest.endpoints:
        assert endpoint.phase7_ring_order == PHASE7_RING_ORDER[endpoint.phase7_ring]
        assert endpoint.phase7_rings == tuple(
            sorted(endpoint.phase7_rings, key=PHASE7_RING_ORDER.__getitem__)
        )
        job = jobs[f"{endpoint.endpoint_id}.backfill"]
        assert job.phase7_ring == endpoint.phase7_ring
        assert job.phase7_ring_order == endpoint.phase7_ring_order
        assert job.phase7_rings == endpoint.phase7_rings


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
    assert all("phase7_ring" in endpoint for endpoint in decoded["endpoints"])
    assert all("phase7_ring_order" in job for job in decoded["jobs"])
