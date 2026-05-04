from __future__ import annotations

import json
from dataclasses import replace

from sciona.physics_ingest import build_physics_source_retrieval_run_plan
from sciona.physics_ingest.sources import (
    SourceRetrievalManifest,
    build_physics_source_retrieval_manifest,
    build_physics_source_retrieval_run_plan_dict,
)


def test_retrieval_run_plan_builds_ordered_dry_run_steps() -> None:
    plan = build_physics_source_retrieval_run_plan(max_jobs=3)
    plan_dict = plan.to_dict()

    assert plan.dry_run is True
    assert [step.job_id for step in plan.steps] == [
        "foundational_manual_seed.backfill",
        "nist_codata_constants.backfill",
        "qudt_units_quantity_kinds.backfill",
    ]
    assert all(step.dry_run is True for step in plan.steps)
    assert all(step.replay_key.startswith("physics-source-retrieval:") for step in plan.steps)

    manual = plan.steps[0]
    assert manual.url == "manual://sciona.physics_ingest/foundational_physics/v1"
    assert manual.method == "MANUAL"
    assert manual.adapter_module == "sciona.physics_ingest.sources.foundational_physics"
    assert manual.adapter_version == "wave1.foundational_physics_backfill.v1"
    assert manual.target_adapter_input == "curated_seed_records"
    assert manual.phase7_ring == "ring_1_foundational"
    assert manual.phase7_ring_order == 1
    assert manual.phase7_rings == ("ring_1_foundational",)
    assert manual.provenance["license_expression"]
    assert manual.retry_policy["max_attempts"] == 1
    assert plan_dict["summary"] == {
        "step_count": 3,
        "diagnostic_count": 0,
        "dry_run": True,
        "dry_run_step_count": 3,
        "non_dry_run_step_count": 0,
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
        "by_endpoint_kind": {
            "ascii_table": 1,
            "curated_seed": 1,
            "rdf_dump": 1,
        },
        "by_method": {"GET": 2, "MANUAL": 1},
        "by_target_adapter_input": {
            "curated_seed_records": 1,
            "raw_document": 1,
            "raw_records": 1,
        },
    }


def test_retrieval_run_plan_filters_and_limit_do_not_mutate_manifest() -> None:
    manifest = build_physics_source_retrieval_manifest()
    original_limit = manifest.job_by_id()["wikidata_equation_candidates.backfill"].limit

    plan = build_physics_source_retrieval_run_plan(
        manifest=manifest,
        source_system="wikidata",
        source_family="knowledge_graph",
        phase7_ring="ring_3_wikidata_physical_equations",
        job_id="wikidata_equation_candidates.backfill",
        limit=25,
    )

    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.job_id == "wikidata_equation_candidates.backfill"
    assert step.params["LIMIT"] == 25
    assert step.paging["limit"] == 25
    assert plan.filters["source_system"] == ["wikidata"]
    assert plan.filters["source_family"] == ["knowledge_graph"]
    assert plan.filters["phase7_ring"] == ["ring_3_wikidata_physical_equations"]
    assert plan.filters["job_id"] == ["wikidata_equation_candidates.backfill"]
    assert manifest.job_by_id()["wikidata_equation_candidates.backfill"].limit == original_limit
    assert plan.to_dict()["summary"] == {
        "step_count": 1,
        "diagnostic_count": 0,
        "dry_run": True,
        "dry_run_step_count": 1,
        "non_dry_run_step_count": 0,
        "by_source_family": {"knowledge_graph": 1},
        "by_source_system": {"wikidata": 1},
        "by_phase7_ring": {"ring_3_wikidata_physical_equations": 1},
        "by_endpoint_kind": {"sparql": 1},
        "by_method": {"POST": 1},
        "by_target_adapter_input": {"json_records": 1},
    }


def test_retrieval_run_plan_filters_by_phase7_ring_without_mutating_manifest() -> None:
    manifest = build_physics_source_retrieval_manifest()
    original_jobs = [job.to_dict() for job in manifest.jobs]

    foundational = build_physics_source_retrieval_run_plan(
        manifest=manifest,
        phase7_ring="ring_1_foundational",
    )
    assert [step.source_system for step in foundational.steps] == [
        "manual",
        "nist_codata",
        "qudt",
        "nist_dlmf",
    ]
    assert all(step.phase7_ring == "ring_1_foundational" for step in foundational.steps)
    assert foundational.filters["phase7_ring"] == ["ring_1_foundational"]

    reference = build_physics_source_retrieval_run_plan(
        manifest=manifest,
        phase7_ring="ring_5_reference_datasets",
    )
    assert [step.source_system for step in reference.steps] == [
        "nist_codata",
        "qudt",
        "hitran",
        "materials_project",
    ]
    assert all(
        "ring_5_reference_datasets" in step.phase7_rings
        for step in reference.steps
    )
    assert reference.steps[0].phase7_ring == "ring_1_foundational"
    assert reference.steps[2].phase7_ring == "ring_2_existing_sciona_domains"
    assert [job.to_dict() for job in manifest.jobs] == original_jobs


def test_retrieval_run_plan_replay_keys_are_deterministic() -> None:
    first = build_physics_source_retrieval_run_plan(
        source_system=("nist_codata", "qudt"),
        limit=10,
    )
    second = build_physics_source_retrieval_run_plan(
        source_system=("nist_codata", "qudt"),
        limit=10,
    )

    assert [step.to_dict() for step in first.steps] == [
        step.to_dict() for step in second.steps
    ]
    assert [step.replay_key for step in first.steps] == [
        step.replay_key for step in second.steps
    ]


def test_retrieval_run_step_request_envelope_is_json_safe_and_deterministic() -> None:
    first = build_physics_source_retrieval_run_plan(
        job_id="wikidata_equation_candidates.backfill",
        limit=25,
    )
    second = build_physics_source_retrieval_run_plan(
        job_id="wikidata_equation_candidates.backfill",
        limit=25,
    )

    step = first.steps[0]
    envelope = step.request_envelope
    encoded = json.dumps(envelope, sort_keys=True)
    decoded = json.loads(encoded)

    assert decoded == envelope
    assert envelope == second.steps[0].request_envelope
    assert envelope["envelope_version"] == "physics-source-request-envelope.v1"
    assert envelope["method"] == step.method == "POST"
    assert envelope["url"] == step.url == "https://query.wikidata.org/sparql"
    assert envelope["headers"] == step.headers
    assert envelope["params"] == step.params == {"LIMIT": 25}
    assert envelope["body_template"] == step.body_template
    assert envelope["paging"] == step.paging
    assert envelope["auth"] == {
        "requires_auth": False,
        "auth_hint": "",
    }
    assert envelope["storage"] == {
        "snapshot_key": step.snapshot_key,
        "target_adapter_input": "json_records",
        "storage_required": True,
        "write_required": False,
        "side_effect_free": True,
    }
    assert envelope["provenance"] == step.provenance
    assert envelope["retry_policy"] == step.retry_policy
    assert envelope["rate_limit"] == step.rate_limit
    assert envelope["replay_key"] == step.replay_key
    assert envelope["snapshot_key"] == step.snapshot_key
    assert envelope["adapter_target"] == {
        "module": "sciona.physics_ingest.sources.wikidata",
        "version": "0.1.0",
        "target_input": "json_records",
    }
    assert envelope["execution"] == {
        "mode": "network",
        "dry_run": True,
        "io_performed": False,
        "network_required": True,
        "network_io_allowed": False,
        "manual_source": False,
    }


def test_manual_retrieval_request_envelope_remains_non_network_manual() -> None:
    plan = build_physics_source_retrieval_run_plan(
        job_id="foundational_manual_seed.backfill",
    )

    step = plan.steps[0]
    envelope = step.request_envelope

    assert step.method == "MANUAL"
    assert step.url == "manual://sciona.physics_ingest/foundational_physics/v1"
    assert envelope["method"] == "MANUAL"
    assert envelope["url"].startswith("manual://")
    assert envelope["execution"] == {
        "mode": "manual",
        "dry_run": True,
        "io_performed": False,
        "network_required": False,
        "network_io_allowed": False,
        "manual_source": True,
    }
    assert envelope["storage"]["storage_required"] is False
    assert envelope["storage"]["write_required"] is False
    assert envelope["retry_policy"]["max_attempts"] == 1
    assert envelope["rate_limit"]["requests_per_second"] == 0.0
    assert envelope["adapter_target"] == {
        "module": "sciona.physics_ingest.sources.foundational_physics",
        "version": "wave1.foundational_physics_backfill.v1",
        "target_input": "curated_seed_records",
    }


def test_retrieval_run_plan_warns_on_incomplete_endpoint_or_adapter_metadata() -> None:
    manifest = build_physics_source_retrieval_manifest()
    endpoint = manifest.endpoints[0]
    job = manifest.jobs[0]
    incomplete_endpoint = replace(
        endpoint,
        url="",
        adapter_name="",
        adapter_version="",
    )
    incomplete_job = replace(
        job,
        adapter_name="",
        adapter_version="",
    )
    incomplete_manifest = SourceRetrievalManifest(
        manifest_version=manifest.manifest_version,
        snapshot_key_prefix=manifest.snapshot_key_prefix,
        endpoints=(incomplete_endpoint,),
        jobs=(incomplete_job,),
    )

    plan = build_physics_source_retrieval_run_plan(manifest=incomplete_manifest)

    assert len(plan.steps) == 1
    assert "endpoint url is missing" in plan.steps[0].warnings
    assert "adapter module is missing" in plan.steps[0].warnings
    assert "adapter version is missing" in plan.steps[0].warnings
    assert [diagnostic.message for diagnostic in plan.diagnostics] == list(
        plan.steps[0].warnings
    )
    assert plan.to_dict()["summary"] == {
        "step_count": 1,
        "diagnostic_count": 3,
        "dry_run": True,
        "dry_run_step_count": 1,
        "non_dry_run_step_count": 0,
        "by_source_family": {"knowledge_graph": 1},
        "by_source_system": {"wikidata": 1},
        "by_phase7_ring": {"ring_3_wikidata_physical_equations": 1},
        "by_endpoint_kind": {"sparql": 1},
        "by_method": {"POST": 1},
        "by_target_adapter_input": {"json_records": 1},
    }


def test_retrieval_run_plan_dict_is_json_safe() -> None:
    plan_dict = build_physics_source_retrieval_run_plan_dict(max_jobs=2, limit=5)

    encoded = json.dumps(plan_dict, sort_keys=True)
    decoded = json.loads(encoded)

    assert decoded == plan_dict
    assert decoded["dry_run"] is True
    assert len(decoded["steps"]) == 2
    assert decoded["steps"][0]["paging"]["limit"] == 5
    assert decoded["steps"][0]["phase7_ring"] == "ring_1_foundational"
    assert decoded["filters"]["phase7_ring"] is None
    assert "diagnostics" in decoded
    assert decoded["summary"]["step_count"] == 2
    assert decoded["summary"]["dry_run_step_count"] == 2
    assert decoded["summary"]["by_method"] == {"GET": 1, "MANUAL": 1}
