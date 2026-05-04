from __future__ import annotations

from copy import deepcopy
import json

from sciona.physics_ingest.deployment import (
    build_physics_ingest_deployment_storage_bundle,
)
from sciona.physics_ingest.deployment_runtime import (
    DEPLOYMENT_RUNTIME_REPORT_VERSION,
    build_physics_ingest_deployment_runtime_report,
    build_physics_ingest_deployment_runtime_report_dict,
)
from sciona.physics_ingest.sources import (
    build_physics_source_retrieval_run_plan,
    build_source_retrieval_runtime_execution_report_dict,
)


def test_deployment_runtime_report_generates_source_runtime_preflight_for_multi_job_plan() -> None:
    plan = build_physics_source_retrieval_run_plan(max_jobs=3, limit=5)

    report = build_physics_ingest_deployment_runtime_report_dict(
        source_retrieval_run_plan=plan,
    )

    assert report["report_version"] == DEPLOYMENT_RUNTIME_REPORT_VERSION
    assert report["ok"] is True
    assert report["blocked"] is False
    assert report["side_effect_free"] is True
    assert report["preflight"] is True
    assert report["flags"] == {
        "side_effect_free": True,
        "preflight": True,
        "source_runtime_preflight": True,
        "storage_preflight": False,
        "execution_performed": False,
        "db_write_performed": False,
    }
    assert report["source_runtime_counts"]["total_steps"] == 3
    assert report["source_runtime_counts"]["dry_run_step_count"] == 3
    assert report["summary"]["source_runtime_step_count"] == 3
    assert report["dashboard_summary"] == {
        "report_version": "physics-ingest-deployment-runtime-dashboard.v1",
        "ok": True,
        "blocked": False,
        "side_effect_free": True,
        "preflight": True,
        "source_runtime": {
            "step_count": 3,
            "network_step_count": report["summary"][
                "source_runtime_network_step_count"
            ],
            "manual_step_count": report["summary"]["source_runtime_manual_step_count"],
            "dry_run_step_count": 3,
            "requires_http_client_count": 0,
            "requires_snapshot_sink_count": 0,
            "requires_auth_count": 0,
            "execution_result_count": 0,
        },
        "storage": {
            "table_count": 0,
            "total_row_count": 0,
            "table_row_counts": {},
        },
        "diagnostics": {
            "diagnostic_count": 0,
            "blocking_diagnostic_count": 0,
            "by_stage": {},
            "by_severity": {},
            "blocking_by_code": {},
        },
        "replay": {
            "source_runtime_count": 3,
            "source_runtime_digest": report["replay_keys"][
                "source_runtime_digest"
            ],
        },
    }
    assert report["source_runtime_execution_preflight"]["execution_requested"] is True
    assert report["source_runtime_execution_preflight"]["execution_performed"] is False
    assert report["source_runtime_execution_preflight"]["execution_skipped_reason"] == (
        "preflight_only"
    )
    assert report["replay_keys"]["source_runtime"] == [
        step.replay_key for step in plan.steps
    ]
    assert len(report["replay_keys"]["source_runtime_digest"]) == 64


def test_deployment_runtime_report_accepts_existing_preflight_without_mutation() -> None:
    plan = build_physics_source_retrieval_run_plan(max_jobs=2)
    source_preflight = build_source_retrieval_runtime_execution_report_dict(
        plan,
        execute=True,
        preflight=True,
    )
    original = deepcopy(source_preflight)

    report = build_physics_ingest_deployment_runtime_report_dict(
        source_runtime_execution_preflight=source_preflight,
    )

    assert source_preflight == original
    assert report["source_runtime_execution_preflight"] == original
    assert report["source_runtime_execution_preflight"] is not source_preflight
    source_preflight["summary"]["total_steps"] = 99
    assert report["source_runtime_counts"]["total_steps"] == 2


def test_deployment_runtime_report_combines_storage_preflight() -> None:
    bundle = build_physics_ingest_deployment_storage_bundle(
        {
            "physics_ingest_snapshots": [{"snapshot_id": "snap-1"}],
            "artifact_symbolic_expressions": [{"expression_id": "expr-1"}],
        },
        review_queue_rows={
            "physics_review_queue_tasks": [{"task_id": "task-1"}],
        },
        table_modes={
            "artifact_symbolic_expressions": "upsert",
            "physics_review_queue_tasks": "upsert",
        },
    )

    report = build_physics_ingest_deployment_runtime_report_dict(
        source_retrieval_run_plan=build_physics_source_retrieval_run_plan(max_jobs=1),
        storage_bundle=bundle,
    )

    assert report["ok"] is True
    assert report["storage_preflight"]["table_count"] == 3
    assert report["storage_preflight"]["total_row_count"] == 3
    assert report["storage_table_counts"] == {
        "artifact_symbolic_expressions": 1,
        "physics_ingest_snapshots": 1,
        "physics_review_queue_tasks": 1,
    }
    assert report["summary"]["storage_table_count"] == 3
    assert report["summary"]["storage_total_row_count"] == 3
    assert report["storage_bundle_summary"]["total_row_count"] == 3
    assert report["dashboard_summary"]["storage"] == {
        "table_count": 3,
        "total_row_count": 3,
        "table_row_counts": {
            "artifact_symbolic_expressions": 1,
            "physics_ingest_snapshots": 1,
            "physics_review_queue_tasks": 1,
        },
    }


def test_deployment_runtime_report_summarizes_blocked_diagnostics() -> None:
    plan = build_physics_source_retrieval_run_plan(
        job_id=(
            "hitran_lines.backfill",
            "materials_project_documents.backfill",
        ),
        dry_run=False,
    )
    storage_preflight = {
        "table_count": 1,
        "total_row_count": 1,
        "tables": [
            {
                "table": "custom_table",
                "mode": "upsert",
                "row_count": 1,
                "conflict_keys": [],
                "missing_conflict_metadata": True,
            }
        ],
        "missing_conflict_metadata_for_upserts": ["custom_table"],
    }

    report = build_physics_ingest_deployment_runtime_report_dict(
        source_retrieval_run_plan=plan,
        storage_preflight=storage_preflight,
    )

    assert report["ok"] is False
    assert report["blocked"] is True
    assert report["source_runtime_counts"]["total_steps"] == 2
    assert report["source_runtime_counts"]["requires_auth_count"] == 2
    assert report["diagnostic_summary"]["blocking_by_code"] == {
        "missing_auth": 2,
        "missing_http_client": 2,
        "missing_snapshot_sink": 2,
        "missing_storage_conflict_metadata": 1,
    }
    assert report["diagnostic_summary"]["by_stage"] == {
        "source_runtime": 6,
        "storage_preflight": 1,
    }
    assert report["summary"]["blocking_diagnostic_count"] == 7
    assert report["dashboard_summary"]["blocked"] is True
    assert report["dashboard_summary"]["diagnostics"]["blocking_diagnostic_count"] == 7
    assert report["dashboard_summary"]["diagnostics"]["blocking_by_code"] == {
        "missing_auth": 2,
        "missing_http_client": 2,
        "missing_snapshot_sink": 2,
        "missing_storage_conflict_metadata": 1,
    }
    assert [diagnostic["blocking"] for diagnostic in report["diagnostics"]] == [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]


def test_deployment_runtime_report_is_json_serializable() -> None:
    bundle = build_physics_ingest_deployment_storage_bundle(
        {
            "physics_ingest_snapshots": [
                {
                    "snapshot_id": "snap-1",
                    "score": float("nan"),
                    "tags": {"physics", "runtime"},
                }
            ]
        }
    )

    report = build_physics_ingest_deployment_runtime_report(
        source_retrieval_run_plan=build_physics_source_retrieval_run_plan(max_jobs=1),
        storage_bundle=bundle,
    )
    report_dict = report.to_dict()

    encoded = json.dumps(report_dict, sort_keys=True, allow_nan=False)
    decoded = json.loads(encoded)

    assert decoded == report_dict
    assert report.ok is True
    assert report.blocked is False
    assert decoded["storage_preflight"]["total_row_count"] == 1
