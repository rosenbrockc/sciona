from __future__ import annotations

import json
from pathlib import Path

from sciona.physics_ingest import (
    REVIEW_DEPLOYMENT_REPORT_KIND,
    REVIEW_DEPLOYMENT_REPORT_VERSION,
    REVIEW_QUEUE_TASKS_TABLE,
    PhysicsIngestReviewDeploymentReport,
    build_physics_ingest_review_deployment_report,
)


def test_review_deployment_report_packages_workflow_and_storage_rows() -> None:
    report = build_physics_ingest_review_deployment_report(
        [
            _review(
                trust_status="needs_human",
                achieved_status="source_verified",
                blockers=("expression review_status must be human_reviewed",),
            ),
            _review(
                trust_status="blocked",
                achieved_status="raw_imported",
                blockers=("candidate_status is blocked",),
            ),
        ],
        candidates=[
            _candidate(candidate_id="cand-1", source_family="mechanics"),
            _candidate(
                candidate_id="cand-2",
                candidate_status="blocked",
                source_family="thermo",
            ),
        ],
        expressions=[
            _expression(expression_id="expr-1", candidate_id="cand-1"),
            _expression(
                expression_id="expr-2",
                candidate_id="cand-2",
                review_status="blocked",
                source_family="thermo",
            ),
        ],
    )

    rows = report.to_dict()

    assert isinstance(report, PhysicsIngestReviewDeploymentReport)
    assert rows["report_version"] == REVIEW_DEPLOYMENT_REPORT_VERSION
    assert rows["report_kind"] == REVIEW_DEPLOYMENT_REPORT_KIND
    assert rows["side_effect_free"] is True
    assert rows["preflight"] is True
    assert rows["summary"]["ok"] is True
    assert rows["workflow"]["review_summary"]["assessment_count"] == 2
    assert rows["workflow"]["review_queue_summary"]["task_kind_counts"] == {
        "human_review_required": 1,
        "blocked_resolution": 1,
    }
    assert rows["summary"]["review_queue_row_count"] == 2
    assert rows["summary"]["candidate_status_patch_count"] == 2
    assert rows["summary"]["expression_status_patch_count"] == 2
    assert rows["dashboard_summary"]["report_version"] == (
        "physics-ingest-reviewer-workflow-dashboard.v1"
    )
    assert rows["dashboard_summary"]["review"] == {
        "review_count": 2,
        "trust_status_counts": {"blocked": 1, "needs_human": 1},
        "achieved_status_counts": {"raw_imported": 1, "source_verified": 1},
        "publishable_counts": {
            "not_publishable": 2,
            "publishable": 0,
            "unknown": 0,
        },
    }
    assert rows["dashboard_summary"]["queue"] == {
        "task_count": 2,
        "row_count": 2,
        "task_kind_counts": {
            "human_review_required": 1,
            "blocked_resolution": 1,
        },
        "task_status_counts": {"blocked": 1, "open": 1},
        "trust_status_counts": {"blocked": 1, "needs_human": 1},
        "severity_counts": {"critical": 1, "medium": 1},
        "priority_counts": {"p0": 1, "p2": 1},
        "source_family_counts": {"mechanics": 1, "thermo": 1},
        "blocker_reason_counts": {"blocked_status": 1, "human_review": 1},
    }
    assert rows["dashboard_summary"]["publication_status"] == {
        "candidate_status_patch_count": 2,
        "expression_status_patch_count": 2,
        "diagnostic_count": 0,
    }
    assert rows["dashboard_summary"]["storage"] == {
        "table_count": 3,
        "total_row_count": 6,
        "bundle_total_row_count": 6,
        "table_row_counts": {
            "artifact_symbolic_expressions": 2,
            "physics_equation_candidates": 2,
            REVIEW_QUEUE_TASKS_TABLE: 2,
        },
        "missing_conflict_metadata_count": 0,
        "missing_conflict_metadata_for_upserts": [],
    }
    assert rows["dashboard_summary"]["diagnostics"] == {
        "publication_status_diagnostic_count": 0,
        "review_queue_diagnostic_count": 0,
        "error_diagnostic_count": 0,
    }
    assert rows["dashboard_summary"]["replay"]["review_queue_replay_key_count"] == 2
    assert rows["dashboard_summary"]["replay"]["replay_keys_digest"] == rows[
        "workflow"
    ]["replay_keys_digest"]
    assert rows["storage_preflight"]["missing_conflict_metadata_for_upserts"] == []
    assert rows["storage_preflight"]["table_count"] == 3
    assert rows["storage_bundle"]["summary"]["component_order"] == [
        "publication",
        "review_queue",
    ]
    assert rows["storage_bundle"]["summary"]["write_plan_table_modes"] == {
        "physics_equation_candidates": "upsert",
        "artifact_symbolic_expressions": "upsert",
        REVIEW_QUEUE_TASKS_TABLE: "upsert",
    }
    assert rows["publication_status_rows"]["physics_equation_candidates"] == [
        {"candidate_id": "cand-1", "candidate_status": "source_verified"},
        {"candidate_id": "cand-2", "candidate_status": "blocked"},
    ]
    assert rows["publication_status_rows"]["artifact_symbolic_expressions"] == [
        {"expression_id": "expr-1", "review_status": "automated_pass"},
        {"expression_id": "expr-2", "review_status": "blocked"},
    ]


def test_review_deployment_report_is_deterministic_and_json_safe() -> None:
    kwargs = {
        "reviews": [
            _review(
                trust_status="human_reviewed",
                achieved_status="published",
                publishable=True,
                blockers=(),
            )
        ],
        "candidates": [_candidate(candidate_id="cand-1")],
        "expressions": [_expression(expression_id="expr-1")],
    }

    first = build_physics_ingest_review_deployment_report(**kwargs).to_dict()
    second = build_physics_ingest_review_deployment_report(**kwargs).to_dict()

    assert json.loads(json.dumps(first, sort_keys=True, allow_nan=False)) == first
    assert first == second
    assert first["dashboard_summary"]["ok"] is True
    assert first["dashboard_summary"]["queue"]["task_kind_counts"] == {
        "audit_complete": 1
    }
    assert len(first["workflow"]["replay_keys_digest"]) == 64
    assert len(first["workflow"]["storage_rows_digest"]) == 64
    assert first["review_queue_rows"]["write_plan"]["batches"][0]["table"] == (
        REVIEW_QUEUE_TASKS_TABLE
    )
    assert first["review_queue_rows"]["write_plan"]["batches"][0]["row_count"] == 1
    assert first["review_queue_rows"]["write_plan"]["batches"][0][
        "conflict_keys"
    ] == ["task_id"]


def test_review_deployment_report_can_skip_completed_audit_and_status_patches() -> None:
    report = build_physics_ingest_review_deployment_report(
        [
            _review(
                trust_status="human_reviewed",
                achieved_status="published",
                publishable=True,
                blockers=(),
            )
        ],
        candidates=[_candidate(candidate_id="cand-1")],
        expressions=[_expression(expression_id="expr-1")],
        include_audit_complete=False,
        include_publication_status_rows=False,
    ).to_dict()

    assert report["summary"]["review_queue_row_count"] == 0
    assert report["summary"]["candidate_status_patch_count"] == 0
    assert report["summary"]["expression_status_patch_count"] == 0
    assert report["dashboard_summary"]["queue"]["task_count"] == 0
    assert report["dashboard_summary"]["publication_status"] == {
        "candidate_status_patch_count": 0,
        "expression_status_patch_count": 0,
        "diagnostic_count": 0,
    }
    assert report["storage_preflight"]["total_row_count"] == 0
    assert report["storage_bundle"]["insert_rows"] == {}


def test_review_deployment_module_does_not_import_supabase_package() -> None:
    source = Path("sciona/physics_ingest/review_deployment.py").read_text()

    assert "import supabase" not in source
    assert "from supabase" not in source


def _review(
    *,
    trust_status: str,
    achieved_status: str,
    blockers: tuple[str, ...],
    publishable: bool = False,
) -> dict[str, object]:
    return {
        "achieved_status": achieved_status,
        "trust_status": trust_status,
        "publishable": publishable,
        "blocked": trust_status == "blocked",
        "needs_human": trust_status == "needs_human",
        "human_reviewed": trust_status == "human_reviewed",
        "blockers": list(blockers),
        "gates": [
            {"status": "raw_imported", "passed": True, "blockers": []},
            {"status": "parsed", "passed": True, "blockers": []},
            {"status": "dimension_resolved", "passed": True, "blockers": []},
            {"status": "symbolically_validated", "passed": True, "blockers": []},
            {"status": "source_verified", "passed": True, "blockers": []},
        ],
    }


def _candidate(
    *,
    candidate_id: str,
    candidate_status: str = "source_verified",
    source_family: str = "mechanics",
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "source_candidate_id": f"source-{candidate_id}",
        "candidate_status": candidate_status,
        "source_system": "manual",
        "source_family": source_family,
        "snapshot_id": "snapshot-1",
    }


def _expression(
    *,
    expression_id: str,
    candidate_id: str = "cand-1",
    review_status: str = "needs_human",
    source_family: str = "mechanics",
) -> dict[str, object]:
    return {
        "expression_id": expression_id,
        "candidate_id": candidate_id,
        "source_expression_id": f"source-{expression_id}",
        "artifact_id": f"artifact-{expression_id}",
        "version_id": f"version-{expression_id}",
        "review_status": review_status,
        "source_payload": {
            "source_family": source_family,
            "source_system": "manual",
        },
    }
