from __future__ import annotations

import json

from sciona.physics_ingest.review import (
    ReviewTrustReport,
    build_review_queue_task_rows,
    materialize_review_queue_task_rows,
    summarize_review_queue_tasks,
)


def test_review_queue_materializes_needs_human_work_task() -> None:
    rows = build_review_queue_task_rows(
        _review(
            trust_status="needs_human",
            achieved_status="source_verified",
            blockers=(
                "expression review_status must be human_reviewed",
                "human review evidence must identify a reviewer",
            ),
        ),
        candidate=_candidate(),
        expression=_expression(),
    )

    assert len(rows.tasks) == 1
    task = rows.to_rows()[0]
    assert task["task_kind"] == "human_review_required"
    assert task["task_status"] == "open"
    assert task["severity"] == "medium"
    assert task["priority"] == "p2"
    assert task["trust_status"] == "needs_human"
    assert task["achieved_status"] == "source_verified"
    assert task["target_ids"] == {
        "candidate_id": "cand-1",
        "source_candidate_id": "fixture-force",
        "expression_id": "expr-1",
        "source_expression_id": "fixture-force-expr",
        "artifact_id": "artifact-1",
        "version_id": "version-1",
        "snapshot_id": "snapshot-1",
    }
    assert task["source_family"] == "mechanics"
    assert task["source_system"] == "manual"
    assert task["blocker_summaries"] == [
        {
            "reason": "human_review",
            "detail": "expression review_status must be human_reviewed",
        },
        {
            "reason": "human_review",
            "detail": "human review evidence must identify a reviewer",
        },
    ]
    assert task["suggested_reviewer_focus"] == [
        "complete human review evidence and validity-bound signoff"
    ]


def test_review_queue_materializes_blocked_resolution_task() -> None:
    rows = build_review_queue_task_rows(
        _review(
            trust_status="blocked",
            achieved_status="raw_imported",
            blockers=("candidate_status is blocked",),
        ),
        candidate=_candidate(candidate_status="blocked"),
        expression=_expression(review_status="blocked"),
    )

    task = rows.to_rows()[0]
    assert task["task_kind"] == "blocked_resolution"
    assert task["task_status"] == "blocked"
    assert task["severity"] == "critical"
    assert task["priority"] == "p0"
    assert task["blocker_summaries"] == [
        {"reason": "blocked_status", "detail": "candidate_status is blocked"}
    ]
    assert task["suggested_reviewer_focus"] == [
        "resolve blocked status or upstream failure"
    ]


def test_review_queue_materializes_human_reviewed_audit_complete_not_work() -> None:
    report = ReviewTrustReport(
        achieved_status="published",
        trust_status="human_reviewed",
        publishable=True,
        blocked=False,
        needs_human=False,
        human_reviewed=True,
        blockers=(),
        gates=(),
    )

    rows = build_review_queue_task_rows(
        report,
        candidate=_candidate(candidate_status="human_reviewed"),
        expression=_expression(review_status="human_reviewed"),
    )

    assert [task.task_kind for task in rows.tasks] == ["audit_complete"]
    task = rows.to_rows()[0]
    assert task["task_status"] == "complete"
    assert task["severity"] == "low"
    assert task["priority"] == "p3"
    assert task["publishable"] is True
    assert task["blocker_summaries"] == []
    assert task["suggested_reviewer_focus"] == [
        "audit/completion metadata only; no human work required"
    ]

    active_only = build_review_queue_task_rows(
        report,
        candidate=_candidate(candidate_status="human_reviewed"),
        expression=_expression(review_status="human_reviewed"),
        include_audit_complete=False,
    )
    assert active_only.to_rows() == []


def test_review_queue_ids_are_deterministic_and_replay_sensitive() -> None:
    review = _review(
        trust_status="needs_human",
        achieved_status="parsed",
        blockers=("dimensional_hash is missing",),
    )
    first = build_review_queue_task_rows(
        review,
        candidate=_candidate(),
        expression=_expression(),
    ).to_rows()[0]
    second = build_review_queue_task_rows(
        dict(review),
        candidate=dict(_candidate()),
        expression=dict(_expression()),
    ).to_rows()[0]
    changed = build_review_queue_task_rows(
        _review(
            trust_status="needs_human",
            achieved_status="parsed",
            blockers=("variables missing dim_signature: m",),
        ),
        candidate=_candidate(),
        expression=_expression(),
    ).to_rows()[0]

    assert first["task_id"] == second["task_id"]
    assert first["replay_key"] == second["replay_key"]
    assert first["task_id"] != changed["task_id"]
    assert first["replay_key"] != changed["replay_key"]


def test_review_queue_summary_and_rows_are_json_safe() -> None:
    materialized = materialize_review_queue_task_rows(
        [
            _review(
                trust_status="needs_human",
                achieved_status="source_verified",
                blockers=("human review evidence must include a review timestamp",),
            ),
            _review(
                trust_status="blocked",
                achieved_status="raw_imported",
                blockers=("candidate_status is blocked",),
            ),
            _review(
                trust_status="human_reviewed",
                achieved_status="published",
                publishable=True,
                blockers=(),
            ),
        ],
        candidates=[
            _candidate(source_family="mechanics"),
            _candidate(source_family="thermo"),
            _candidate(source_family="mechanics"),
        ],
        expressions=[
            _expression(source_family="mechanics"),
            _expression(source_family="thermo"),
            _expression(source_family="mechanics"),
        ],
    )
    rows = materialized.to_dict()
    summary = summarize_review_queue_tasks(materialized.tasks)

    assert json.loads(json.dumps(rows)) == rows
    assert json.loads(json.dumps(summary)) == summary
    assert summary == {
        "task_count": 3,
        "task_kind_counts": {
            "human_review_required": 1,
            "blocked_resolution": 1,
            "audit_complete": 1,
        },
        "task_status_counts": {
            "open": 1,
            "blocked": 1,
            "complete": 1,
        },
        "trust_status_counts": {
            "needs_human": 1,
            "human_reviewed": 1,
            "blocked": 1,
        },
        "severity_counts": {
            "critical": 1,
            "medium": 1,
            "low": 1,
        },
        "priority_counts": {
            "p0": 1,
            "p2": 1,
            "p3": 1,
        },
        "source_family_counts": {"mechanics": 2, "thermo": 1},
        "blocker_reason_counts": {
            "blocked_status": 1,
            "human_review": 1,
        },
    }


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
        "gates": [],
    }


def _candidate(
    *,
    candidate_status: str = "source_verified",
    source_family: str = "mechanics",
) -> dict[str, object]:
    return {
        "candidate_id": "cand-1",
        "source_candidate_id": "fixture-force",
        "candidate_status": candidate_status,
        "source_system": "manual",
        "source_family": source_family,
        "snapshot_id": "snapshot-1",
    }


def _expression(
    *,
    review_status: str = "needs_human",
    source_family: str = "mechanics",
) -> dict[str, object]:
    return {
        "expression_id": "expr-1",
        "candidate_id": "cand-1",
        "source_expression_id": "fixture-force-expr",
        "artifact_id": "artifact-1",
        "version_id": "version-1",
        "review_status": review_status,
        "source_payload": {
            "source_family": source_family,
            "source_system": "manual",
        },
    }
