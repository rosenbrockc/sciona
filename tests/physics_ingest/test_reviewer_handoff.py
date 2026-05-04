from __future__ import annotations

import json
from pathlib import Path

from sciona.physics_ingest import (
    REVIEWER_HANDOFF_REPORT_KIND,
    REVIEWER_HANDOFF_REPORT_VERSION,
    REVIEW_QUEUE_TASKS_TABLE,
    ReviewerQueueView,
    build_physics_ingest_review_deployment_report,
    build_reviewer_handoff_from_review_deployment_report,
    build_reviewer_handoff_from_review_queue_rows,
    materialize_review_queue_task_rows,
)


def test_reviewer_handoff_builds_task_cards_from_queue_rows() -> None:
    queue_rows = materialize_review_queue_task_rows(
        [
            _review(
                trust_status="needs_human",
                achieved_status="source_verified",
                blockers=("human review evidence must identify a reviewer",),
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
            _candidate(candidate_id="cand-1", source_family="mechanics"),
            _candidate(candidate_id="cand-2", source_family="thermo"),
            _candidate(candidate_id="cand-3", source_family="mechanics"),
        ],
        expressions=[
            _expression(expression_id="expr-1", candidate_id="cand-1"),
            _expression(
                expression_id="expr-2",
                candidate_id="cand-2",
                source_family="thermo",
            ),
            _expression(expression_id="expr-3", candidate_id="cand-3"),
        ],
    )

    handoff = build_reviewer_handoff_from_review_queue_rows(queue_rows)
    rows = handoff.to_dict()

    assert isinstance(handoff, ReviewerQueueView)
    assert rows["report_version"] == REVIEWER_HANDOFF_REPORT_VERSION
    assert rows["report_kind"] == REVIEWER_HANDOFF_REPORT_KIND
    assert rows["side_effect_free"] is True
    assert rows["source"]["source_kind"] == "review_queue_rows"
    assert rows["summary"]["card_count"] == 3
    assert rows["summary"]["active_card_count"] == 2
    assert rows["summary"]["completed_card_count"] == 1
    assert rows["dashboard_summary"] == {
        "report_version": "physics-ingest-reviewer-handoff-dashboard.v1",
        "side_effect_free": True,
        "source": {
            "source_kind": "review_queue_rows",
            "task_count": 3,
            "deployment_report_kind": "",
        },
        "queue": {
            "card_count": 3,
            "active_card_count": 2,
            "completed_card_count": 1,
            "blocked_card_count": 1,
            "open_card_count": 1,
            "complete_card_count": 1,
        },
        "counts": {
            "queue_groups": {"blocked": 1, "open": 1, "complete": 1},
            "task_kinds": {
                "human_review_required": 1,
                "blocked_resolution": 1,
                "audit_complete": 1,
            },
            "task_statuses": {"blocked": 1, "open": 1, "complete": 1},
            "trust_statuses": {
                "needs_human": 1,
                "blocked": 1,
                "human_reviewed": 1,
            },
            "severities": {"critical": 1, "medium": 1, "low": 1},
            "priorities": {"p0": 1, "p2": 1, "p3": 1},
            "source_families": {"mechanics": 2, "thermo": 1},
            "blocker_reasons": {
                "blocked_status": 1,
                "human_review": 1,
            },
        },
        "actions": {
            "action_count": 6,
            "enabled_action_count": 6,
            "terminal_action_count": 1,
            "by_kind": {
                "approve_review": 1,
                "block_review": 1,
                "reopen_review": 2,
                "resolve_blocker": 1,
                "start_review": 1,
            },
            "enabled_by_kind": {
                "approve_review": 1,
                "block_review": 1,
                "reopen_review": 2,
                "resolve_blocker": 1,
                "start_review": 1,
            },
        },
    }
    assert rows["grouped_counts"]["queue_group_counts"] == {
        "blocked": 1,
        "open": 1,
        "complete": 1,
    }
    assert rows["grouped_counts"]["source_family_counts"] == {
        "mechanics": 2,
        "thermo": 1,
    }
    assert [card["queue_group"] for card in rows["task_cards"]] == [
        "blocked",
        "open",
        "complete",
    ]

    blocked, open_card, complete = rows["task_cards"]
    assert blocked["status_badge"] == {"label": "Blocked", "tone": "danger"}
    assert [action["action_kind"] for action in blocked["actions"]] == [
        "resolve_blocker",
        "reopen_review",
    ]
    assert open_card["status_badge"] == {"label": "Open", "tone": "attention"}
    assert [action["action_kind"] for action in open_card["actions"]] == [
        "start_review",
        "approve_review",
        "block_review",
    ]
    assert complete["status_badge"] == {"label": "Complete", "tone": "success"}
    assert complete["actions"][0]["action_kind"] == "reopen_review"
    assert rows["action_metadata"]["action_count"] == 6


def test_reviewer_handoff_from_deployment_report_is_deterministic_json_safe() -> None:
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
            _candidate(candidate_id="cand-1"),
            _candidate(candidate_id="cand-2", source_family="thermo"),
        ],
        expressions=[
            _expression(expression_id="expr-1", candidate_id="cand-1"),
            _expression(expression_id="expr-2", candidate_id="cand-2"),
        ],
    )

    first = build_reviewer_handoff_from_review_deployment_report(report).to_dict()
    second = build_reviewer_handoff_from_review_deployment_report(
        report.to_dict()
    ).to_dict()

    assert json.loads(json.dumps(first, sort_keys=True, allow_nan=False)) == first
    assert first == second
    assert first["source"]["source_kind"] == "review_deployment_report"
    assert first["source"]["deployment_report_kind"] == (
        "physics_ingest_reviewer_workflow"
    )
    assert first["dashboard_summary"]["source"] == {
        "source_kind": "review_deployment_report",
        "task_count": 2,
        "deployment_report_kind": "physics_ingest_reviewer_workflow",
    }
    assert first["dashboard_summary"]["queue"]["blocked_card_count"] == 1
    assert first["dashboard_summary"]["queue"]["open_card_count"] == 1
    assert len(first["source"]["source_report_digest"]) == 64
    assert len(first["summary"]["card_digest"]) == 64
    assert first["summary"]["queue_summary"]["task_kind_counts"] == {
        "human_review_required": 1,
        "blocked_resolution": 1,
    }
    assert first["queue_groups"][0]["group_id"] == "blocked"
    assert first["queue_groups"][1]["group_id"] == "open"
    assert first["task_cards"][0]["replay_digest"]
    assert first["task_cards"][0]["actions"][0]["replay_key"] == (
        first["task_cards"][0]["replay_key"]
    )


def test_reviewer_handoff_accepts_write_plan_row_mapping() -> None:
    task = _review_queue_task()
    handoff = build_reviewer_handoff_from_review_queue_rows(
        {"insert_rows": {REVIEW_QUEUE_TASKS_TABLE: [task]}}
    ).to_dict()

    assert handoff["summary"]["card_count"] == 1
    assert handoff["task_cards"][0]["task_id"] == "task-1"
    assert handoff["task_cards"][0]["title"] == (
        "Human Review Required: source-cand-1"
    )


def test_reviewer_handoff_module_does_not_import_app_or_network_clients() -> None:
    source = Path("sciona/physics_ingest/reviewer_handoff.py").read_text()

    forbidden = (
        "import supabase",
        "from supabase",
        "import httpx",
        "import requests",
        "import selenium",
        "import playwright",
    )
    assert not any(pattern in source for pattern in forbidden)


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
    source_family: str = "mechanics",
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "source_candidate_id": f"source-{candidate_id}",
        "candidate_status": "source_verified",
        "source_system": "manual",
        "source_family": source_family,
        "snapshot_id": "snapshot-1",
    }


def _expression(
    *,
    expression_id: str,
    candidate_id: str,
    source_family: str = "mechanics",
) -> dict[str, object]:
    return {
        "expression_id": expression_id,
        "candidate_id": candidate_id,
        "source_expression_id": f"source-{expression_id}",
        "artifact_id": f"artifact-{expression_id}",
        "version_id": f"version-{expression_id}",
        "review_status": "needs_human",
        "source_payload": {
            "source_family": source_family,
            "source_system": "manual",
        },
    }


def _review_queue_task() -> dict[str, object]:
    return {
        "task_id": "task-1",
        "replay_key": "physics-review:abc",
        "task_kind": "human_review_required",
        "task_status": "open",
        "priority": "p2",
        "severity": "medium",
        "target_ids": {
            "candidate_id": "cand-1",
            "source_candidate_id": "source-cand-1",
        },
        "trust_status": "needs_human",
        "achieved_status": "source_verified",
        "publishable": False,
        "blocker_summaries": [
            {
                "reason": "human_review",
                "detail": "expression review_status must be human_reviewed",
            }
        ],
        "suggested_reviewer_focus": [
            "complete human review evidence and validity-bound signoff"
        ],
        "source_family": "mechanics",
        "source_system": "manual",
    }
