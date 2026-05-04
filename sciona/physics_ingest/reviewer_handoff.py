"""Reviewer UX handoff view models for physics ingestion.

This module reshapes already-materialized review queue rows and review
deployment reports into deterministic JSON-safe task cards. It performs no
external IO and constructs no Supabase, HTTP, browser, or app clients.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from sciona.physics_ingest.review import (
    REVIEW_QUEUE_PRIORITIES,
    REVIEW_QUEUE_SEVERITIES,
    REVIEW_QUEUE_TASK_KINDS,
    REVIEW_QUEUE_TASK_STATUSES,
    REVIEW_QUEUE_TASKS_TABLE,
    REVIEW_STATUSES,
    ReviewQueueRows,
    ReviewQueueTask,
    summarize_review_queue_tasks,
)
from sciona.physics_ingest.review_deployment import PhysicsIngestReviewDeploymentReport


REVIEWER_HANDOFF_REPORT_VERSION = "physics-ingest-reviewer-handoff.v1"
REVIEWER_HANDOFF_REPORT_KIND = "physics_ingest_reviewer_handoff"

_QUEUE_GROUP_ORDER = ("blocked", "open", "complete")
_ACTION_LABELS = {
    "start_review": "Start review",
    "approve_review": "Approve",
    "block_review": "Block",
    "resolve_blocker": "Resolve blocker",
    "reopen_review": "Reopen",
}


@dataclass(frozen=True)
class ReviewerActionMetadata:
    """Inert action descriptor for app-owned reviewer workflows."""

    action_id: str
    action_kind: str
    label: str
    enabled: bool
    task_id: str
    replay_key: str
    replay_digest: str
    allowed_task_statuses: tuple[str, ...]
    target_status: str
    requires_reviewer_identity: bool
    requires_resolution_note: bool
    terminal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_kind": self.action_kind,
            "label": self.label,
            "enabled": self.enabled,
            "task_id": self.task_id,
            "replay_key": self.replay_key,
            "replay_digest": self.replay_digest,
            "allowed_task_statuses": list(self.allowed_task_statuses),
            "target_status": self.target_status,
            "requires_reviewer_identity": self.requires_reviewer_identity,
            "requires_resolution_note": self.requires_resolution_note,
            "terminal": self.terminal,
        }


@dataclass(frozen=True)
class ReviewerTaskCard:
    """JSON-safe card view model for one review queue task."""

    card_id: str
    task_id: str
    replay_key: str
    replay_digest: str
    digest: str
    title: str
    subtitle: str
    queue_group: str
    status_badge: Mapping[str, str]
    priority: str
    severity: str
    task_kind: str
    task_status: str
    trust_status: str
    achieved_status: str
    publishable: bool
    source_context: Mapping[str, str]
    target_ids: Mapping[str, str]
    blocker_summaries: tuple[Mapping[str, str], ...] = ()
    suggested_reviewer_focus: tuple[str, ...] = ()
    actions: tuple[ReviewerActionMetadata, ...] = ()
    sort_key: tuple[Any, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "task_id": self.task_id,
            "replay_key": self.replay_key,
            "replay_digest": self.replay_digest,
            "digest": self.digest,
            "title": self.title,
            "subtitle": self.subtitle,
            "queue_group": self.queue_group,
            "status_badge": dict(self.status_badge),
            "priority": self.priority,
            "severity": self.severity,
            "task_kind": self.task_kind,
            "task_status": self.task_status,
            "trust_status": self.trust_status,
            "achieved_status": self.achieved_status,
            "publishable": self.publishable,
            "source_context": dict(self.source_context),
            "target_ids": dict(self.target_ids),
            "blocker_summaries": [
                dict(summary) for summary in self.blocker_summaries
            ],
            "suggested_reviewer_focus": list(self.suggested_reviewer_focus),
            "actions": [action.to_dict() for action in self.actions],
            "sort_key": list(self.sort_key),
        }


@dataclass(frozen=True)
class ReviewerQueueView:
    """Top-level deterministic handoff payload for reviewer UX callers."""

    report_version: str
    report_kind: str
    side_effect_free: bool
    view_id: str
    source: Mapping[str, Any]
    summary: Mapping[str, Any]
    grouped_counts: Mapping[str, Any]
    queue_groups: tuple[Mapping[str, Any], ...]
    task_cards: tuple[ReviewerTaskCard, ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "report_version": self.report_version,
                "report_kind": self.report_kind,
                "side_effect_free": self.side_effect_free,
                "view_id": self.view_id,
                "source": dict(self.source),
                "summary": dict(self.summary),
                "grouped_counts": dict(self.grouped_counts),
                "queue_groups": [dict(group) for group in self.queue_groups],
                "task_cards": [card.to_dict() for card in self.task_cards],
                "action_metadata": {
                    "actions_by_task_id": {
                        card.task_id: [action.to_dict() for action in card.actions]
                        for card in self.task_cards
                    },
                    "action_count": sum(len(card.actions) for card in self.task_cards),
                },
            }
        )


def build_reviewer_handoff_from_review_queue_rows(
    tasks: ReviewQueueRows
    | ReviewQueueTask
    | Mapping[str, Any]
    | Iterable[ReviewQueueTask | Mapping[str, Any]],
    *,
    source_kind: str = "review_queue_rows",
) -> ReviewerQueueView:
    """Build deterministic reviewer task cards from queue rows."""

    task_rows = _task_rows(tasks)
    source = {
        "source_kind": source_kind,
        "task_count": len(task_rows),
        "task_rows_digest": _stable_json_sha256(task_rows),
    }
    return _build_view(task_rows, source=source)


def build_reviewer_handoff_from_review_deployment_report(
    report: PhysicsIngestReviewDeploymentReport | Mapping[str, Any],
) -> ReviewerQueueView:
    """Build reviewer task cards from a review deployment report."""

    report_dict = (
        report.to_dict()
        if isinstance(report, PhysicsIngestReviewDeploymentReport)
        else _json_safe(report)
    )
    task_rows = _queue_rows_from_deployment_report(report_dict)
    source = {
        "source_kind": "review_deployment_report",
        "source_report_digest": _stable_json_sha256(report_dict),
        "task_count": len(task_rows),
        "task_rows_digest": _stable_json_sha256(task_rows),
        "deployment_report_version": str(report_dict.get("report_version") or ""),
        "deployment_report_kind": str(report_dict.get("report_kind") or ""),
    }
    return _build_view(task_rows, source=source)


def _build_view(
    task_rows: Sequence[Mapping[str, Any]],
    *,
    source: Mapping[str, Any],
) -> ReviewerQueueView:
    cards = tuple(sorted((_task_card(row) for row in task_rows), key=_card_sort_key))
    grouped_counts = _grouped_counts(cards)
    queue_groups = _queue_groups(cards)
    view_identity = {
        "source": source,
        "card_digests": [card.digest for card in cards],
    }
    summary = {
        "card_count": len(cards),
        "active_card_count": sum(1 for card in cards if card.queue_group != "complete"),
        "completed_card_count": sum(
            1 for card in cards if card.queue_group == "complete"
        ),
        "action_count": sum(len(card.actions) for card in cards),
        "card_digest": _stable_json_sha256([card.to_dict() for card in cards]),
        "queue_summary": summarize_review_queue_tasks(task_rows),
    }
    view_id = f"physics-reviewer-handoff:{_stable_json_sha256(view_identity)[:24]}"
    return ReviewerQueueView(
        report_version=REVIEWER_HANDOFF_REPORT_VERSION,
        report_kind=REVIEWER_HANDOFF_REPORT_KIND,
        side_effect_free=True,
        view_id=view_id,
        source=source,
        summary=summary,
        grouped_counts=grouped_counts,
        queue_groups=queue_groups,
        task_cards=cards,
    )


def _task_rows(
    tasks: ReviewQueueRows
    | ReviewQueueTask
    | Mapping[str, Any]
    | Iterable[ReviewQueueTask | Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if isinstance(tasks, ReviewQueueRows):
        values: Iterable[Any] = tasks.tasks
    elif isinstance(tasks, ReviewQueueTask):
        values = (tasks,)
    elif isinstance(tasks, Mapping):
        if "insert_rows" in tasks:
            insert_rows = tasks.get("insert_rows")
            if not isinstance(insert_rows, Mapping):
                raise ValueError("insert_rows must be a mapping")
            values = insert_rows.get(REVIEW_QUEUE_TASKS_TABLE) or ()
        elif REVIEW_QUEUE_TASKS_TABLE in tasks:
            values = tasks.get(REVIEW_QUEUE_TASKS_TABLE) or ()
        elif "tasks" in tasks:
            values = tasks.get("tasks") or ()
        elif "task_id" in tasks:
            values = (tasks,)
        else:
            values = ()
    elif isinstance(tasks, (str, bytes)) or not isinstance(tasks, Iterable):
        raise ValueError("tasks must be review queue rows, task rows, or iterable rows")
    else:
        values = tasks
    return tuple(
        _json_safe(value.to_dict() if hasattr(value, "to_dict") else value)
        for value in values
        if isinstance(value, ReviewQueueTask) or isinstance(value, Mapping)
    )


def _queue_rows_from_deployment_report(report: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    review_queue_rows = report.get("review_queue_rows")
    if isinstance(review_queue_rows, Mapping):
        return _task_rows(review_queue_rows)
    workflow = report.get("workflow")
    if isinstance(workflow, Mapping):
        rows = workflow.get("review_queue_rows")
        if isinstance(rows, Mapping | Sequence) and not isinstance(rows, str | bytes):
            return _task_rows(rows)
    return ()


def _task_card(row: Mapping[str, Any]) -> ReviewerTaskCard:
    task_id = _text(row, "task_id")
    replay_key = _text(row, "replay_key")
    task_kind = _text(row, "task_kind")
    task_status = _text(row, "task_status")
    priority = _text(row, "priority")
    severity = _text(row, "severity")
    trust_status = _text(row, "trust_status")
    achieved_status = _text(row, "achieved_status")
    target_ids = _string_mapping(row.get("target_ids"))
    blocker_summaries = tuple(
        _string_mapping(summary) for summary in _sequence(row.get("blocker_summaries"))
    )
    replay_digest = _stable_json_sha256(replay_key)
    queue_group = _queue_group(task_status)
    source_context = {
        "source_family": _text(row, "source_family"),
        "source_system": _text(row, "source_system"),
    }
    card_identity = {
        "task_id": task_id,
        "replay_key": replay_key,
        "task_kind": task_kind,
        "task_status": task_status,
        "target_ids": target_ids,
    }
    card_digest = _stable_json_sha256(card_identity)
    actions = _actions_for_task(
        task_id=task_id,
        replay_key=replay_key,
        replay_digest=replay_digest,
        task_status=task_status,
    )
    return ReviewerTaskCard(
        card_id=f"physics-review-card:{card_digest[:24]}",
        task_id=task_id,
        replay_key=replay_key,
        replay_digest=replay_digest,
        digest=card_digest,
        title=_task_title(task_kind, target_ids),
        subtitle=_task_subtitle(row, target_ids, blocker_summaries),
        queue_group=queue_group,
        status_badge={
            "label": _label(task_status),
            "tone": _status_tone(task_status, severity),
        },
        priority=priority,
        severity=severity,
        task_kind=task_kind,
        task_status=task_status,
        trust_status=trust_status,
        achieved_status=achieved_status,
        publishable=_bool(row.get("publishable")),
        source_context=source_context,
        target_ids=target_ids,
        blocker_summaries=blocker_summaries,
        suggested_reviewer_focus=tuple(
            str(value) for value in _sequence(row.get("suggested_reviewer_focus"))
        ),
        actions=actions,
        sort_key=_card_sort_key_values(
            queue_group=queue_group,
            priority=priority,
            severity=severity,
            task_kind=task_kind,
            task_id=task_id,
        ),
    )


def _actions_for_task(
    *,
    task_id: str,
    replay_key: str,
    replay_digest: str,
    task_status: str,
) -> tuple[ReviewerActionMetadata, ...]:
    if task_status == "blocked":
        kinds = ("resolve_blocker", "reopen_review")
    elif task_status == "complete":
        kinds = ("reopen_review",)
    else:
        kinds = ("start_review", "approve_review", "block_review")
    return tuple(
        _action(
            kind=kind,
            task_id=task_id,
            replay_key=replay_key,
            replay_digest=replay_digest,
            task_status=task_status,
        )
        for kind in kinds
    )


def _action(
    *,
    kind: str,
    task_id: str,
    replay_key: str,
    replay_digest: str,
    task_status: str,
) -> ReviewerActionMetadata:
    target_status = {
        "start_review": "open",
        "approve_review": "complete",
        "block_review": "blocked",
        "resolve_blocker": "open",
        "reopen_review": "open",
    }[kind]
    allowed_statuses = {
        "start_review": ("open",),
        "approve_review": ("open",),
        "block_review": ("open",),
        "resolve_blocker": ("blocked",),
        "reopen_review": ("complete", "blocked"),
    }[kind]
    action_identity = {
        "kind": kind,
        "task_id": task_id,
        "replay_key": replay_key,
        "target_status": target_status,
    }
    digest = _stable_json_sha256(action_identity)
    return ReviewerActionMetadata(
        action_id=f"physics-review-action:{digest[:24]}",
        action_kind=kind,
        label=_ACTION_LABELS[kind],
        enabled=task_status in allowed_statuses,
        task_id=task_id,
        replay_key=replay_key,
        replay_digest=replay_digest,
        allowed_task_statuses=allowed_statuses,
        target_status=target_status,
        requires_reviewer_identity=kind in {"start_review", "approve_review"},
        requires_resolution_note=kind
        in {"approve_review", "block_review", "resolve_blocker"},
        terminal=kind == "approve_review",
    )


def _grouped_counts(cards: Sequence[ReviewerTaskCard]) -> dict[str, Any]:
    return {
        "queue_group_counts": _ordered_counts(
            (card.queue_group for card in cards),
            _QUEUE_GROUP_ORDER,
        ),
        "task_kind_counts": _ordered_counts(
            (card.task_kind for card in cards),
            REVIEW_QUEUE_TASK_KINDS,
        ),
        "task_status_counts": _ordered_counts(
            (card.task_status for card in cards),
            REVIEW_QUEUE_TASK_STATUSES,
        ),
        "trust_status_counts": _ordered_counts(
            (card.trust_status for card in cards),
            REVIEW_STATUSES,
        ),
        "severity_counts": _ordered_counts(
            (card.severity for card in cards),
            REVIEW_QUEUE_SEVERITIES,
        ),
        "priority_counts": _ordered_counts(
            (card.priority for card in cards),
            REVIEW_QUEUE_PRIORITIES,
        ),
        "source_family_counts": _ordered_counts(
            (card.source_context.get("source_family") or "" for card in cards)
        ),
        "blocker_reason_counts": _ordered_counts(
            summary.get("reason") or ""
            for card in cards
            for summary in card.blocker_summaries
        ),
    }


def _queue_groups(cards: Sequence[ReviewerTaskCard]) -> tuple[Mapping[str, Any], ...]:
    by_group = {group: [] for group in _QUEUE_GROUP_ORDER}
    for card in cards:
        by_group.setdefault(card.queue_group, []).append(card.card_id)
    return tuple(
        {
            "group_id": group,
            "label": _label(group),
            "count": len(card_ids),
            "card_ids": card_ids,
        }
        for group, card_ids in by_group.items()
        if card_ids
    )


def _queue_group(task_status: str) -> str:
    if task_status == "blocked":
        return "blocked"
    if task_status == "complete":
        return "complete"
    return "open"


def _card_sort_key(card: ReviewerTaskCard) -> tuple[Any, ...]:
    return card.sort_key


def _card_sort_key_values(
    *,
    queue_group: str,
    priority: str,
    severity: str,
    task_kind: str,
    task_id: str,
) -> tuple[Any, ...]:
    return (
        _rank_or_after(queue_group, _QUEUE_GROUP_ORDER),
        _rank_or_after(priority, REVIEW_QUEUE_PRIORITIES),
        _rank_or_after(severity, REVIEW_QUEUE_SEVERITIES),
        _rank_or_after(task_kind, REVIEW_QUEUE_TASK_KINDS),
        task_id,
    )


def _task_title(task_kind: str, target_ids: Mapping[str, str]) -> str:
    target = (
        target_ids.get("source_candidate_id")
        or target_ids.get("candidate_id")
        or target_ids.get("expression_id")
        or "unknown target"
    )
    return f"{_label(task_kind)}: {target}"


def _task_subtitle(
    row: Mapping[str, Any],
    target_ids: Mapping[str, str],
    blocker_summaries: Sequence[Mapping[str, str]],
) -> str:
    source_family = _text(row, "source_family")
    achieved_status = _text(row, "achieved_status")
    blocker = blocker_summaries[0].get("detail") if blocker_summaries else ""
    parts = [
        value
        for value in (
            source_family,
            achieved_status,
            target_ids.get("expression_id") or "",
            blocker,
        )
        if value
    ]
    return " | ".join(parts)


def _status_tone(task_status: str, severity: str) -> str:
    if task_status == "blocked" or severity in {"critical", "high"}:
        return "danger"
    if task_status == "complete":
        return "success"
    return "attention"


def _ordered_counts(
    values: Iterable[str],
    preferred_order: Sequence[str] = (),
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value:
            counts[str(value)] = counts.get(str(value), 0) + 1
    ordered: dict[str, int] = {}
    for value in preferred_order:
        if value in counts:
            ordered[value] = counts.pop(value)
    for value in sorted(counts):
        ordered[value] = counts[value]
    return ordered


def _rank_or_after(value: str, preferred_order: Sequence[str]) -> tuple[int, str]:
    try:
        return preferred_order.index(value), value
    except ValueError:
        return len(preferred_order), value


def _label(value: str) -> str:
    return value.replace("_", " ").strip().title() if value else ""


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value).strip()


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}
    return bool(value)


def _string_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): str(item)
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        if item is not None and str(item) != ""
    }


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None or isinstance(value, str | bytes):
        return ()
    if isinstance(value, Sequence):
        return tuple(value)
    return ()


def _stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    return json.loads(
        json.dumps(
            _sanitize(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_sanitize(item) for item in value]
    if isinstance(value, set | frozenset):
        return [_sanitize(item) for item in sorted(value, key=repr)]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, str | int | bool):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _sanitize(to_dict())
    return str(value)


__all__ = [
    "REVIEWER_HANDOFF_REPORT_KIND",
    "REVIEWER_HANDOFF_REPORT_VERSION",
    "ReviewerActionMetadata",
    "ReviewerQueueView",
    "ReviewerTaskCard",
    "build_reviewer_handoff_from_review_deployment_report",
    "build_reviewer_handoff_from_review_queue_rows",
]
