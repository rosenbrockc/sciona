"""Phase 5 audit and review contracts for physics ingestion.

The review layer consumes already-staged Wave 0 rows or plain dictionaries and
computes publishability gates without reaching into Supabase. It is deliberately
side-effect free so loaders, CLIs, and CI checks can share the same contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sciona.physics_ingest.write_plan import PublicationWritePlan, WriteMode


REVIEW_STATUSES: tuple[str, ...] = (
    "unreviewed",
    "automated_pass",
    "needs_human",
    "human_reviewed",
    "blocked",
)

WORKFLOW_STATUSES: tuple[str, ...] = (
    "raw_imported",
    "parsed",
    "dimension_resolved",
    "symbolically_validated",
    "source_verified",
    "human_reviewed",
    "published",
)

_STATUS_RANK = {status: index for index, status in enumerate(WORKFLOW_STATUSES)}
_PARSED_PARSE_STATUSES = {"parsed", "normalized"}
_PASS_VALUES = {"pass", "passed", "ok", "success", "succeeded", "true", True}
_REVIEWED_BOUND_STATUSES = {"automated_pass", "human_reviewed"}
_DEPENDENCY_KINDS = {"uses_constant", "uses_data_artifact"}
_BLOCKED_STATUSES = {"blocked", "failed", "parse_failed"}
_UNKNOWN_DIM_SIGNATURES = {"", "?", "unknown", "unresolved", "tbd"}
_UNKNOWN_DIMENSION_SOURCES = {"", "unknown", "unresolved", "tbd"}
REVIEW_QUEUE_TASK_KINDS: tuple[str, ...] = (
    "human_review_required",
    "blocked_resolution",
    "audit_complete",
)
REVIEW_QUEUE_TASK_STATUSES: tuple[str, ...] = (
    "open",
    "blocked",
    "complete",
)
REVIEW_QUEUE_SEVERITIES: tuple[str, ...] = ("critical", "high", "medium", "low")
REVIEW_QUEUE_PRIORITIES: tuple[str, ...] = ("p0", "p1", "p2", "p3")
REVIEW_QUEUE_TASKS_TABLE = "physics_review_queue_tasks"


@dataclass(frozen=True)
class ReviewGateResult:
    """Result for one publishability gate."""

    status: str
    passed: bool
    blockers: tuple[str, ...] = ()
    evidence: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ReviewAssessment:
    """Deterministic Phase 5 review result."""

    achieved_status: str
    publishable: bool
    gates: tuple[ReviewGateResult, ...]
    trust_status: str = "needs_human"

    @property
    def blockers(self) -> tuple[str, ...]:
        """All blockers from failed concrete gates, preserving gate order."""

        blockers: list[str] = []
        for gate in self.gates:
            if gate.status == "published":
                continue
            blockers.extend(gate.blockers)
        return tuple(blockers)

    def gate(self, status: str) -> ReviewGateResult:
        """Return the gate result for ``status``."""

        for gate in self.gates:
            if gate.status == status:
                return gate
        raise KeyError(status)

    @property
    def blocked(self) -> bool:
        return self.trust_status == "blocked"

    @property
    def needs_human(self) -> bool:
        return self.trust_status == "needs_human"

    @property
    def human_reviewed(self) -> bool:
        return self.trust_status == "human_reviewed"

    def to_report(self) -> "ReviewTrustReport":
        """Return a JSON-friendly, side-effect-free review report."""

        return ReviewTrustReport.from_assessment(self)


@dataclass(frozen=True)
class ReviewTrustReport:
    """Compact trust report for CLI and pipeline callers."""

    achieved_status: str
    trust_status: str
    publishable: bool
    blocked: bool
    needs_human: bool
    human_reviewed: bool
    blockers: tuple[str, ...]
    gates: tuple[Mapping[str, Any], ...]

    @classmethod
    def from_assessment(cls, assessment: ReviewAssessment) -> "ReviewTrustReport":
        return cls(
            achieved_status=assessment.achieved_status,
            trust_status=assessment.trust_status,
            publishable=assessment.publishable,
            blocked=assessment.blocked,
            needs_human=assessment.needs_human,
            human_reviewed=assessment.human_reviewed,
            blockers=assessment.blockers,
            gates=tuple(
                {
                    "status": gate.status,
                    "passed": gate.passed,
                    "blockers": list(gate.blockers),
                    "evidence": dict(gate.evidence or {}),
                }
                for gate in assessment.gates
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "achieved_status": self.achieved_status,
            "trust_status": self.trust_status,
            "publishable": self.publishable,
            "blocked": self.blocked,
            "needs_human": self.needs_human,
            "human_reviewed": self.human_reviewed,
            "blockers": list(self.blockers),
            "gates": [dict(gate) for gate in self.gates],
        }


@dataclass(frozen=True)
class ReviewPublicationDiagnostic:
    """One non-fatal status-row publication diagnostic."""

    table: str
    reason: str
    row_id: str = ""
    severity: str = "skipped"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "reason": self.reason,
            "row_id": self.row_id,
            "severity": self.severity,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ReviewPublicationRows:
    """Side-effect-free status row patches derived from a review decision."""

    artifact_symbolic_expressions: tuple[dict[str, Any], ...] = ()
    physics_equation_candidates: tuple[dict[str, Any], ...] = ()
    diagnostics: tuple[ReviewPublicationDiagnostic, ...] = ()

    def to_upsert_rows(self) -> dict[str, list[dict[str, Any]]]:
        rows: dict[str, list[dict[str, Any]]] = {}
        if self.physics_equation_candidates:
            rows["physics_equation_candidates"] = [
                dict(row) for row in self.physics_equation_candidates
            ]
        if self.artifact_symbolic_expressions:
            rows["artifact_symbolic_expressions"] = [
                dict(row) for row in self.artifact_symbolic_expressions
            ]
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_symbolic_expressions": [
                dict(row) for row in self.artifact_symbolic_expressions
            ],
            "physics_equation_candidates": [
                dict(row) for row in self.physics_equation_candidates
            ],
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class ReviewQueueTask:
    """JSON-safe task row for review queues and audit dashboards."""

    task_id: str
    replay_key: str
    task_kind: str
    task_status: str
    priority: str
    severity: str
    target_ids: Mapping[str, str]
    trust_status: str
    achieved_status: str
    publishable: bool
    blocker_summaries: tuple[Mapping[str, str], ...] = ()
    suggested_reviewer_focus: tuple[str, ...] = ()
    source_family: str = ""
    source_system: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "replay_key": self.replay_key,
            "task_kind": self.task_kind,
            "task_status": self.task_status,
            "priority": self.priority,
            "severity": self.severity,
            "target_ids": dict(self.target_ids),
            "trust_status": self.trust_status,
            "achieved_status": self.achieved_status,
            "publishable": self.publishable,
            "blocker_summaries": [
                dict(summary) for summary in self.blocker_summaries
            ],
            "suggested_reviewer_focus": list(self.suggested_reviewer_focus),
            "source_family": self.source_family,
            "source_system": self.source_system,
        }


@dataclass(frozen=True)
class ReviewQueueRows:
    """Side-effect-free review queue materialization."""

    tasks: tuple[ReviewQueueTask, ...] = ()

    def to_rows(self) -> list[dict[str, Any]]:
        return [task.to_dict() for task in self.tasks]

    def to_dict(self) -> dict[str, Any]:
        return {"tasks": self.to_rows()}


@dataclass(frozen=True)
class ReviewQueueWritePlanRows:
    """Review queue rows shaped for the inert publication writer boundary."""

    rows_by_table: Mapping[str, tuple[dict[str, Any], ...]]
    diagnostics: tuple[Mapping[str, Any], ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)
    write_plan: "PublicationWritePlan | None" = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rows_by_table",
            {
                str(table): tuple(dict(_json_safe(row)) for row in rows)
                for table, rows in self.rows_by_table.items()
            },
        )
        object.__setattr__(
            self,
            "diagnostics",
            tuple(_json_safe(diagnostic) for diagnostic in self.diagnostics),
        )
        object.__setattr__(self, "summary", _json_safe(self.summary))

    def to_insert_rows(self) -> dict[str, list[dict[str, Any]]]:
        return {
            table: [dict(row) for row in rows]
            for table, rows in self.rows_by_table.items()
            if rows
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "insert_rows": self.to_insert_rows(),
            "diagnostics": [dict(diagnostic) for diagnostic in self.diagnostics],
            "summary": dict(self.summary),
            "write_plan": None
            if self.write_plan is None
            else self.write_plan.to_dict(),
        }


def assess_publishability(
    *,
    candidate: Mapping[str, Any] | Any | None = None,
    expression: Mapping[str, Any] | Any | None = None,
    variables: Iterable[Mapping[str, Any] | Any] = (),
    references: Iterable[Mapping[str, Any] | Any] = (),
    validity_bounds: Iterable[Mapping[str, Any] | Any] = (),
    relationships: Iterable[Mapping[str, Any] | Any] = (),
    io_specs: Iterable[Mapping[str, Any] | Any] = (),
    min_parse_confidence: float = 0.8,
) -> ReviewAssessment:
    """Assess Phase 5 physics-ingest publishability from local row data.

    Inputs may be Pydantic rows from :mod:`sciona.physics_ingest.staging`, plain
    dictionaries, or objects with attributes. No database client is accepted or
    consulted.
    """

    candidate_row = _row(candidate)
    expression_row = _row(expression)
    variable_rows = tuple(_row(row) for row in variables)
    reference_rows = tuple(_row(row) for row in references)
    bound_rows = tuple(_row(row) for row in validity_bounds)
    relationship_rows = tuple(_row(row) for row in relationships)
    io_spec_rows = tuple(_row(row) for row in io_specs)

    gates = (
        _raw_imported_gate(candidate_row, expression_row),
        _parsed_gate(candidate_row, expression_row, min_parse_confidence),
        _dimension_resolved_gate(expression_row, variable_rows, io_spec_rows),
        _symbolically_validated_gate(expression_row, bound_rows),
        _source_verified_gate(expression_row, reference_rows, relationship_rows),
        _human_reviewed_gate(expression_row, bound_rows),
    )
    published_gate = _published_gate(gates)
    all_gates = (*gates, published_gate)

    achieved_status = "raw_imported"
    for gate in all_gates:
        if gate.passed:
            achieved_status = gate.status
        else:
            break

    return ReviewAssessment(
        achieved_status=achieved_status,
        publishable=published_gate.passed,
        gates=all_gates,
        trust_status=_trust_status(candidate_row, expression_row, bound_rows, all_gates),
    )


def build_review_trust_report(**kwargs: Any) -> dict[str, Any]:
    """Build a JSON-serializable Phase 5 review report without database access."""

    return assess_publishability(**kwargs).to_report().to_dict()


def summarize_review_assessments(
    reviews: Iterable[ReviewAssessment | ReviewTrustReport | Mapping[str, Any]],
) -> dict[str, Any]:
    """Build deterministic JSON-safe rollups for Phase 5 review dashboards."""

    rows = [_review_row(review) for review in reviews]
    gate_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        for gate in _review_gates(row):
            status = _text(gate, "status")
            if not status:
                continue
            counts = gate_counts.setdefault(status, {"passed": 0, "failed": 0})
            if _bool(gate.get("passed")):
                counts["passed"] += 1
            else:
                counts["failed"] += 1

    return {
        "assessment_count": len(rows),
        "achieved_status_counts": _ordered_counts(
            (_summary_text(row, "achieved_status") for row in rows),
            WORKFLOW_STATUSES,
        ),
        "trust_status_counts": _ordered_counts(
            (_summary_text(row, "trust_status") for row in rows),
            REVIEW_STATUSES,
        ),
        "publishable_counts": {
            "publishable": sum(1 for row in rows if _bool(row.get("publishable"))),
            "not_publishable": sum(
                1
                for row in rows
                if "publishable" in row and not _bool(row.get("publishable"))
            ),
            "unknown": sum(1 for row in rows if "publishable" not in row),
        },
        "blocker_counts": _ordered_counts(
            blocker
            for row in rows
            for blocker in _review_blockers(row)
        ),
        "gate_counts": _ordered_gate_counts(gate_counts),
    }


def build_review_publication_status_rows(
    review: ReviewAssessment | ReviewTrustReport | Mapping[str, Any],
    *,
    candidate: Mapping[str, Any] | Any | None = None,
    expression: Mapping[str, Any] | Any | None = None,
) -> ReviewPublicationRows:
    """Build deterministic publication status patches from a review decision.

    The returned rows are intentionally minimal and inert. They include only the
    table conflict key plus status columns that callers may choose to upsert:
    ``artifact_symbolic_expressions.review_status`` and
    ``physics_equation_candidates.candidate_status``. Rows are omitted when the
    target ID is unavailable or the review decision is not actionable.
    """

    review_row = _review_row(review)
    review_status = _publication_review_status(review, review_row)
    diagnostics: list[ReviewPublicationDiagnostic] = []
    expression_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    if review_status == "unreviewed":
        diagnostics.append(
            ReviewPublicationDiagnostic(
                table="review",
                reason="non_actionable_assessment",
                severity="info",
                detail="review decision did not progress beyond unreviewed",
            )
        )
        return ReviewPublicationRows(diagnostics=tuple(diagnostics))

    expression_row = _row(expression)
    expression_id = _text(expression_row, "expression_id")
    if expression_row:
        if expression_id:
            expression_rows.append(
                {
                    "expression_id": expression_id,
                    "review_status": review_status,
                }
            )
        else:
            diagnostics.append(
                ReviewPublicationDiagnostic(
                    table="artifact_symbolic_expressions",
                    reason="missing_expression_id",
                    detail="review_status patch requires expression_id",
                )
            )

    candidate_row = _row(candidate)
    candidate_id = _text(candidate_row, "candidate_id")
    if candidate_row:
        if candidate_id:
            candidate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_status": _candidate_status_for_review(
                        review_row, review_status
                    ),
                }
            )
        else:
            diagnostics.append(
                ReviewPublicationDiagnostic(
                    table="physics_equation_candidates",
                    reason="missing_candidate_id",
                    detail="candidate_status patch requires candidate_id",
                )
            )

    if not expression_row and not candidate_row:
        diagnostics.append(
            ReviewPublicationDiagnostic(
                table="review",
                reason="missing_target_rows",
                detail="provide candidate and/or expression rows to build status patches",
            )
        )

    return ReviewPublicationRows(
        artifact_symbolic_expressions=tuple(expression_rows),
        physics_equation_candidates=tuple(candidate_rows),
        diagnostics=tuple(diagnostics),
    )


def build_review_queue_task_rows(
    review: ReviewAssessment | ReviewTrustReport | Mapping[str, Any],
    *,
    candidate: Mapping[str, Any] | Any | None = None,
    expression: Mapping[str, Any] | Any | None = None,
    include_audit_complete: bool = True,
) -> ReviewQueueRows:
    """Build deterministic queue/audit rows from a review decision.

    The function is intentionally inert: it does not read or write any external
    system, and task IDs are stable hashes of target IDs, statuses, and blockers.
    Human-reviewed decisions produce completion metadata instead of an open human
    work task; callers that only want active work can disable that row with
    ``include_audit_complete=False``.
    """

    review_row = _review_row(review)
    candidate_row = _row(candidate)
    expression_row = _row(expression)
    trust_status = _queue_trust_status(review_row)
    achieved_status = _summary_text(review_row, "achieved_status")
    publishable = _bool(review_row.get("publishable"))
    blockers = _review_blockers(review_row)

    if trust_status == "blocked" or _bool(review_row.get("blocked")):
        task_kind = "blocked_resolution"
        task_status = "blocked"
    elif trust_status == "human_reviewed" or _bool(review_row.get("human_reviewed")):
        if not include_audit_complete:
            return ReviewQueueRows()
        task_kind = "audit_complete"
        task_status = "complete"
    elif trust_status in {"needs_human", "automated_pass"} or _bool(
        review_row.get("needs_human")
    ):
        task_kind = "human_review_required"
        task_status = "open"
    else:
        return ReviewQueueRows()

    blocker_summaries = tuple(_blocker_summary(blocker) for blocker in blockers)
    severity, priority = _queue_severity_priority(
        task_kind,
        achieved_status,
        blocker_summaries,
    )
    target_ids = _queue_target_ids(candidate_row, expression_row)
    source_family = _source_context_text(candidate_row, expression_row, "family")
    source_system = _source_context_text(candidate_row, expression_row, "system")
    stable_identity = {
        "task_kind": task_kind,
        "task_status": task_status,
        "target_ids": target_ids,
        "trust_status": trust_status,
        "achieved_status": achieved_status,
        "blockers": list(blockers),
    }
    digest = _stable_digest(stable_identity)
    task = ReviewQueueTask(
        task_id=f"physics-review-task:{digest[:24]}",
        replay_key=f"physics-review:{digest}",
        task_kind=task_kind,
        task_status=task_status,
        priority=priority,
        severity=severity,
        target_ids=target_ids,
        trust_status=trust_status,
        achieved_status=achieved_status,
        publishable=publishable,
        blocker_summaries=blocker_summaries,
        suggested_reviewer_focus=_suggested_reviewer_focus(
            task_kind,
            blocker_summaries,
        ),
        source_family=source_family,
        source_system=source_system,
    )
    return ReviewQueueRows(tasks=(task,))


def materialize_review_queue_task_rows(
    reviews: Iterable[ReviewAssessment | ReviewTrustReport | Mapping[str, Any]],
    *,
    candidates: Iterable[Mapping[str, Any] | Any | None] = (),
    expressions: Iterable[Mapping[str, Any] | Any | None] = (),
    include_audit_complete: bool = True,
) -> ReviewQueueRows:
    """Build queue rows for many review decisions, aligning context by position."""

    candidate_rows = tuple(candidates)
    expression_rows = tuple(expressions)
    tasks: list[ReviewQueueTask] = []
    for index, review in enumerate(reviews):
        rows = build_review_queue_task_rows(
            review,
            candidate=candidate_rows[index] if index < len(candidate_rows) else None,
            expression=expression_rows[index] if index < len(expression_rows) else None,
            include_audit_complete=include_audit_complete,
        )
        tasks.extend(rows.tasks)
    return ReviewQueueRows(tasks=tuple(tasks))


def summarize_review_queue_tasks(
    tasks: Iterable[ReviewQueueTask | Mapping[str, Any]],
) -> dict[str, Any]:
    """Build deterministic JSON-safe rollups for review queue dashboards."""

    rows = [_queue_task_row(task) for task in tasks]
    return {
        "task_count": len(rows),
        "task_kind_counts": _ordered_counts(
            (_summary_text(row, "task_kind") for row in rows),
            REVIEW_QUEUE_TASK_KINDS,
        ),
        "task_status_counts": _ordered_counts(
            (_summary_text(row, "task_status") for row in rows),
            REVIEW_QUEUE_TASK_STATUSES,
        ),
        "trust_status_counts": _ordered_counts(
            (_summary_text(row, "trust_status") for row in rows),
            REVIEW_STATUSES,
        ),
        "severity_counts": _ordered_counts(
            (_summary_text(row, "severity") for row in rows),
            REVIEW_QUEUE_SEVERITIES,
        ),
        "priority_counts": _ordered_counts(
            (_summary_text(row, "priority") for row in rows),
            REVIEW_QUEUE_PRIORITIES,
        ),
        "source_family_counts": _ordered_counts(
            (_summary_text(row, "source_family") for row in rows)
        ),
        "blocker_reason_counts": _ordered_counts(
            _summary_text(summary, "reason")
            for row in rows
            for summary in _queue_blocker_summaries(row)
        ),
    }


def build_review_queue_write_plan_rows(
    tasks: ReviewQueueRows
    | ReviewQueueTask
    | Mapping[str, Any]
    | Iterable[ReviewQueueTask | Mapping[str, Any]],
    *,
    include_write_plan: bool = False,
    table_name: str = REVIEW_QUEUE_TASKS_TABLE,
    table_modes: Mapping[str, "WriteMode"] | None = None,
) -> ReviewQueueWritePlanRows:
    """Adapt review queue task rows to the injected publication writer path.

    The helper performs no database IO. It accepts materialized
    ``ReviewQueueRows``, individual tasks, task mappings, or iterable task
    values, then returns deterministic JSON-safe table rows plus write-plan
    summary metadata.
    """

    if not isinstance(table_name, str) or not table_name.strip():
        raise ValueError("table_name must be a non-empty string")
    table_name = table_name.strip()

    normalized_rows, diagnostics = _normalize_review_queue_write_rows(
        tasks,
        table_name=table_name,
    )
    rows_by_table = {table_name: normalized_rows} if normalized_rows else {}

    from sciona.physics_ingest.write_plan import build_publication_write_plan

    write_plan = build_publication_write_plan(
        rows_by_table,
        table_modes={table_name: "upsert", **dict(table_modes or {})},
    )
    summary = _build_review_queue_write_plan_rows_summary(
        rows=normalized_rows,
        diagnostics=diagnostics,
        write_plan=write_plan,
        table_name=table_name,
    )
    return ReviewQueueWritePlanRows(
        rows_by_table=rows_by_table,
        diagnostics=diagnostics,
        summary=summary,
        write_plan=write_plan if include_write_plan else None,
    )


def require_publishable(**kwargs: Any) -> ReviewAssessment:
    """Return an assessment or raise ``ValueError`` with gate blockers."""

    assessment = assess_publishability(**kwargs)
    if not assessment.publishable:
        raise ValueError("; ".join(assessment.blockers))
    return assessment


def _review_row(
    review: ReviewAssessment | ReviewTrustReport | Mapping[str, Any],
) -> Mapping[str, Any]:
    if isinstance(review, ReviewAssessment):
        return review.to_report().to_dict()
    if isinstance(review, ReviewTrustReport):
        return review.to_dict()
    if isinstance(review, Mapping):
        return review
    raise TypeError(f"cannot treat {type(review)!r} as a review decision")


def _publication_review_status(
    review: ReviewAssessment | ReviewTrustReport | Mapping[str, Any],
    review_row: Mapping[str, Any],
) -> str:
    trust_status = _text(review_row, "trust_status")
    if trust_status in REVIEW_STATUSES:
        if trust_status == "needs_human" and _automated_gates_passed(review_row):
            return "automated_pass"
        return trust_status
    if _bool(review_row.get("blocked")):
        return "blocked"
    if _bool(review_row.get("human_reviewed")):
        return "human_reviewed"
    if _bool(review_row.get("needs_human")):
        return "automated_pass" if _automated_gates_passed(review_row) else "needs_human"
    if _bool(review_row.get("publishable")):
        return "human_reviewed"
    if isinstance(review, ReviewAssessment):
        return "needs_human"
    return "unreviewed"


def _automated_gates_passed(review_row: Mapping[str, Any]) -> bool:
    gates = review_row.get("gates")
    if not isinstance(gates, Sequence) or isinstance(gates, (str, bytes)):
        return False
    passed_by_status: dict[str, bool] = {}
    for gate in gates:
        if isinstance(gate, Mapping):
            status = _text(gate, "status")
            if status:
                passed_by_status[status] = _bool(gate.get("passed"))
    automated_statuses = WORKFLOW_STATUSES[:5]
    return all(passed_by_status.get(status) is True for status in automated_statuses)


def _candidate_status_for_review(
    review_row: Mapping[str, Any],
    review_status: str,
) -> str:
    if review_status == "blocked":
        return "blocked"
    if review_status == "human_reviewed":
        return "human_reviewed"
    achieved_status = _text(review_row, "achieved_status")
    if achieved_status in _STATUS_RANK and achieved_status != "published":
        return achieved_status
    if review_status == "automated_pass":
        return "source_verified"
    return "raw_imported"


def _queue_trust_status(review_row: Mapping[str, Any]) -> str:
    trust_status = _text(review_row, "trust_status")
    if trust_status:
        return trust_status
    if _bool(review_row.get("blocked")):
        return "blocked"
    if _bool(review_row.get("human_reviewed")) or _bool(review_row.get("publishable")):
        return "human_reviewed"
    if _bool(review_row.get("needs_human")):
        return "needs_human"
    return "unreviewed"


def _queue_task_row(task: ReviewQueueTask | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(task, ReviewQueueTask):
        return task.to_dict()
    return task if isinstance(task, Mapping) else {}


def _normalize_review_queue_write_rows(
    tasks: ReviewQueueRows
    | ReviewQueueTask
    | Mapping[str, Any]
    | Iterable[ReviewQueueTask | Mapping[str, Any]],
    *,
    table_name: str,
) -> tuple[tuple[dict[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    task_values = _review_queue_task_values(tasks)
    diagnostics: list[Mapping[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, task in enumerate(task_values):
        row = _queue_task_row(task)
        if not row:
            diagnostics.append(
                _review_queue_write_diagnostic(
                    table_name=table_name,
                    reason="empty_task_row",
                    row_index=index,
                    detail="task row must be a ReviewQueueTask or mapping",
                )
            )
            continue
        safe_row = dict(_json_safe(row))
        if not _text(safe_row, "task_id"):
            diagnostics.append(
                _review_queue_write_diagnostic(
                    table_name=table_name,
                    reason="missing_task_id",
                    row_index=index,
                    detail="review queue write rows require task_id conflict key",
                )
            )
            continue
        if not _text(safe_row, "replay_key"):
            diagnostics.append(
                _review_queue_write_diagnostic(
                    table_name=table_name,
                    reason="missing_replay_key",
                    row_index=index,
                    severity="warning",
                    detail="review queue task row is missing replay_key",
                )
            )
        rows.append(safe_row)
    return tuple(sorted(rows, key=_review_queue_write_row_sort_key)), tuple(diagnostics)


def _review_queue_task_values(
    tasks: ReviewQueueRows
    | ReviewQueueTask
    | Mapping[str, Any]
    | Iterable[ReviewQueueTask | Mapping[str, Any]],
) -> tuple[ReviewQueueTask | Mapping[str, Any], ...]:
    if isinstance(tasks, ReviewQueueRows):
        return tasks.tasks
    if isinstance(tasks, ReviewQueueTask):
        return (tasks,)
    if isinstance(tasks, Mapping):
        task_rows = tasks.get("tasks")
        if task_rows is None:
            return (tasks,)
        if (
            isinstance(task_rows, Mapping)
            or isinstance(task_rows, (str, bytes))
            or task_rows is None
        ):
            raise ValueError("tasks mapping value must be an iterable of task rows")
        return tuple(task_rows)
    if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Iterable):
        raise ValueError("tasks must be ReviewQueueRows, a task row, or iterable rows")
    return tuple(tasks)


def _review_queue_write_row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    priority = _summary_text(row, "priority")
    severity = _summary_text(row, "severity")
    task_status = _summary_text(row, "task_status")
    task_kind = _summary_text(row, "task_kind")
    return (
        _rank_or_after(priority, REVIEW_QUEUE_PRIORITIES),
        _rank_or_after(severity, REVIEW_QUEUE_SEVERITIES),
        _rank_or_after(task_status, REVIEW_QUEUE_TASK_STATUSES),
        _rank_or_after(task_kind, REVIEW_QUEUE_TASK_KINDS),
        _summary_text(row, "task_id"),
        _summary_text(row, "replay_key"),
    )


def _rank_or_after(value: str, preferred_order: Sequence[str]) -> tuple[int, str]:
    try:
        return preferred_order.index(value), value
    except ValueError:
        return len(preferred_order), value


def _review_queue_write_diagnostic(
    *,
    table_name: str,
    reason: str,
    row_index: int,
    detail: str,
    severity: str = "skipped",
) -> dict[str, Any]:
    return {
        "table": table_name,
        "reason": reason,
        "severity": severity,
        "row_index": row_index,
        "detail": detail,
    }


def _build_review_queue_write_plan_rows_summary(
    *,
    rows: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    write_plan: "PublicationWritePlan",
    table_name: str,
) -> dict[str, Any]:
    task_summary = summarize_review_queue_tasks(rows)
    return dict(
        _json_safe(
            {
                "summary_kind": "review_queue_write_plan_rows_summary.v1",
                "review_queue_table": table_name,
                "review_queue_row_count": len(rows),
                "review_queue_table_row_counts": {table_name: len(rows)}
                if rows
                else {},
                "diagnostic_count": len(diagnostics),
                "diagnostics_by_severity": _ordered_counts(
                    _summary_text(diagnostic, "severity")
                    for diagnostic in diagnostics
                ),
                "task_kind_counts": task_summary["task_kind_counts"],
                "task_status_counts": task_summary["task_status_counts"],
                "trust_status_counts": task_summary["trust_status_counts"],
                "severity_counts": task_summary["severity_counts"],
                "priority_counts": task_summary["priority_counts"],
                "source_family_counts": task_summary["source_family_counts"],
                "blocker_reason_counts": task_summary["blocker_reason_counts"],
                "write_plan_table_order": list(write_plan.ordered_tables()),
                "write_plan_table_modes": {
                    table: write_plan.mode_for(table)
                    for table in write_plan.ordered_tables()
                },
                "write_plan_row_counts": dict(
                    write_plan.audit_summary.planned_row_counts
                ),
            }
        )
    )


def _queue_blocker_summaries(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    summaries = row.get("blocker_summaries")
    if not isinstance(summaries, Sequence) or isinstance(summaries, (str, bytes)):
        return ()
    return tuple(summary for summary in summaries if isinstance(summary, Mapping))


def _queue_target_ids(
    candidate: Mapping[str, Any],
    expression: Mapping[str, Any],
) -> dict[str, str]:
    values = {
        "candidate_id": _first_context_text(
            (candidate, "candidate_id"),
            (expression, "candidate_id"),
        ),
        "source_candidate_id": _first_context_text(
            (candidate, "source_candidate_id"),
            (expression, "source_candidate_id"),
        ),
        "expression_id": _first_context_text((expression, "expression_id")),
        "source_expression_id": _first_context_text(
            (expression, "source_expression_id"),
            (candidate, "source_expression_id"),
        ),
        "artifact_id": _first_context_text(
            (expression, "artifact_id"),
            (candidate, "artifact_id"),
        ),
        "version_id": _first_context_text((expression, "version_id")),
        "snapshot_id": _first_context_text(
            (candidate, "snapshot_id"),
            (expression, "snapshot_id"),
        ),
    }
    return {key: value for key, value in values.items() if value}


def _source_context_text(
    candidate: Mapping[str, Any],
    expression: Mapping[str, Any],
    kind: str,
) -> str:
    keys = (
        ("source_family", "fixture_family", "family")
        if kind == "family"
        else ("source_system", "system")
    )
    for row in (expression, candidate):
        for key in keys:
            value = _text(row, key)
            if value:
                return value
        payload = _mapping(row.get("source_payload"))
        for key in keys:
            value = _text(payload, key)
            if value:
                return value
    return ""


def _first_context_text(*pairs: tuple[Mapping[str, Any], str]) -> str:
    for row, key in pairs:
        value = row.get(key)
        if value is None:
            continue
        text = value if isinstance(value, str) else str(value)
        text = text.strip()
        if text:
            return text
    return ""


def _blocker_summary(blocker: str) -> dict[str, str]:
    return {
        "reason": _blocker_reason(blocker),
        "detail": blocker,
    }


def _blocker_reason(blocker: str) -> str:
    text = blocker.lower()
    if "candidate_status" in text or "review_status is blocked" in text:
        return "blocked_status"
    if "parse" in text or "roundtrip" in text or "canonical symbolic" in text:
        return "parse"
    if "dimension" in text or "dimensional" in text or "dim_signature" in text:
        return "dimension"
    if (
        "validation" in text
        or "validity bound" in text
        or "numpy" in text
        or "sympy" in text
        or "canonical_expr_hash" in text
        or "topology_hash" in text
    ):
        return "symbolic_validation"
    if (
        "reference" in text
        or "source" in text
        or "relationship" in text
        or "mechanism classification" in text
    ):
        return "source_verification"
    if "review" in text or "reviewer" in text or "reviewed" in text:
        return "human_review"
    if "missing" in text or "required" in text:
        return "missing_payload"
    return "other"


def _queue_severity_priority(
    task_kind: str,
    achieved_status: str,
    blocker_summaries: Sequence[Mapping[str, str]],
) -> tuple[str, str]:
    reasons = {_text(summary, "reason") for summary in blocker_summaries}
    if task_kind == "blocked_resolution":
        if {"blocked_status", "parse"} & reasons:
            return "critical", "p0"
        return "high", "p1"
    if task_kind == "human_review_required":
        if achieved_status in {"raw_imported", "parsed"}:
            return "high", "p1"
        return "medium", "p2"
    return "low", "p3"


def _suggested_reviewer_focus(
    task_kind: str,
    blocker_summaries: Sequence[Mapping[str, str]],
) -> tuple[str, ...]:
    if task_kind == "audit_complete":
        return ("audit/completion metadata only; no human work required",)
    reason_focus = {
        "blocked_status": "resolve blocked status or upstream failure",
        "parse": "inspect parse confidence, roundtrip, and canonical symbolic payload",
        "dimension": "verify dimensions and dimensional consistency evidence",
        "symbolic_validation": "review symbolic validation, validity bounds, and NumPy evidence",
        "source_verification": "verify references, mechanism classification, and dependency relationships",
        "human_review": "complete human review evidence and validity-bound signoff",
        "missing_payload": "locate missing ingestion payload or target identifiers",
        "other": "inspect unresolved review blockers",
    }
    focus: list[str] = []
    for summary in blocker_summaries:
        reason = _text(summary, "reason") or "other"
        item = reason_focus.get(reason, reason_focus["other"])
        if item not in focus:
            focus.append(item)
    if not focus:
        focus.append("complete human review signoff")
    return tuple(focus)


def _stable_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _review_gates(review_row: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    gates = review_row.get("gates")
    if not isinstance(gates, Sequence) or isinstance(gates, (str, bytes)):
        return ()
    return tuple(_gate_row(gate) for gate in gates)


def _gate_row(gate: Any) -> Mapping[str, Any]:
    if isinstance(gate, ReviewGateResult):
        return {
            "status": gate.status,
            "passed": gate.passed,
            "blockers": list(gate.blockers),
            "evidence": dict(gate.evidence or {}),
        }
    return gate if isinstance(gate, Mapping) else {}


def _review_blockers(review_row: Mapping[str, Any]) -> tuple[str, ...]:
    blockers = review_row.get("blockers")
    if isinstance(blockers, str):
        return (blockers,) if blockers else ()
    if isinstance(blockers, Iterable) and not isinstance(blockers, Mapping):
        blocker_texts: list[str] = []
        for blocker in blockers:
            if blocker is None:
                continue
            text = str(blocker).strip()
            if text:
                blocker_texts.append(text)
        return tuple(blocker_texts)
    return tuple(
        blocker
        for gate in _review_gates(review_row)
        for blocker in _review_blockers(gate)
    )


def _summary_text(row: Mapping[str, Any], key: str) -> str:
    return _text(row, key) or "<missing>"


def _ordered_counts(
    values: Iterable[str],
    preferred_order: Sequence[str] = (),
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    ordered = {
        value: counts[value]
        for value in preferred_order
        if value in counts
    }
    for value in sorted(counts):
        if value not in ordered:
            ordered[value] = counts[value]
    return ordered


def _ordered_gate_counts(
    gate_counts: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    ordered: dict[str, dict[str, int]] = {}
    for status in WORKFLOW_STATUSES:
        if status in gate_counts:
            counts = gate_counts[status]
            ordered[status] = {
                "passed": int(counts.get("passed", 0)),
                "failed": int(counts.get("failed", 0)),
            }
    for status in sorted(gate_counts):
        if status in ordered:
            continue
        counts = gate_counts[status]
        ordered[status] = {
            "passed": int(counts.get("passed", 0)),
            "failed": int(counts.get("failed", 0)),
        }
    return ordered


def _raw_imported_gate(
    candidate: Mapping[str, Any],
    expression: Mapping[str, Any],
) -> ReviewGateResult:
    blockers = []
    blockers.extend(_blocked_status_blockers(candidate, expression))
    if not (
        _text(candidate, "raw_formula")
        or _text(expression, "raw_formula")
        or _text(expression, "sympy_srepr")
        or _mapping(candidate.get("source_payload"))
    ):
        blockers.append("raw import payload is missing")
    return _gate("raw_imported", blockers)


def _parsed_gate(
    candidate: Mapping[str, Any],
    expression: Mapping[str, Any],
    min_parse_confidence: float,
) -> ReviewGateResult:
    blockers = []
    parse_status = _text(expression, "parse_status")
    parse_confidence = max(
        _float(candidate.get("parse_confidence")),
        _float(expression.get("parse_confidence")),
    )
    evidence = _evidence(expression)
    roundtrip_status = _nested_text(
        evidence,
        ("parse_roundtrip", "status"),
        "parse_roundtrip_status",
        ("roundtrip", "status"),
    )

    if parse_status not in _PARSED_PARSE_STATUSES:
        blockers.append("expression parse_status must be parsed or normalized")
    if parse_status in _BLOCKED_STATUSES:
        blockers.append(f"expression parse_status is {parse_status}")
    if parse_confidence < min_parse_confidence:
        blockers.append(
            f"parse_confidence must be >= {min_parse_confidence:g}"
        )
    if not (_text(expression, "sympy_srepr") or _text(expression, "canonical_expr_hash")):
        blockers.append("canonical symbolic payload is missing")
    if not _is_pass(roundtrip_status):
        blockers.append("parse roundtrip evidence must pass")
    return _gate("parsed", blockers, {"parse_roundtrip_status": roundtrip_status})


def _dimension_resolved_gate(
    expression: Mapping[str, Any],
    variables: Sequence[Mapping[str, Any]],
    io_specs: Sequence[Mapping[str, Any]],
) -> ReviewGateResult:
    blockers = []
    evidence = _evidence(expression)
    dimensional_status = _nested_text(
        evidence,
        ("dimensional_analysis", "status"),
        "dimensional_consistency_status",
        ("dimension_check", "status"),
    )

    if not _text(expression, "dimensional_hash"):
        blockers.append("dimensional_hash is missing")
    if variables:
        missing_dims = [
            _text(variable, "symbol_name") or "<unnamed>"
            for variable in variables
            if _text(variable, "variable_role") != "intermediate"
            and _unknown_dimension_signature(_text(variable, "dim_signature"))
        ]
        unknown_sources = [
            _text(variable, "symbol_name") or "<unnamed>"
            for variable in variables
            if _text(variable, "dim_signature")
            and _text(variable, "dimension_source") in _UNKNOWN_DIMENSION_SOURCES
        ]
        if missing_dims:
            blockers.append(
                "variables missing dim_signature: " + ", ".join(missing_dims)
            )
        if unknown_sources:
            blockers.append(
                "variables missing dimension_source: " + ", ".join(unknown_sources)
            )
    else:
        blockers.append("symbolic variables are required for dimension review")

    io_missing = [
        _text(row, "name") or _text(row, "symbol_name") or _text(row, "label") or "<io>"
        for row in io_specs
        if _unknown_dimension_signature(_text(row, "dim_signature"))
    ]
    if io_missing:
        blockers.append("io_specs missing dim_signature: " + ", ".join(io_missing))
    if not _is_pass(dimensional_status):
        blockers.append("dimensional consistency evidence must pass")
    return _gate(
        "dimension_resolved",
        blockers,
        {"dimensional_consistency_status": dimensional_status},
    )


def _symbolically_validated_gate(
    expression: Mapping[str, Any],
    bounds: Sequence[Mapping[str, Any]],
) -> ReviewGateResult:
    blockers = []
    evidence = _evidence(expression)
    mechanism_evidence = _mechanism_classification_evidence(expression, evidence)
    numpy_evidence = _mapping(
        evidence.get("numpy_runtime")
        or evidence.get("generated_numpy")
        or evidence.get("numpy_codegen")
    )

    if _text(expression, "validation_status") != "passed":
        blockers.append("validation_status must be passed")
    if _text(expression, "validation_status") in _BLOCKED_STATUSES:
        blockers.append(f"validation_status is {_text(expression, 'validation_status')}")
    if not _text(expression, "canonical_expr_hash"):
        blockers.append("canonical_expr_hash is missing")
    if not _text(expression, "topology_hash"):
        blockers.append("topology_hash is missing")
    if not bounds and not _bool(evidence.get("validity_bounds_not_required")):
        blockers.append("validity bounds are required or must be explicitly waived")
    for index, bound in enumerate(bounds):
        label = _text(bound, "variable_name") or _text(bound, "regime_label") or str(index)
        if _text(bound, "review_status") == "blocked":
            blockers.append(f"validity bound {label} is blocked")
        if not (
            _text(bound, "validity_statement")
            or _text(bound, "regime_label")
            or bound.get("lower_value") is not None
            or bound.get("upper_value") is not None
        ):
            blockers.append(f"validity bound {label} has no constraint")

    if numpy_evidence:
        imports = tuple(str(item) for item in numpy_evidence.get("runtime_imports", ()))
        source = str(numpy_evidence.get("source", ""))
        if not _bool(numpy_evidence.get("no_sympy_runtime")):
            blockers.append("generated NumPy evidence must assert no_sympy_runtime")
        if "sympy" in imports or "import sympy" in source or "from sympy" in source:
            blockers.append("generated NumPy runtime evidence imports SymPy")
        if numpy_evidence.get("tests_passed") is not None and not _bool(
            numpy_evidence.get("tests_passed")
        ):
            blockers.append("generated NumPy runtime tests did not pass")

    return _gate(
        "symbolically_validated",
        blockers,
        {
            "numpy_runtime_checked": bool(numpy_evidence),
            **mechanism_evidence,
        },
    )


def _source_verified_gate(
    expression: Mapping[str, Any],
    references: Sequence[Mapping[str, Any]],
    relationships: Sequence[Mapping[str, Any]],
) -> ReviewGateResult:
    blockers = []
    evidence = _evidence(expression)
    mechanism_evidence = _mechanism_classification_evidence(expression, evidence)
    verified_refs = [ref for ref in references if _reference_verified(ref)]
    if not verified_refs:
        blockers.append("at least one verified reference is required")
    if not mechanism_evidence["has_mechanism_classification"]:
        blockers.append("mechanism classification evidence is required")

    dependencies = _declared_dependencies(evidence)
    dependency_relationships = [
        row for row in relationships if _text(row, "relationship_kind") in _DEPENDENCY_KINDS
    ]
    if dependencies and not dependency_relationships:
        blockers.append("declared constants/data dependencies need relationships")
    for relationship in dependency_relationships:
        kind = _text(relationship, "relationship_kind")
        label = _text(relationship, "relationship_label") or kind
        if not _bool(relationship.get("verified")):
            blockers.append(f"{label} relationship must be verified")
        if not (
            relationship.get("target_artifact_id")
            or relationship.get("target_version_id")
            or relationship.get("target_expression_id")
            or _text(relationship, "target_node_id")
        ):
            blockers.append(f"{label} relationship is missing a target")

    return _gate(
        "source_verified",
        blockers,
        {
            "verified_reference_count": len(verified_refs),
            "dependency_relationship_count": len(dependency_relationships),
            **mechanism_evidence,
        },
    )


def _human_reviewed_gate(
    expression: Mapping[str, Any],
    bounds: Sequence[Mapping[str, Any]],
) -> ReviewGateResult:
    blockers = []
    evidence = _evidence(expression)
    human_review = _mapping(evidence.get("human_review"))
    review_status = _text(expression, "review_status") or "unreviewed"
    if review_status == "blocked":
        blockers.append("expression review_status is blocked")
    elif review_status == "needs_human":
        blockers.append("expression review_status needs human review")
    elif review_status != "human_reviewed":
        blockers.append("expression review_status must be human_reviewed")
    if not (human_review.get("reviewer_id") or human_review.get("reviewed_by")):
        blockers.append("human review evidence must identify a reviewer")
    if not (human_review.get("reviewed_at") or human_review.get("timestamp")):
        blockers.append("human review evidence must include a review timestamp")
    for index, bound in enumerate(bounds):
        if _text(bound, "review_status") not in _REVIEWED_BOUND_STATUSES:
            label = _text(bound, "variable_name") or _text(bound, "regime_label") or str(index)
            blockers.append(f"validity bound {label} must be reviewed")
    return _gate("human_reviewed", blockers)


def _trust_status(
    candidate: Mapping[str, Any],
    expression: Mapping[str, Any],
    bounds: Sequence[Mapping[str, Any]],
    gates: Sequence[ReviewGateResult],
) -> str:
    statuses = {
        _text(candidate, "candidate_status"),
        _text(expression, "parse_status"),
        _text(expression, "review_status"),
        _text(expression, "validation_status"),
        *(_text(bound, "review_status") for bound in bounds),
    }
    if statuses & _BLOCKED_STATUSES:
        return "blocked"
    if all(gate.passed for gate in gates) and _text(expression, "review_status") == "human_reviewed":
        return "human_reviewed"
    return "needs_human"


def _blocked_status_blockers(
    candidate: Mapping[str, Any],
    expression: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    candidate_status = _text(candidate, "candidate_status")
    expression_status = _text(expression, "review_status")
    if candidate_status in _BLOCKED_STATUSES:
        blockers.append(f"candidate_status is {candidate_status}")
    if expression_status == "blocked":
        blockers.append("expression review_status is blocked")
    return blockers


def _published_gate(gates: Sequence[ReviewGateResult]) -> ReviewGateResult:
    blockers = []
    for gate in gates:
        blockers.extend(gate.blockers)
    return _gate("published", blockers)


def _gate(
    status: str,
    blockers: Sequence[str],
    evidence: Mapping[str, Any] | None = None,
) -> ReviewGateResult:
    return ReviewGateResult(
        status=status,
        passed=not blockers,
        blockers=tuple(blockers),
        evidence=evidence,
    )


def _row(value: Mapping[str, Any] | Any | None) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"cannot treat {type(value)!r} as a review row")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _evidence(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(row.get("evidence_json"))


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return value if isinstance(value, str) else ""


def _unknown_dimension_signature(value: str) -> bool:
    return value.strip().lower() in _UNKNOWN_DIM_SIGNATURES


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _PASS_VALUES
    return bool(value)


def _is_pass(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in _PASS_VALUES
    return value in _PASS_VALUES


def _nested_text(row: Mapping[str, Any], *paths: str | tuple[str, ...]) -> str:
    for path in paths:
        parts = (path,) if isinstance(path, str) else path
        current: Any = row
        for part in parts:
            if not isinstance(current, Mapping) or part not in current:
                current = None
                break
            current = current[part]
        if isinstance(current, str):
            return current
        if isinstance(current, bool):
            return str(current).lower()
    return ""


def _reference_verified(reference: Mapping[str, Any]) -> bool:
    has_locator = any(
        _text(reference, key)
        for key in (
            "doi",
            "url",
            "source_uri",
            "reference_uri",
            "isbn",
            "arxiv_id",
            "title",
        )
    )
    if not has_locator:
        return False
    if reference.get("verified") is not None:
        return _bool(reference.get("verified"))
    status = _text(reference, "review_status") or _text(reference, "verification_status")
    return status in {"verified", "source_verified", "human_reviewed", "automated_pass"}


def _declared_dependencies(evidence: Mapping[str, Any]) -> tuple[Any, ...]:
    constants = evidence.get("required_constants") or evidence.get("constants") or ()
    data = evidence.get("required_data_artifacts") or evidence.get("data_dependencies") or ()
    if isinstance(constants, str):
        constants = (constants,)
    if isinstance(data, str):
        data = (data,)
    if not isinstance(constants, Sequence):
        constants = ()
    if not isinstance(data, Sequence):
        data = ()
    return (*constants, *data)


def _mechanism_classification_evidence(
    expression: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    tags: list[str] = []
    archetypes: list[str] = []
    details: list[str] = []
    sources: list[str] = []

    def add_strings(target: list[str], values: Any, source: str) -> None:
        items = _string_sequence(values)
        if not items:
            return
        target.extend(item for item in items if item not in target)
        if source not in sources:
            sources.append(source)

    def add_detail(value: Any, source: str) -> None:
        items = _string_sequence(value)
        if not items:
            return
        details.extend(item for item in items if item not in details)
        if source not in sources:
            sources.append(source)

    add_strings(tags, expression.get("mechanism_tags"), "expression.mechanism_tags")
    add_strings(
        archetypes,
        expression.get("behavioral_archetypes"),
        "expression.behavioral_archetypes",
    )
    for key in (
        "mechanism_classification",
        "mechanism_label",
        "classification_label",
    ):
        add_detail(expression.get(key), f"expression.{key}")

    add_strings(tags, evidence.get("mechanism_tags"), "evidence_json.mechanism_tags")
    add_strings(
        archetypes,
        evidence.get("behavioral_archetypes"),
        "evidence_json.behavioral_archetypes",
    )
    for key in (
        "mechanism_classification",
        "mechanism_label",
        "classification_label",
    ):
        add_detail(evidence.get(key), f"evidence_json.{key}")

    for container_key in ("mechanism", "classification", "mechanism_classification"):
        value = evidence.get(container_key)
        if isinstance(value, Mapping):
            source = f"evidence_json.{container_key}"
            add_strings(
                tags,
                value.get("mechanism_tags")
                or value.get("mechanisms")
                or value.get("physics_mechanisms"),
                source,
            )
            add_strings(
                archetypes,
                value.get("behavioral_archetypes") or value.get("archetypes"),
                source,
            )
            for detail_key in (
                "mechanism",
                "mechanism_label",
                "mechanism_class",
                "classification",
                "classification_label",
                "class",
                "category",
                "rationale",
                "basis",
                "justification",
                "notes",
            ):
                add_detail(value.get(detail_key), source)
        else:
            add_detail(value, f"evidence_json.{container_key}")

    return {
        "has_mechanism_classification": bool(tags or archetypes or details),
        "mechanism_tag_count": len(tags),
        "behavioral_archetype_count": len(archetypes),
        "classification_detail_count": len(details),
        "mechanism_evidence_sources": list(sources),
    }


def _string_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, Mapping) or not isinstance(value, Iterable):
        return ()
    strings = []
    for item in value:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                strings.append(stripped)
    return tuple(strings)
