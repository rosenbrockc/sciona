"""Production reviewer workflow/storage boundary for physics ingestion.

This module only packages already-computed review decisions into deterministic
workflow reports and inert storage write plans. It constructs no Supabase or
storage clients; callers own reviewer UX, persistence clients, and write timing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from sciona.physics_ingest.deployment import PhysicsIngestDeploymentStorageBundle
from sciona.physics_ingest.deployment import (
    build_physics_ingest_deployment_storage_bundle,
)
from sciona.physics_ingest.review import (
    REVIEW_QUEUE_TASKS_TABLE,
    ReviewAssessment,
    ReviewPublicationRows,
    ReviewQueueWritePlanRows,
    ReviewTrustReport,
    build_review_publication_status_rows,
    build_review_queue_write_plan_rows,
    materialize_review_queue_task_rows,
    summarize_review_assessments,
    summarize_review_queue_tasks,
)
from sciona.physics_ingest.write_plan import WriteMode, merge_publication_insert_rows


REVIEW_DEPLOYMENT_REPORT_VERSION = "physics-ingest-reviewer-workflow-report.v1"
REVIEW_DEPLOYMENT_REPORT_KIND = "physics_ingest_reviewer_workflow"


@dataclass(frozen=True)
class PhysicsIngestReviewDeploymentReport:
    """Side-effect-free reviewer workflow report and storage plan."""

    report: Mapping[str, Any]
    review_queue_rows: ReviewQueueWritePlanRows
    publication_status_rows: ReviewPublicationRows
    storage_bundle: PhysicsIngestDeploymentStorageBundle
    storage_preflight: Mapping[str, Any]

    @property
    def ok(self) -> bool:
        return bool(self.report["ok"])

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self.report)


def build_physics_ingest_review_deployment_report(
    reviews: Iterable[ReviewAssessment | ReviewTrustReport | Mapping[str, Any]],
    *,
    candidates: Iterable[Mapping[str, Any] | Any | None] = (),
    expressions: Iterable[Mapping[str, Any] | Any | None] = (),
    include_audit_complete: bool = True,
    include_publication_status_rows: bool = True,
    include_write_plans: bool = True,
    table_modes: Mapping[str, WriteMode] | None = None,
    conflict_keys_by_table: Mapping[str, Sequence[str]] | None = None,
) -> PhysicsIngestReviewDeploymentReport:
    """Package review rows into caller-owned reviewer workflow/storage reports.

    The returned storage bundle is inert. It can be preflighted or applied by the
    caller through the injected-client deployment helpers, but this function does
    not construct clients and performs no external IO.
    """

    review_rows = tuple(reviews)
    candidate_rows = tuple(candidates)
    expression_rows = tuple(expressions)
    review_queue = materialize_review_queue_task_rows(
        review_rows,
        candidates=candidate_rows,
        expressions=expression_rows,
        include_audit_complete=include_audit_complete,
    )
    review_queue_rows = build_review_queue_write_plan_rows(
        review_queue,
        include_write_plan=include_write_plans,
        table_modes=table_modes,
    )
    publication_status_rows = (
        _materialize_publication_status_rows(
            review_rows,
            candidates=candidate_rows,
            expressions=expression_rows,
        )
        if include_publication_status_rows
        else ReviewPublicationRows()
    )
    storage_table_modes = {
        "physics_equation_candidates": "upsert",
        "artifact_symbolic_expressions": "upsert",
        REVIEW_QUEUE_TASKS_TABLE: "upsert",
        **dict(table_modes or {}),
    }
    storage_bundle = build_physics_ingest_deployment_storage_bundle(
        publication_status_rows.to_upsert_rows(),
        review_queue_rows=review_queue_rows,
        table_modes=storage_table_modes,
    )
    storage_preflight = storage_bundle.preflight_supabase_write(
        conflict_keys_by_table=conflict_keys_by_table,
    ).to_dict()
    workflow_report = _workflow_report(
        reviews=review_rows,
        candidates=candidate_rows,
        expressions=expression_rows,
        publication_status_rows=publication_status_rows,
        review_queue_rows=review_queue_rows,
    )
    summary = _summary(
        workflow_report=workflow_report,
        publication_status_rows=publication_status_rows,
        review_queue_rows=review_queue_rows,
        storage_bundle=storage_bundle,
        storage_preflight=storage_preflight,
    )
    report = {
        "report_version": REVIEW_DEPLOYMENT_REPORT_VERSION,
        "report_kind": REVIEW_DEPLOYMENT_REPORT_KIND,
        "ok": summary["ok"],
        "side_effect_free": True,
        "preflight": True,
        "input": {
            "review_count": len(review_rows),
            "candidate_context_count": len(candidate_rows),
            "expression_context_count": len(expression_rows),
            "include_audit_complete": include_audit_complete,
            "include_publication_status_rows": include_publication_status_rows,
            "include_write_plans": include_write_plans,
        },
        "summary": summary,
        "workflow": workflow_report,
        "publication_status_rows": publication_status_rows.to_dict(),
        "review_queue_rows": review_queue_rows.to_dict(),
        "storage_bundle": storage_bundle.to_dict(),
        "storage_preflight": storage_preflight,
    }
    return PhysicsIngestReviewDeploymentReport(
        report=_json_safe(report),
        review_queue_rows=review_queue_rows,
        publication_status_rows=publication_status_rows,
        storage_bundle=storage_bundle,
        storage_preflight=storage_preflight,
    )


def _materialize_publication_status_rows(
    reviews: Sequence[ReviewAssessment | ReviewTrustReport | Mapping[str, Any]],
    *,
    candidates: Sequence[Mapping[str, Any] | Any | None],
    expressions: Sequence[Mapping[str, Any] | Any | None],
) -> ReviewPublicationRows:
    status_rows = [
        build_review_publication_status_rows(
            review,
            candidate=candidates[index] if index < len(candidates) else None,
            expression=expressions[index] if index < len(expressions) else None,
        )
        for index, review in enumerate(reviews)
    ]
    merged = merge_publication_insert_rows(
        *(rows.to_upsert_rows() for rows in status_rows)
    )
    diagnostics = tuple(
        diagnostic
        for rows in status_rows
        for diagnostic in rows.diagnostics
    )
    return ReviewPublicationRows(
        artifact_symbolic_expressions=tuple(
            merged.get("artifact_symbolic_expressions", ())
        ),
        physics_equation_candidates=tuple(
            merged.get("physics_equation_candidates", ())
        ),
        diagnostics=diagnostics,
    )


def _workflow_report(
    *,
    reviews: Sequence[ReviewAssessment | ReviewTrustReport | Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any] | Any | None],
    expressions: Sequence[Mapping[str, Any] | Any | None],
    publication_status_rows: ReviewPublicationRows,
    review_queue_rows: ReviewQueueWritePlanRows,
) -> dict[str, Any]:
    queue_rows = review_queue_rows.to_insert_rows().get(REVIEW_QUEUE_TASKS_TABLE, [])
    status_insert_rows = publication_status_rows.to_upsert_rows()
    replay_keys = [str(row.get("replay_key") or "") for row in queue_rows]
    workflow = {
        "workflow_kind": "physics_ingest_reviewer_workflow.v1",
        "review_summary": summarize_review_assessments(reviews),
        "review_queue_summary": summarize_review_queue_tasks(queue_rows),
        "publication_status_summary": {
            "candidate_status_patch_count": len(
                status_insert_rows.get("physics_equation_candidates", ())
            ),
            "expression_status_patch_count": len(
                status_insert_rows.get("artifact_symbolic_expressions", ())
            ),
            "diagnostic_count": len(publication_status_rows.diagnostics),
        },
        "target_context": {
            "candidate_context_count": len(candidates),
            "expression_context_count": len(expressions),
        },
        "replay_keys": replay_keys,
        "replay_keys_digest": _stable_json_sha256(replay_keys),
        "storage_rows_digest": _stable_json_sha256(
            {
                "publication_status_rows": status_insert_rows,
                "review_queue_rows": queue_rows,
            }
        ),
    }
    return _json_safe(workflow)


def _summary(
    *,
    workflow_report: Mapping[str, Any],
    publication_status_rows: ReviewPublicationRows,
    review_queue_rows: ReviewQueueWritePlanRows,
    storage_bundle: PhysicsIngestDeploymentStorageBundle,
    storage_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    status_rows = publication_status_rows.to_upsert_rows()
    queue_rows = review_queue_rows.to_insert_rows().get(REVIEW_QUEUE_TASKS_TABLE, [])
    missing_conflict_metadata = list(
        storage_preflight.get("missing_conflict_metadata_for_upserts") or ()
    )
    error_diagnostic_count = _severity_count(
        (
            *publication_status_rows.diagnostics,
            *review_queue_rows.diagnostics,
        ),
        "error",
    )
    return {
        "ok": not missing_conflict_metadata and error_diagnostic_count == 0,
        "review_count": int(
            (workflow_report.get("review_summary") or {}).get("assessment_count") or 0
        )
        if isinstance(workflow_report.get("review_summary"), Mapping)
        else 0,
        "review_queue_row_count": len(queue_rows),
        "candidate_status_patch_count": len(
            status_rows.get("physics_equation_candidates", ())
        ),
        "expression_status_patch_count": len(
            status_rows.get("artifact_symbolic_expressions", ())
        ),
        "publication_status_diagnostic_count": len(publication_status_rows.diagnostics),
        "review_queue_diagnostic_count": len(review_queue_rows.diagnostics),
        "error_diagnostic_count": error_diagnostic_count,
        "storage_table_count": int(storage_preflight.get("table_count") or 0),
        "storage_total_row_count": int(storage_preflight.get("total_row_count") or 0),
        "storage_bundle_total_row_count": int(
            storage_bundle.summary.get("total_row_count") or 0
        ),
        "missing_conflict_metadata_for_upserts": missing_conflict_metadata,
    }


def _severity_count(rows: Iterable[Any], severity: str) -> int:
    count = 0
    for row in rows:
        value = row.to_dict() if hasattr(row, "to_dict") else row
        if isinstance(value, Mapping) and str(value.get("severity") or "") == severity:
            count += 1
    return count


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
    "REVIEW_DEPLOYMENT_REPORT_KIND",
    "REVIEW_DEPLOYMENT_REPORT_VERSION",
    "PhysicsIngestReviewDeploymentReport",
    "build_physics_ingest_review_deployment_report",
]
