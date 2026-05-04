"""Side-effect-free bulk backfill planning for physics ingestion."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
import hashlib
import json
from typing import Any

from sciona.physics_ingest.coverage import build_phase7_coverage_summary_dict
from sciona.physics_ingest.pipeline import run_physics_publication_pipeline
from sciona.physics_ingest.publication import (
    ArtifactBinding,
    build_publication_load_result_from_normalized_drafts,
)
from sciona.physics_ingest.sources.execution_plan import (
    build_source_execution_readiness_report_dict,
)
from sciona.physics_ingest.supabase_adapter import (
    preflight_publication_postgrest_write,
)
from sciona.physics_ingest.write_plan import WriteMode, merge_publication_insert_rows


BACKFILL_REPORT_KIND = "physics_ingest_bulk_backfill_plan"


def build_physics_ingest_backfill_report(
    *,
    source_bundles: Iterable[Any] = (),
    publication_manifests: Iterable[Mapping[str, Any]] = (),
    normalized_drafts: Iterable[Any] = (),
    pdg_publication_rows: Any | None = None,
    review_publication_rows: Any | None = None,
    review_status_rows: Any | None = None,
    source_retrieval_run_plan: Any | None = None,
    retrieval_run_plan: Any | None = None,
    review_diagnostics: Iterable[Mapping[str, Any]] = (),
    normalization_diagnostics: Iterable[Mapping[str, Any]] = (),
    artifact_bindings: Mapping[str, Mapping[str, Any] | ArtifactBinding] | None = None,
    table_modes: Mapping[str, WriteMode] | None = None,
    include_phase7_coverage_summary: bool = True,
    include_source_request_envelopes: bool = False,
    include_publication_write_preflight: bool = False,
    include_execution_boundary_preflight: bool = False,
    include_rows: bool = False,
) -> dict[str, Any]:
    """Build a deterministic dry-run report for a bulk physics backfill.

    The helper performs no database or network IO. It accepts ordinary source
    bundles, symbolic publication manifests, optional normalized expression
    drafts, optional PDG publication rows, optional review publication status
    rows, and caller-supplied review or normalization diagnostics. All rows are
    routed through the existing publication pipeline and write-plan ordering so
    the report mirrors the eventual writer batches without touching a client.
    """

    if source_retrieval_run_plan is not None and retrieval_run_plan is not None:
        raise ValueError(
            "pass only one of source_retrieval_run_plan or retrieval_run_plan"
        )

    source_bundle_list = tuple(source_bundles)
    publication_manifest_list = tuple(publication_manifests)
    normalized_draft_list = tuple(normalized_drafts)
    data_artifact_seed_summaries = _data_artifact_seed_summaries(source_bundle_list)
    retrieval_report = _source_retrieval_report_section(
        source_retrieval_run_plan
        if source_retrieval_run_plan is not None
        else retrieval_run_plan
    )
    normalized_rows, normalized_publication_diagnostics = (
        _publication_rows_from_normalized_drafts(normalized_draft_list)
    )
    pdg_rows, pdg_diagnostics = _normalize_pdg_publication_rows(pdg_publication_rows)
    review_rows, review_row_diagnostics = _normalize_review_publication_rows(
        review_publication_rows,
        review_status_rows=review_status_rows,
    )
    review_diagnostic_rows = _normalize_diagnostics(
        review_diagnostics,
        default_stage="review",
    )
    normalization_diagnostic_rows = _normalize_diagnostics(
        normalization_diagnostics,
        default_stage="normalization",
    )
    pdg_diagnostic_rows = _normalize_diagnostics(
        pdg_diagnostics,
        default_stage="pdg_cdg_publication",
    )
    normalized_publication_diagnostic_rows = _normalize_diagnostics(
        normalized_publication_diagnostics,
        default_stage="normalized_draft_publication",
    )
    review_publication_diagnostic_rows = _normalize_diagnostics(
        review_row_diagnostics,
        default_stage="review_publication",
    )
    retrieval_diagnostic_rows = _normalize_source_retrieval_diagnostics(
        retrieval_report["diagnostics"]
    )
    external_diagnostics = (
        *review_diagnostic_rows,
        *normalization_diagnostic_rows,
        *normalized_publication_diagnostic_rows,
        *review_publication_diagnostic_rows,
        *pdg_diagnostic_rows,
        *retrieval_diagnostic_rows,
    )
    additional_insert_rows = merge_publication_insert_rows(
        normalized_rows,
        pdg_rows,
        review_rows,
    )
    effective_table_modes = _table_modes_with_review_upserts(
        table_modes,
        review_rows,
    )

    pipeline_result = run_physics_publication_pipeline(
        source_bundles=source_bundle_list,
        publication_manifests=publication_manifest_list,
        additional_insert_rows=additional_insert_rows,
        additional_diagnostics=external_diagnostics,
        artifact_bindings=artifact_bindings,
        table_modes=effective_table_modes,
        dry_run=True,
    )
    insert_rows = pipeline_result.write_plan.to_insert_rows()
    coverage_rows_by_table = _phase7_coverage_rows_by_table(insert_rows)
    phase7_coverage_row_counts = {
        table: len(rows) for table, rows in coverage_rows_by_table.items()
    }
    publication_readiness_summary = _publication_readiness_summary(insert_rows)
    diagnostics = tuple(pipeline_result.diagnostics)
    retry_diagnostics = _diagnostics_by_severity(diagnostics, severity="error")
    skip_diagnostics = _diagnostics_by_severity(diagnostics, severity="skipped")
    has_errors = bool(retry_diagnostics)
    replay_keys = _replay_keys(pipeline_result.write_plan.to_dict()["batches"])

    report: dict[str, Any] = {
        "report_kind": BACKFILL_REPORT_KIND,
        "dry_run": True,
        "ok": not has_errors,
        "input_summary": {
            "source_bundle_count": len(source_bundle_list),
            "publication_manifest_count": len(publication_manifest_list),
            "data_artifact_seed_count": len(data_artifact_seed_summaries),
            "normalized_draft_count": len(normalized_draft_list),
            "normalized_draft_table_count": len(normalized_rows),
            "normalized_draft_row_count": sum(
                len(rows) for rows in normalized_rows.values()
            ),
            "pdg_table_count": len(pdg_rows),
            "pdg_row_count": sum(len(rows) for rows in pdg_rows.values()),
            "review_publication_table_count": len(review_rows),
            "review_publication_row_count": sum(
                len(rows) for rows in review_rows.values()
            ),
            "phase7_coverage_row_count": sum(phase7_coverage_row_counts.values()),
            "review_diagnostic_count": len(review_diagnostic_rows),
            "normalization_diagnostic_count": len(normalization_diagnostic_rows),
            "source_retrieval_step_count": retrieval_report["step_count"],
            "source_retrieval_diagnostic_count": len(retrieval_diagnostic_rows),
        },
        "source_family_counts": _source_family_counts(
            source_bundle_list,
            publication_manifest_list,
            pdg_rows,
            normalized_rows=normalized_rows,
            review_rows=review_rows,
        ),
        "data_artifact_seeds": data_artifact_seed_summaries,
        "table_row_counts": dict(pipeline_result.write_plan.audit_summary.planned_row_counts),
        "publication_readiness_summary": publication_readiness_summary,
        "dry_run_write_plan": _write_plan_summary(pipeline_result),
        "replay_keys": replay_keys,
        "audit_replay": _audit_replay_section(
            pipeline_result=pipeline_result,
            input_summary={
                "source_bundle_count": len(source_bundle_list),
                "publication_manifest_count": len(publication_manifest_list),
                "data_artifact_seed_count": len(data_artifact_seed_summaries),
                "normalized_draft_count": len(normalized_draft_list),
                "pdg_table_count": len(pdg_rows),
                "pdg_row_count": sum(len(rows) for rows in pdg_rows.values()),
                "review_publication_table_count": len(review_rows),
                "review_publication_row_count": sum(
                    len(rows) for rows in review_rows.values()
                ),
                "source_retrieval_step_count": retrieval_report["step_count"],
                "source_retrieval_diagnostic_count": len(retrieval_diagnostic_rows),
            },
            data_artifact_seed_summaries=data_artifact_seed_summaries,
            replay_keys=replay_keys,
            diagnostics=diagnostics,
            retry_diagnostics=retry_diagnostics,
            skip_diagnostics=skip_diagnostics,
            source_retrieval_report=retrieval_report["report"],
        ),
        "retry_diagnostics": retry_diagnostics,
        "skip_diagnostics": skip_diagnostics,
        "external_diagnostics": {
            "review": [dict(row) for row in review_diagnostic_rows],
            "normalization": [dict(row) for row in normalization_diagnostic_rows],
            "normalized_draft_publication": [
                dict(row) for row in normalized_publication_diagnostic_rows
            ],
            "review_publication": [
                dict(row) for row in review_publication_diagnostic_rows
            ],
            "pdg_cdg_publication": [dict(row) for row in pdg_diagnostic_rows],
            "source_retrieval": [dict(row) for row in retrieval_diagnostic_rows],
        },
        "source_retrieval_run_plan": retrieval_report["report"],
        "diagnostic_summary": _diagnostic_summary(diagnostics),
    }
    if include_source_request_envelopes or include_execution_boundary_preflight:
        report["source_request_envelope_preflight"] = (
            _source_request_envelope_preflight_section(retrieval_report["plan"])
        )
    if include_publication_write_preflight or include_execution_boundary_preflight:
        report["publication_storage_write_preflight"] = (
            _publication_storage_write_preflight_section(pipeline_result.write_plan)
        )
    if include_phase7_coverage_summary:
        report["phase7_coverage_row_counts"] = phase7_coverage_row_counts
        report["phase7_coverage_summary"] = build_phase7_coverage_summary_dict(
            row
            for rows in coverage_rows_by_table.values()
            for row in rows
        )
    report["dashboard_summary"] = _dashboard_summary(report)
    if include_rows:
        report["insert_rows_by_table"] = insert_rows

    _assert_json_serializable(report)
    return report


def _publication_rows_from_normalized_drafts(
    drafts: tuple[Any, ...],
) -> tuple[dict[str, list[dict[str, Any]]], tuple[Any, ...]]:
    if not drafts:
        return {}, ()
    result = build_publication_load_result_from_normalized_drafts(drafts)
    return _copy_rows_by_table(result.to_insert_rows()), tuple(result.diagnostics)


def _normalize_pdg_publication_rows(
    value: Any | None,
) -> tuple[dict[str, list[dict[str, Any]]], tuple[Mapping[str, Any], ...]]:
    if value is None:
        return {}, ()
    if hasattr(value, "to_insert_rows"):
        rows = value.to_insert_rows()
        diagnostics = tuple(getattr(value, "diagnostics", ()) or ())
        return _copy_rows_by_table(rows), diagnostics
    if isinstance(value, Mapping):
        return _copy_rows_by_table(value), ()
    raise ValueError("pdg_publication_rows must be a mapping or expose to_insert_rows()")


def _normalize_review_publication_rows(
    value: Any | None,
    *,
    review_status_rows: Any | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], tuple[Any, ...]]:
    rows_values = tuple(
        row_value for row_value in (value, review_status_rows) if row_value is not None
    )
    if not rows_values:
        return {}, ()

    rows_by_table: list[dict[str, list[dict[str, Any]]]] = []
    diagnostics: list[Any] = []
    for row_value in rows_values:
        if hasattr(row_value, "to_upsert_rows"):
            rows = row_value.to_upsert_rows()
            diagnostics.extend(tuple(getattr(row_value, "diagnostics", ()) or ()))
        elif isinstance(row_value, Mapping):
            rows = row_value
        else:
            raise ValueError(
                "review publication rows must be a mapping or expose to_upsert_rows()"
            )
        rows_by_table.append(_copy_rows_by_table(rows))
    return merge_publication_insert_rows(*rows_by_table), tuple(diagnostics)


def _table_modes_with_review_upserts(
    table_modes: Mapping[str, WriteMode] | None,
    review_rows: Mapping[str, list[dict[str, Any]]],
) -> dict[str, WriteMode]:
    effective: dict[str, WriteMode] = {
        table: "upsert" for table, rows in review_rows.items() if rows
    }
    effective.update(dict(table_modes or {}))
    return effective


def _copy_rows_by_table(
    rows_by_table: Mapping[str, Iterable[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    copied: dict[str, list[dict[str, Any]]] = {}
    for table, rows in rows_by_table.items():
        copied[str(table)] = [dict(row) for row in rows]
    return copied


def _phase7_coverage_rows_by_table(
    rows_by_table: Mapping[str, Iterable[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        table: [dict(row) for row in rows_by_table.get(table, ())]
        for table in (
            "physics_equation_candidates",
            "artifact_symbolic_expressions",
        )
    }


def _publication_readiness_summary(
    rows_by_table: Mapping[str, Iterable[Mapping[str, Any]]],
) -> dict[str, Any]:
    table_rows = _phase7_coverage_rows_by_table(rows_by_table)
    combined_rows = [
        row
        for table in ("physics_equation_candidates", "artifact_symbolic_expressions")
        for row in table_rows[table]
    ]

    return {
        "report_version": "physics-ingest-publication-readiness-summary.v1",
        "row_count": len(combined_rows),
        "table_row_counts": {
            table: len(rows) for table, rows in sorted(table_rows.items())
        },
        "by_candidate_status": _status_counts(combined_rows, "candidate_status"),
        "by_parse_status": _status_counts(combined_rows, "parse_status"),
        "by_review_status": _status_counts(combined_rows, "review_status"),
        "by_validation_status": _status_counts(combined_rows, "validation_status"),
        "readiness_stage_counts": _readiness_stage_counts(combined_rows),
        "by_table": {
            table: {
                "row_count": len(rows),
                "by_candidate_status": _status_counts(rows, "candidate_status"),
                "by_parse_status": _status_counts(rows, "parse_status"),
                "by_review_status": _status_counts(rows, "review_status"),
                "by_validation_status": _status_counts(rows, "validation_status"),
                "readiness_stage_counts": _readiness_stage_counts(rows),
            }
            for table, rows in sorted(table_rows.items())
        },
    }


def _status_counts(
    rows: Iterable[Mapping[str, Any]],
    status_key: str,
) -> dict[str, int]:
    counts = Counter(
        str(row.get(status_key) or "unknown")
        for row in rows
        if status_key in row
    )
    return dict(sorted(counts.items()))


def _readiness_stage_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(_readiness_stage(row) for row in rows)
    return dict(sorted(counts.items()))


def _readiness_stage(row: Mapping[str, Any]) -> str:
    candidate_status = str(row.get("candidate_status") or "")
    parse_status = str(row.get("parse_status") or "")
    review_status = str(row.get("review_status") or "")
    validation_status = str(row.get("validation_status") or "")
    publication_status = str(
        row.get("publication_status") or row.get("published_status") or ""
    )

    if (
        candidate_status in {"blocked", "parse_failed"}
        or parse_status in {"blocked", "parse_failed"}
        or review_status == "blocked"
        or validation_status in {"blocked", "error", "fail", "failed"}
        or bool(row.get("blockers"))
    ):
        return "blocked"
    if (
        candidate_status == "published"
        or publication_status == "published"
        or bool(row.get("published"))
        or bool(row.get("published_at"))
    ):
        return "published"
    if (
        candidate_status == "human_reviewed"
        or (
            review_status == "human_reviewed"
            and validation_status == "passed"
            and parse_status in {"parsed", "normalized"}
        )
    ):
        return "publishable_candidate"
    if review_status == "human_reviewed":
        return "human_reviewed"
    if review_status == "needs_human":
        return "needs_human_review"
    if review_status == "automated_pass":
        return "automated_pass"
    if validation_status == "passed":
        return "validated"
    if (
        candidate_status
        in {
            "parsed",
            "dimension_resolved",
            "symbolically_validated",
            "source_verified",
        }
        or parse_status in {"parsed", "normalized"}
        or bool(row.get("sympy_srepr"))
        or bool(row.get("canonical_expr_hash"))
        or bool(row.get("topology_hash"))
    ):
        return "parsed"
    if candidate_status or parse_status:
        return "raw_or_pending"
    return "unknown"


def _data_artifact_seed_summaries(
    source_bundles: tuple[Any, ...],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for bundle_index, bundle in enumerate(source_bundles):
        snapshot_row = _snapshot_row(bundle)
        bundle_key = _bundle_key(bundle, snapshot_row, bundle_index)
        for seed_index, seed in enumerate(_bundle_data_artifact_seeds(bundle)):
            summaries.append(
                {
                    "bundle_index": bundle_index,
                    "bundle_key": bundle_key,
                    "seed_index": seed_index,
                    "source_system": str(
                        seed.get("source_system")
                        or snapshot_row.get("source_system")
                        or ""
                    ),
                    "source_id": str(seed.get("source_id") or ""),
                    "source_uri": str(
                        seed.get("source_uri")
                        or snapshot_row.get("source_uri")
                        or ""
                    ),
                    "artifact_kind": str(seed.get("artifact_kind") or ""),
                    "artifact_role": str(seed.get("artifact_role") or ""),
                    "fqdn": str(seed.get("fqdn") or ""),
                    "label": str(seed.get("label") or ""),
                }
            )
    return summaries


def _bundle_data_artifact_seeds(bundle: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(bundle, Mapping):
        seeds = bundle.get("data_artifact_seeds") or ()
    else:
        seeds = getattr(bundle, "data_artifact_seeds", ()) or ()
    return tuple(seed for seed in seeds if isinstance(seed, Mapping))


def _bundle_key(bundle: Any, snapshot_row: Mapping[str, Any], index: int) -> str:
    if isinstance(bundle, Mapping):
        for key_name in ("bundle_key", "key", "name"):
            if bundle.get(key_name):
                return str(bundle[key_name])
    else:
        for key_name in ("bundle_key", "key", "name"):
            value = getattr(bundle, key_name, None)
            if value:
                return str(value)
    for key_name in ("source_system", "adapter_name", "source_uri"):
        if snapshot_row.get(key_name):
            return str(snapshot_row[key_name])
    return f"source_bundle:{index}"


def _normalize_diagnostics(
    diagnostics: Iterable[Mapping[str, Any] | Any],
    *,
    default_stage: str,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "stage": str(row.get("stage") or default_stage),
            "table": str(row.get("table") or ""),
            "reason": str(row.get("reason") or ""),
            "severity": str(row.get("severity") or "info"),
            "artifact_key": str(row.get("artifact_key") or ""),
            "atom_name": str(row.get("atom_name") or ""),
            "detail": str(row.get("detail") or ""),
        }
        for row in (_diagnostic_row(diagnostic) for diagnostic in diagnostics)
    )


def _diagnostic_row(diagnostic: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(diagnostic, Mapping):
        return diagnostic
    if hasattr(diagnostic, "to_dict"):
        row = diagnostic.to_dict()
        if isinstance(row, Mapping):
            return row
    return {
        "stage": getattr(diagnostic, "stage", ""),
        "table": getattr(diagnostic, "table", ""),
        "reason": getattr(diagnostic, "reason", ""),
        "severity": getattr(diagnostic, "severity", "info"),
        "artifact_key": getattr(diagnostic, "artifact_key", ""),
        "atom_name": getattr(diagnostic, "atom_name", ""),
        "detail": getattr(diagnostic, "detail", ""),
    }


def _source_retrieval_report_section(value: Any | None) -> dict[str, Any]:
    if value is None:
        empty_plan = {
            "manifest_version": "",
            "snapshot_key_prefix": "",
            "dry_run": True,
            "filters": {},
            "steps": [],
            "diagnostics": [],
        }
        return {
            "step_count": 0,
            "diagnostics": (),
            "plan": empty_plan,
            "report": {
                "manifest_version": "",
                "snapshot_key_prefix": "",
                "dry_run": True,
                "filters": {},
                "step_count": 0,
                "diagnostic_count": 0,
                "steps": [],
                "replay_keys": [],
            },
        }

    plan = _source_retrieval_plan_mapping(value)
    steps = tuple(_source_retrieval_step_mapping(step) for step in plan.get("steps", ()))
    diagnostics = tuple(
        _source_retrieval_diagnostic_mapping(diagnostic)
        for diagnostic in plan.get("diagnostics", ())
    )
    replay_keys = [
        str(step.get("replay_key") or "")
        for step in steps
        if step.get("replay_key")
    ]
    normalized_plan = {
        "manifest_version": str(plan.get("manifest_version") or ""),
        "snapshot_key_prefix": str(plan.get("snapshot_key_prefix") or ""),
        "dry_run": bool(plan.get("dry_run", True)),
        "filters": _json_safe_mapping(plan.get("filters") or {}),
        "steps": [_json_safe_mapping(dict(step)) for step in steps],
        "diagnostics": [_json_safe_mapping(dict(diagnostic)) for diagnostic in diagnostics],
    }

    return {
        "step_count": len(steps),
        "diagnostics": diagnostics,
        "plan": normalized_plan,
        "report": {
            "manifest_version": str(plan.get("manifest_version") or ""),
            "snapshot_key_prefix": str(plan.get("snapshot_key_prefix") or ""),
            "dry_run": bool(plan.get("dry_run", True)),
            "filters": _json_safe_mapping(plan.get("filters") or {}),
            "step_count": len(steps),
            "diagnostic_count": len(diagnostics),
            "steps": [_source_retrieval_step_summary(step) for step in steps],
            "replay_keys": replay_keys,
        },
    }


def _source_retrieval_plan_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict"):
        row = value.to_dict()
        if isinstance(row, Mapping):
            return row
    return {
        "manifest_version": getattr(value, "manifest_version", ""),
        "snapshot_key_prefix": getattr(value, "snapshot_key_prefix", ""),
        "dry_run": getattr(value, "dry_run", True),
        "filters": getattr(value, "filters", {}),
        "steps": getattr(value, "steps", ()),
        "diagnostics": getattr(value, "diagnostics", ()),
    }


def _source_retrieval_step_mapping(value: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict"):
        row = value.to_dict()
        if isinstance(row, Mapping):
            return row
    return {
        key: getattr(value, key, "")
        for key in (
            "step_index",
            "job_id",
            "endpoint_id",
            "source_system",
            "source_family",
            "snapshot_key",
            "method",
            "url",
            "endpoint_kind",
            "dry_run",
            "replay_key",
            "warnings",
        )
    }


def _source_retrieval_diagnostic_mapping(
    value: Mapping[str, Any] | Any,
) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict"):
        row = value.to_dict()
        if isinstance(row, Mapping):
            return row
    return {
        "severity": getattr(value, "severity", "info"),
        "job_id": getattr(value, "job_id", ""),
        "endpoint_id": getattr(value, "endpoint_id", ""),
        "message": getattr(value, "message", ""),
    }


def _source_retrieval_step_summary(step: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "step_index": int(step.get("step_index") or 0),
        "job_id": str(step.get("job_id") or ""),
        "endpoint_id": str(step.get("endpoint_id") or ""),
        "source_system": str(step.get("source_system") or ""),
        "source_family": str(step.get("source_family") or ""),
        "snapshot_key": str(step.get("snapshot_key") or ""),
        "method": str(step.get("method") or ""),
        "url": str(step.get("url") or ""),
        "endpoint_kind": str(step.get("endpoint_kind") or ""),
        "dry_run": bool(step.get("dry_run", True)),
        "replay_key": str(step.get("replay_key") or ""),
        "warnings": [str(warning) for warning in step.get("warnings", ())],
    }


def _source_request_envelope_preflight_section(
    retrieval_plan: Mapping[str, Any],
) -> dict[str, Any]:
    readiness = build_source_execution_readiness_report_dict(retrieval_plan)
    readiness_steps = {
        (
            int(step.get("step_index") or 0),
            str(step.get("job_id") or ""),
        ): step
        for step in readiness.get("steps", ())
        if isinstance(step, Mapping)
    }
    envelope_rows = []
    counts = Counter({"manual": 0, "network": 0, "blocked": 0})
    for step in retrieval_plan.get("steps", ()):
        if not isinstance(step, Mapping):
            continue
        readiness_step = readiness_steps.get(
            (
                int(step.get("step_index") or 0),
                str(step.get("job_id") or ""),
            ),
            {},
        )
        envelope = step.get("request_envelope") or {}
        envelope_mapping = envelope if isinstance(envelope, Mapping) else {}
        execution = envelope_mapping.get("execution") or {}
        execution_mapping = execution if isinstance(execution, Mapping) else {}
        expectation = _retrieval_execution_expectation(
            readiness_step=readiness_step,
            execution=execution_mapping,
            method=str(step.get("method") or ""),
            url=str(step.get("url") or ""),
        )
        counts[expectation] += 1
        envelope_rows.append(
            {
                "step_index": int(step.get("step_index") or 0),
                "job_id": str(step.get("job_id") or ""),
                "endpoint_id": str(step.get("endpoint_id") or ""),
                "source_system": str(step.get("source_system") or ""),
                "source_family": str(step.get("source_family") or ""),
                "method": str(step.get("method") or ""),
                "url": str(step.get("url") or ""),
                "endpoint_kind": str(step.get("endpoint_kind") or ""),
                "snapshot_key": str(step.get("snapshot_key") or ""),
                "replay_key": str(step.get("replay_key") or ""),
                "readiness_status": str(readiness_step.get("status") or ""),
                "execution_expectation": expectation,
                "network_required": bool(execution_mapping.get("network_required")),
                "network_io_allowed": bool(
                    execution_mapping.get("network_io_allowed")
                ),
                "manual_source": bool(execution_mapping.get("manual_source")),
                "io_performed": bool(execution_mapping.get("io_performed")),
                "request_envelope": _json_safe_mapping(envelope_mapping),
            }
        )

    execution_counts = {
        key: int(counts.get(key, 0)) for key in ("manual", "network", "blocked")
    }
    return {
        "report_version": "physics-ingest-source-request-envelope-preflight.v1",
        "step_count": len(envelope_rows),
        "manual_retrieval_expected_count": execution_counts["manual"],
        "network_retrieval_expected_count": execution_counts["network"],
        "blocked_retrieval_expected_count": execution_counts["blocked"],
        "execution_expectation_counts": execution_counts,
        "readiness_summary": _json_safe_mapping(readiness.get("summary") or {}),
        "readiness_diagnostic_count": len(readiness.get("diagnostics") or ()),
        "request_envelopes": envelope_rows,
    }


def _retrieval_execution_expectation(
    *,
    readiness_step: Mapping[str, Any],
    execution: Mapping[str, Any],
    method: str,
    url: str,
) -> str:
    readiness_status = str(readiness_step.get("status") or "")
    if readiness_status in {"offline_blocked", "blocked"}:
        return "blocked"
    if readiness_status == "manual":
        return "manual"
    mode = str(execution.get("mode") or "")
    if mode == "manual" or method == "MANUAL" or url.startswith("manual://"):
        return "manual"
    if bool(execution.get("network_required")) or mode == "network":
        return "network"
    if readiness_status:
        return "network"
    return "blocked"


def _publication_storage_write_preflight_section(write_plan: Any) -> dict[str, Any]:
    preflight = preflight_publication_postgrest_write(write_plan).to_dict()
    tables = tuple(
        table for table in preflight.get("tables", ()) if isinstance(table, Mapping)
    )
    return {
        "report_version": "physics-ingest-publication-storage-write-preflight.v1",
        "dry_run": True,
        "table_count": int(preflight.get("table_count") or 0),
        "total_row_count": int(preflight.get("total_row_count") or 0),
        "mode_counts": _sorted_counts(
            str(table.get("mode") or "unknown") for table in tables
        ),
        "tables": [_json_safe_mapping(table) for table in tables],
        "missing_conflict_metadata_for_upserts": [
            str(table)
            for table in preflight.get("missing_conflict_metadata_for_upserts", ())
        ],
        "adapter_capabilities": _json_safe_mapping(
            preflight.get("adapter_capabilities") or {}
        ),
    }


def _normalize_source_retrieval_diagnostics(
    diagnostics: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "stage": str(row.get("stage") or "source_retrieval"),
            "table": str(row.get("table") or ""),
            "reason": str(row.get("reason") or "retrieval_run_plan_diagnostic"),
            "severity": str(row.get("severity") or "info"),
            "artifact_key": str(row.get("artifact_key") or ""),
            "atom_name": str(row.get("atom_name") or ""),
            "detail": str(row.get("detail") or row.get("message") or ""),
            "job_id": str(row.get("job_id") or ""),
            "endpoint_id": str(row.get("endpoint_id") or ""),
        }
        for row in diagnostics
    )


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True))


def _source_family_counts(
    source_bundles: tuple[Any, ...],
    publication_manifests: tuple[Mapping[str, Any], ...],
    pdg_rows: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    normalized_rows: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    review_rows: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
) -> dict[str, dict[str, int]]:
    bundle_counts = Counter(
        _family_from_mapping(_snapshot_row(bundle), fallback="unknown_source_bundle")
        for bundle in source_bundles
    )
    manifest_counts = Counter(
        _family_from_mapping(manifest, fallback="unknown_publication_manifest")
        for manifest in publication_manifests
    )
    pdg_counts = Counter[str]()
    for rows in pdg_rows.values():
        for row in rows:
            pdg_counts[_family_from_mapping(row, fallback="physics_derivation_graph")] += 1
    normalized_counts = _row_family_counts(
        normalized_rows or {},
        fallback="normalized_draft",
    )
    review_publication_counts = _row_family_counts(
        review_rows or {},
        fallback="review_publication",
    )
    combined = (
        bundle_counts
        + manifest_counts
        + pdg_counts
        + normalized_counts
        + review_publication_counts
    )
    return {
        "source_bundles": dict(sorted(bundle_counts.items())),
        "publication_manifests": dict(sorted(manifest_counts.items())),
        "normalized_drafts": dict(sorted(normalized_counts.items())),
        "pdg_publication_rows": dict(sorted(pdg_counts.items())),
        "review_publication_rows": dict(sorted(review_publication_counts.items())),
        "combined": dict(sorted(combined.items())),
    }


def _row_family_counts(
    rows_by_table: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    fallback: str,
) -> Counter[str]:
    counts = Counter[str]()
    for rows in rows_by_table.values():
        for row in rows:
            counts[_family_from_mapping(row, fallback=fallback)] += 1
    return counts


def _snapshot_row(bundle: Any) -> Mapping[str, Any]:
    if isinstance(bundle, Mapping):
        row = bundle.get("snapshot_row")
    else:
        row = getattr(bundle, "snapshot_row", None)
    return row if isinstance(row, Mapping) else {}


def _family_from_mapping(row: Mapping[str, Any], *, fallback: str) -> str:
    for key in (
        "source_family",
        "source_system",
        "provider",
        "source_kind",
        "adapter_name",
    ):
        value = row.get(key)
        if value:
            return str(value)
    return fallback


def _write_plan_summary(pipeline_result: Any) -> dict[str, Any]:
    write_plan = pipeline_result.write_plan
    return {
        "audit_summary": write_plan.audit_summary.to_dict(),
        "data_artifact_seed_count": int(
            getattr(pipeline_result.summary, "data_artifact_seed_count", 0)
        ),
        "batches": [
            {
                "table": batch.table,
                "mode": write_plan.mode_for(batch.table),
                "row_count": batch.row_count,
                "conflict_keys": list(batch.conflict_keys),
                "dry_run": True,
            }
            for batch in write_plan.batches
        ],
    }


def _dashboard_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """Build compact, deterministic, JSON-safe rollups for dashboard cards."""

    input_summary = _json_safe_mapping(report.get("input_summary") or {})
    table_row_counts = _sorted_int_mapping(report.get("table_row_counts") or {})
    write_plan = report.get("dry_run_write_plan") or {}
    batches = tuple(
        batch for batch in write_plan.get("batches", ()) if isinstance(batch, Mapping)
    )
    diagnostic_summary = _json_safe_mapping(report.get("diagnostic_summary") or {})
    source_retrieval_report = report.get("source_retrieval_run_plan") or {}
    source_family_counts = report.get("source_family_counts") or {}
    publication_readiness = report.get("publication_readiness_summary") or {}

    summary: dict[str, Any] = {
        "ok": bool(report.get("ok")),
        "dry_run": bool(report.get("dry_run")),
        "input_counts": input_summary,
        "write_plan": {
            "batch_count": len(batches),
            "table_count": len(table_row_counts),
            "row_count": sum(table_row_counts.values()),
            "dry_run_batch_count": sum(
                1 for batch in batches if bool(batch.get("dry_run"))
            ),
            "mode_counts": _sorted_counts(
                str(batch.get("mode") or "unknown") for batch in batches
            ),
            "table_row_counts": table_row_counts,
        },
        "publication_readiness": _publication_readiness_dashboard_rollup(
            publication_readiness
        ),
        "source_retrieval": {
            "step_count": int(
                input_summary.get("source_retrieval_step_count")
                or source_retrieval_report.get("step_count")
                or 0
            ),
            "diagnostic_count": int(
                input_summary.get("source_retrieval_diagnostic_count")
                or source_retrieval_report.get("diagnostic_count")
                or 0
            ),
        },
        "diagnostics": {
            "by_severity": _sorted_int_mapping(
                diagnostic_summary.get("by_severity") or {}
            ),
            "by_reason": _sorted_int_mapping(diagnostic_summary.get("by_reason") or {}),
            "retry_count": len(report.get("retry_diagnostics") or ()),
            "skip_count": len(report.get("skip_diagnostics") or ()),
        },
        "source_family_counts": {
            "combined": _sorted_int_mapping(
                source_family_counts.get("combined") or {}
            ),
        },
        "data_artifacts": {
            "seed_count": int(input_summary.get("data_artifact_seed_count") or 0),
        },
    }
    phase7_rollup = _phase7_dashboard_rollup(report)
    if phase7_rollup is not None:
        summary["phase7_coverage"] = phase7_rollup
    return _json_safe_mapping(summary)


def _publication_readiness_dashboard_rollup(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "row_count": int(summary.get("row_count") or 0),
        "table_row_counts": _sorted_int_mapping(
            summary.get("table_row_counts") or {}
        ),
        "readiness_stage_counts": _sorted_int_mapping(
            summary.get("readiness_stage_counts") or {}
        ),
        "by_candidate_status": _sorted_int_mapping(
            summary.get("by_candidate_status") or {}
        ),
        "by_parse_status": _sorted_int_mapping(summary.get("by_parse_status") or {}),
        "by_review_status": _sorted_int_mapping(summary.get("by_review_status") or {}),
        "by_validation_status": _sorted_int_mapping(
            summary.get("by_validation_status") or {}
        ),
    }


def _phase7_dashboard_rollup(report: Mapping[str, Any]) -> dict[str, Any] | None:
    row_counts = report.get("phase7_coverage_row_counts")
    coverage_summary = report.get("phase7_coverage_summary")
    if not isinstance(row_counts, Mapping) and not isinstance(
        coverage_summary, Mapping
    ):
        return None

    summary: dict[str, Any] = {
        "row_count": int(
            report.get("input_summary", {}).get("phase7_coverage_row_count") or 0
        )
        if isinstance(report.get("input_summary"), Mapping)
        else 0,
        "table_row_counts": _sorted_int_mapping(row_counts or {}),
    }
    if isinstance(coverage_summary, Mapping):
        summary["summary"] = _json_safe_mapping(coverage_summary.get("summary") or {})
    return summary


def _sorted_int_mapping(value: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(key): int(count or 0)
        for key, count in sorted(value.items(), key=lambda pair: str(pair[0]))
    }


def _sorted_counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _replay_keys(batches: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    replay_keys: dict[str, list[str]] = {}
    for batch in batches:
        table = str(batch["table"])
        conflict_keys = tuple(str(key) for key in batch.get("conflict_keys", ()))
        table_keys = []
        for index, row in enumerate(batch.get("rows", ())):
            if conflict_keys and all(row.get(key) not in (None, "") for key in conflict_keys):
                identity = "|".join(f"{key}={row[key]}" for key in conflict_keys)
            else:
                identity = f"row_index={index}|sha256={_stable_row_hash(row)}"
            table_keys.append(f"{table}|{identity}")
        replay_keys[table] = table_keys
    return replay_keys


def _audit_replay_section(
    *,
    pipeline_result: Any,
    input_summary: Mapping[str, Any],
    data_artifact_seed_summaries: list[dict[str, Any]],
    replay_keys: Mapping[str, list[str]],
    diagnostics: Iterable[Mapping[str, Any]],
    retry_diagnostics: Iterable[Mapping[str, Any]],
    skip_diagnostics: Iterable[Mapping[str, Any]],
    source_retrieval_report: Mapping[str, Any],
) -> dict[str, Any]:
    write_plan = pipeline_result.write_plan
    batch_digests = _batch_digest_rollups(write_plan)
    diagnostic_rollup = _diagnostic_digest_rollup(diagnostics)
    retry_rollup = _diagnostic_digest_rollup(retry_diagnostics)
    skip_rollup = _diagnostic_digest_rollup(skip_diagnostics)
    source_retrieval_rollup = _source_retrieval_replay_rollup(
        source_retrieval_report
    )
    input_fingerprint_source = {
        "report_kind": BACKFILL_REPORT_KIND,
        "dry_run": True,
        "input_summary": _json_safe_mapping(input_summary),
        "data_artifact_seed_digest": _stable_sequence_digest(
            data_artifact_seed_summaries
        ),
        "table_row_counts": dict(write_plan.audit_summary.planned_row_counts),
        "table_modes": dict(sorted(write_plan.table_modes.items())),
        "table_batch_digests": batch_digests,
        "diagnostic_digest": diagnostic_rollup["digest"],
        "source_retrieval_replay_key_digest": source_retrieval_rollup[
            "replay_key_digest"
        ],
    }
    return {
        "schema_version": "physics-ingest-backfill-audit-replay.v1",
        "input_fingerprint_sha256": _stable_row_hash(input_fingerprint_source),
        "input_fingerprint_source": input_fingerprint_source,
        "table_batch_digests": batch_digests,
        "replay_key_rollup": _replay_key_rollup(replay_keys),
        "diagnostic_digest": diagnostic_rollup,
        "retry_digest": retry_rollup,
        "skip_digest": skip_rollup,
        "source_retrieval_replay": source_retrieval_rollup,
    }


def _batch_digest_rollups(write_plan: Any) -> dict[str, dict[str, Any]]:
    rollups: dict[str, dict[str, Any]] = {}
    for batch in write_plan.batches:
        row_hashes = [_stable_row_hash(row) for row in batch.rows]
        conflict_keys = tuple(batch.conflict_keys)
        conflict_identities = [
            _row_conflict_identity(row, conflict_keys, index)
            for index, row in enumerate(batch.rows)
        ]
        conflict_identity_hashes = [
            _stable_row_hash(identity) for identity in conflict_identities
        ]
        duplicate_conflict_identity_count = len(conflict_identity_hashes) - len(
            set(conflict_identity_hashes)
        )
        rollups[batch.table] = {
            "mode": write_plan.mode_for(batch.table),
            "row_count": batch.row_count,
            "conflict_keys": list(conflict_keys),
            "row_hash_digest": _stable_sequence_digest(row_hashes),
            "conflict_identity_digest": _stable_sequence_digest(
                conflict_identities
            ),
            "batch_digest": _stable_row_hash(
                {
                    "table": batch.table,
                    "mode": write_plan.mode_for(batch.table),
                    "row_count": batch.row_count,
                    "conflict_keys": list(conflict_keys),
                    "row_hash_digest": _stable_sequence_digest(row_hashes),
                    "conflict_identity_digest": _stable_sequence_digest(
                        conflict_identities
                    ),
                }
            ),
            "missing_conflict_key_row_count": sum(
                1
                for row in batch.rows
                if conflict_keys
                and any(row.get(key) in (None, "") for key in conflict_keys)
            ),
            "duplicate_conflict_identity_count": duplicate_conflict_identity_count,
        }
    return rollups


def _row_conflict_identity(
    row: Mapping[str, Any],
    conflict_keys: tuple[str, ...],
    index: int,
) -> dict[str, Any]:
    if conflict_keys and all(row.get(key) not in (None, "") for key in conflict_keys):
        return {
            "kind": "conflict_keys",
            "keys": {key: _json_safe_value(row.get(key)) for key in conflict_keys},
        }
    return {
        "kind": "row_hash",
        "row_index": index,
        "sha256": _stable_row_hash(row),
    }


def _replay_key_rollup(
    replay_keys: Mapping[str, list[str]],
) -> dict[str, dict[str, Any]]:
    return {
        table: {
            "count": len(keys),
            "digest": _stable_sequence_digest(keys),
        }
        for table, keys in sorted(replay_keys.items())
    }


def _diagnostic_digest_rollup(
    diagnostics: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [dict(row) for row in diagnostics]
    return {
        "count": len(rows),
        "digest": _stable_sequence_digest(rows),
        "summary": _diagnostic_summary(rows),
    }


def _source_retrieval_replay_rollup(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    replay_keys = [str(key) for key in report.get("replay_keys", ())]
    steps = [
        {
            "step_index": step.get("step_index", 0),
            "job_id": step.get("job_id", ""),
            "endpoint_id": step.get("endpoint_id", ""),
            "snapshot_key": step.get("snapshot_key", ""),
            "replay_key": step.get("replay_key", ""),
        }
        for step in report.get("steps", ())
        if isinstance(step, Mapping)
    ]
    return {
        "manifest_version": str(report.get("manifest_version") or ""),
        "snapshot_key_prefix": str(report.get("snapshot_key_prefix") or ""),
        "step_count": int(report.get("step_count") or 0),
        "diagnostic_count": int(report.get("diagnostic_count") or 0),
        "replay_key_count": len(replay_keys),
        "replay_key_digest": _stable_sequence_digest(replay_keys),
        "step_replay_digest": _stable_sequence_digest(steps),
    }


def _stable_sequence_digest(values: Iterable[Any]) -> str:
    return _stable_row_hash({"items": list(values)})


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _stable_row_hash(row: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_safe_value(row),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _diagnostics_by_severity(
    diagnostics: Iterable[Mapping[str, Any]],
    *,
    severity: str,
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in diagnostics
        if str(row.get("severity") or "") == severity
    ]


def _diagnostic_summary(diagnostics: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    by_severity = Counter(str(row.get("severity") or "info") for row in diagnostics)
    by_reason = Counter(str(row.get("reason") or "") for row in diagnostics)
    return {
        "by_severity": dict(sorted(by_severity.items())),
        "by_reason": dict(sorted(by_reason.items())),
    }


def _assert_json_serializable(report: Mapping[str, Any]) -> None:
    try:
        json.dumps(report, sort_keys=True, ensure_ascii=True)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise ValueError("backfill report must be JSON serializable") from exc
