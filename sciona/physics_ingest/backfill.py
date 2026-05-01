"""Side-effect-free bulk backfill planning for physics ingestion."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
import hashlib
import json
from typing import Any

from sciona.physics_ingest.pipeline import run_physics_publication_pipeline
from sciona.physics_ingest.publication import ArtifactBinding
from sciona.physics_ingest.write_plan import WriteMode


BACKFILL_REPORT_KIND = "physics_ingest_bulk_backfill_plan"


def build_physics_ingest_backfill_report(
    *,
    source_bundles: Iterable[Any] = (),
    publication_manifests: Iterable[Mapping[str, Any]] = (),
    pdg_publication_rows: Any | None = None,
    review_diagnostics: Iterable[Mapping[str, Any]] = (),
    normalization_diagnostics: Iterable[Mapping[str, Any]] = (),
    artifact_bindings: Mapping[str, Mapping[str, Any] | ArtifactBinding] | None = None,
    table_modes: Mapping[str, WriteMode] | None = None,
    include_rows: bool = False,
) -> dict[str, Any]:
    """Build a deterministic dry-run report for a bulk physics backfill.

    The helper performs no database or network IO. It accepts ordinary source
    bundles, symbolic publication manifests, optional PDG publication rows, and
    caller-supplied review or normalization diagnostics. All rows are routed
    through the existing publication pipeline and write-plan ordering so the
    report mirrors the eventual writer batches without touching a client.
    """

    source_bundle_list = tuple(source_bundles)
    publication_manifest_list = tuple(publication_manifests)
    pdg_rows, pdg_diagnostics = _normalize_pdg_publication_rows(pdg_publication_rows)
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
    external_diagnostics = (
        *review_diagnostic_rows,
        *normalization_diagnostic_rows,
        *pdg_diagnostic_rows,
    )

    pipeline_result = run_physics_publication_pipeline(
        source_bundles=source_bundle_list,
        publication_manifests=publication_manifest_list,
        additional_insert_rows=pdg_rows,
        additional_diagnostics=external_diagnostics,
        artifact_bindings=artifact_bindings,
        table_modes=table_modes,
        dry_run=True,
    )
    insert_rows = pipeline_result.write_plan.to_insert_rows()
    diagnostics = tuple(pipeline_result.diagnostics)
    retry_diagnostics = _diagnostics_by_severity(diagnostics, severity="error")
    skip_diagnostics = _diagnostics_by_severity(diagnostics, severity="skipped")
    has_errors = bool(retry_diagnostics)

    report: dict[str, Any] = {
        "report_kind": BACKFILL_REPORT_KIND,
        "dry_run": True,
        "ok": not has_errors,
        "input_summary": {
            "source_bundle_count": len(source_bundle_list),
            "publication_manifest_count": len(publication_manifest_list),
            "pdg_table_count": len(pdg_rows),
            "pdg_row_count": sum(len(rows) for rows in pdg_rows.values()),
            "review_diagnostic_count": len(review_diagnostic_rows),
            "normalization_diagnostic_count": len(normalization_diagnostic_rows),
        },
        "source_family_counts": _source_family_counts(
            source_bundle_list,
            publication_manifest_list,
            pdg_rows,
        ),
        "table_row_counts": dict(pipeline_result.write_plan.audit_summary.planned_row_counts),
        "dry_run_write_plan": _write_plan_summary(pipeline_result),
        "replay_keys": _replay_keys(pipeline_result.write_plan.to_dict()["batches"]),
        "retry_diagnostics": retry_diagnostics,
        "skip_diagnostics": skip_diagnostics,
        "external_diagnostics": {
            "review": [dict(row) for row in review_diagnostic_rows],
            "normalization": [dict(row) for row in normalization_diagnostic_rows],
            "pdg_cdg_publication": [dict(row) for row in pdg_diagnostic_rows],
        },
        "diagnostic_summary": _diagnostic_summary(diagnostics),
    }
    if include_rows:
        report["insert_rows_by_table"] = insert_rows

    _assert_json_serializable(report)
    return report


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


def _copy_rows_by_table(
    rows_by_table: Mapping[str, Iterable[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    copied: dict[str, list[dict[str, Any]]] = {}
    for table, rows in rows_by_table.items():
        copied[str(table)] = [dict(row) for row in rows]
    return copied


def _normalize_diagnostics(
    diagnostics: Iterable[Mapping[str, Any]],
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
        for row in diagnostics
    )


def _source_family_counts(
    source_bundles: tuple[Any, ...],
    publication_manifests: tuple[Mapping[str, Any], ...],
    pdg_rows: Mapping[str, Iterable[Mapping[str, Any]]],
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
    combined = bundle_counts + manifest_counts + pdg_counts
    return {
        "source_bundles": dict(sorted(bundle_counts.items())),
        "publication_manifests": dict(sorted(manifest_counts.items())),
        "pdg_publication_rows": dict(sorted(pdg_counts.items())),
        "combined": dict(sorted(combined.items())),
    }


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


def _stable_row_hash(row: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        row,
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
