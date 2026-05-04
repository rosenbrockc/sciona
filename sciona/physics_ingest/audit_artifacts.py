"""Side-effect-free storage planning for backfill audit artifacts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any

from sciona.physics_ingest.write_plan import (
    PublicationWritePlan,
    WriteMode,
    build_publication_write_plan,
)


BACKFILL_AUDIT_ARTIFACTS_TABLE = "physics_ingest_audit_artifacts"
BACKFILL_AUDIT_ARTIFACT_REQUIRED_FIELDS = (
    "artifact_key",
    "name",
    "source_section",
    "payload_sha256",
    "content_type",
)


@dataclass(frozen=True)
class BackfillAuditArtifactWritePlanRows:
    """Rows and optional inert write plan for persisted backfill audit artifacts."""

    insert_rows_by_table: Mapping[str, tuple[dict[str, Any], ...]]
    diagnostics: tuple[dict[str, Any], ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)
    write_plan: PublicationWritePlan | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "insert_rows_by_table",
            {
                str(table): tuple(_json_safe_mapping(row) for row in rows)
                for table, rows in self.insert_rows_by_table.items()
            },
        )
        object.__setattr__(
            self,
            "diagnostics",
            tuple(_json_safe_mapping(row) for row in self.diagnostics),
        )
        object.__setattr__(
            self,
            "summary",
            _json_safe_mapping(self.summary) if self.summary else {},
        )

    def to_insert_rows(self) -> dict[str, list[dict[str, Any]]]:
        return {
            table: [dict(row) for row in rows]
            for table, rows in self.insert_rows_by_table.items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "insert_rows": self.to_insert_rows(),
            "diagnostics": list(self.diagnostics),
            "summary": dict(self.summary),
            "write_plan": None
            if self.write_plan is None
            else self.write_plan.to_dict(),
        }


def build_backfill_audit_artifact_write_plan_rows(
    backfill_report_or_manifest_rows: Mapping[str, Any] | Iterable[Any],
    *,
    include_write_plan: bool = False,
    table_name: str = BACKFILL_AUDIT_ARTIFACTS_TABLE,
    table_modes: Mapping[str, WriteMode] | None = None,
) -> BackfillAuditArtifactWritePlanRows:
    """Project backfill audit artifact manifests into inert storage rows.

    The helper performs no client construction, filesystem IO, network IO, or
    database writes. It accepts either a backfill report containing
    ``audit_artifact_manifests`` or an iterable of manifest rows.
    """

    manifest_rows, diagnostics = _extract_manifest_rows(
        backfill_report_or_manifest_rows
    )
    storage_rows, row_diagnostics = _build_storage_rows(
        manifest_rows,
        table_name=table_name,
    )
    diagnostics = (*diagnostics, *row_diagnostics)
    rows_by_table = {table_name: tuple(storage_rows)} if storage_rows else {}
    write_plan = build_publication_write_plan(
        rows_by_table,
        table_modes=table_modes,
    )
    summary = _build_summary(
        table_name=table_name,
        rows=storage_rows,
        diagnostics=diagnostics,
        write_plan=write_plan,
    )
    return BackfillAuditArtifactWritePlanRows(
        insert_rows_by_table=rows_by_table,
        diagnostics=diagnostics,
        summary=summary,
        write_plan=write_plan if include_write_plan else None,
    )


def _extract_manifest_rows(
    value: Mapping[str, Any] | Iterable[Any],
) -> tuple[tuple[Any, ...], tuple[dict[str, Any], ...]]:
    if isinstance(value, Mapping):
        if "audit_artifact_manifests" in value:
            rows = value.get("audit_artifact_manifests")
            if _is_row_iterable(rows):
                return tuple(rows), ()
            return (
                (),
                (
                    _diagnostic(
                    reason="invalid_audit_artifact_manifests",
                    table_name=BACKFILL_AUDIT_ARTIFACTS_TABLE,
                    detail="audit_artifact_manifests must be an iterable of rows",
                ),
                ),
            )
        if any(field in value for field in BACKFILL_AUDIT_ARTIFACT_REQUIRED_FIELDS):
            return (value,), ()
        return (
            (),
            (
                _diagnostic(
                    reason="missing_audit_artifact_manifests",
                    table_name=BACKFILL_AUDIT_ARTIFACTS_TABLE,
                    detail="backfill report does not include audit_artifact_manifests",
                    severity="warning",
                ),
            ),
        )
    if _is_row_iterable(value):
        return tuple(value), ()
    return (
        (),
        (
            _diagnostic(
                reason="invalid_manifest_rows",
                table_name=BACKFILL_AUDIT_ARTIFACTS_TABLE,
                detail="manifest rows must be supplied as an iterable",
            ),
        ),
    )


def _build_storage_rows(
    manifest_rows: Iterable[Any],
    *,
    table_name: str,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for index, row in enumerate(manifest_rows):
        if not isinstance(row, Mapping):
            diagnostics.append(
                _diagnostic(
                    reason="invalid_manifest_row_shape",
                    table_name=table_name,
                    detail="manifest row must be a mapping",
                    row_index=index,
                )
            )
            continue
        storage_row, row_diagnostics = _storage_row_from_manifest(
            row,
            row_index=index,
            table_name=table_name,
        )
        diagnostics.extend(row_diagnostics)
        if storage_row is not None:
            rows.append(storage_row)
    return (
        tuple(sorted(rows, key=_storage_row_sort_key)),
        tuple(diagnostics),
    )


def _storage_row_from_manifest(
    row: Mapping[str, Any],
    *,
    row_index: int,
    table_name: str,
) -> tuple[dict[str, Any] | None, tuple[dict[str, Any], ...]]:
    safe_row = _json_safe_mapping(row)
    diagnostics: list[dict[str, Any]] = []
    if not safe_row.get("name") and safe_row.get("artifact_name"):
        safe_row["name"] = safe_row["artifact_name"]
        diagnostics.append(
            _diagnostic(
                reason="filled_missing_name_from_artifact_name",
                table_name=table_name,
                detail="manifest row was missing name but included artifact_name",
                severity="warning",
                row_index=row_index,
            )
        )

    missing_fields = [
        field_name
        for field_name in BACKFILL_AUDIT_ARTIFACT_REQUIRED_FIELDS
        if not safe_row.get(field_name)
    ]
    for field_name in missing_fields:
        diagnostics.append(
            _diagnostic(
                reason="missing_required_manifest_field",
                table_name=table_name,
                detail=f"manifest row is missing required field {field_name}",
                field=field_name,
                row_index=row_index,
            )
        )
    if missing_fields:
        return None, tuple(diagnostics)

    payload_sha256 = str(safe_row["payload_sha256"])
    if not _is_sha256_hex(payload_sha256):
        diagnostics.append(
            _diagnostic(
                reason="invalid_payload_sha256",
                table_name=table_name,
                detail="payload_sha256 must be a 64-character lowercase hex digest",
                field="payload_sha256",
                row_index=row_index,
            )
        )
        return None, tuple(diagnostics)

    storage_row = {
        **safe_row,
        "artifact_key": str(safe_row["artifact_key"]),
        "name": str(safe_row["name"]),
        "source_section": str(safe_row["source_section"]),
        "payload_sha256": payload_sha256,
        "content_type": str(safe_row["content_type"]),
    }
    return _json_safe_mapping(storage_row), tuple(diagnostics)


def _build_summary(
    *,
    table_name: str,
    rows: Iterable[Mapping[str, Any]],
    diagnostics: Iterable[Mapping[str, Any]],
    write_plan: PublicationWritePlan,
) -> dict[str, Any]:
    row_list = tuple(_json_safe_mapping(row) for row in rows)
    diagnostic_rows = tuple(_json_safe_mapping(row) for row in diagnostics)
    digest_source = [
        {
            "artifact_key": row["artifact_key"],
            "content_type": row["content_type"],
            "payload_sha256": row["payload_sha256"],
            "source_section": row["source_section"],
        }
        for row in row_list
    ]
    return _json_safe_mapping(
        {
            "summary_kind": (
                "physics-ingest-backfill-audit-artifact-write-plan-rows-summary.v1"
            ),
            "table": table_name,
            "row_count": len(row_list),
            "has_payload_count": sum(1 for row in row_list if "payload" in row),
            "content_type_counts": dict(
                sorted(Counter(str(row["content_type"]) for row in row_list).items())
            ),
            "artifact_keys": [str(row["artifact_key"]) for row in row_list],
            "payload_sha256_digest": _stable_sequence_digest(digest_source),
            "row_digest_sha256": _stable_sequence_digest(row_list),
            "diagnostic_count": len(diagnostic_rows),
            "diagnostics_by_severity": _count_by_key(diagnostic_rows, "severity"),
            "diagnostics_by_reason": _count_by_key(diagnostic_rows, "reason"),
            "write_plan_table_order": list(write_plan.ordered_tables()),
            "write_plan_batch_count": write_plan.audit_summary.batch_count,
            "write_plan_total_row_count": write_plan.audit_summary.total_row_count,
        }
    )


def _storage_row_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("artifact_key") or ""),
        str(row.get("name") or ""),
        str(row.get("source_section") or ""),
        str(row.get("payload_sha256") or ""),
    )


def _is_row_iterable(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping))


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _diagnostic(
    *,
    reason: str,
    table_name: str,
    detail: str,
    severity: str = "error",
    row_index: int | None = None,
    field: str = "",
) -> dict[str, Any]:
    row = {
        "stage": "backfill_audit_artifact_write_plan",
        "table": table_name,
        "reason": reason,
        "severity": severity,
        "detail": detail,
        "field": field,
    }
    if row_index is not None:
        row["row_index"] = row_index
    return row


def _count_by_key(
    rows: Iterable[Mapping[str, Any]],
    key: str,
) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key) or "") for row in rows).items()))


def _stable_sequence_digest(values: Iterable[Any]) -> str:
    return _stable_json_sha256([_json_safe_value(value) for value in values])


def _stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        _json_safe_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _json_safe_value(item)
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
    }


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)
