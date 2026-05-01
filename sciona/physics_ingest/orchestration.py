"""Side-effect-free orchestration for physics ingestion publication bundles."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from sciona.physics_ingest.publication import (
    ArtifactBinding,
    PublicationDiagnostic,
    load_symbolic_publication_manifest,
)
from sciona.physics_ingest.staging import (
    attach_snapshot_id,
    validate_candidate_row,
    validate_snapshot_row,
)


SNAPSHOT_TABLE = "physics_ingest_snapshots"
CANDIDATE_TABLE = "physics_equation_candidates"
@dataclass(frozen=True)
class PublicationAuditSummary:
    """Compact counts for a side-effect-free publication orchestration run."""

    source_bundle_count: int = 0
    publication_manifest_count: int = 0
    input_row_counts: Mapping[str, int] = field(default_factory=dict)
    insert_row_counts: Mapping[str, int] = field(default_factory=dict)
    skipped_row_count: int = 0
    error_row_count: int = 0

    @property
    def has_errors(self) -> bool:
        return self.error_row_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_bundle_count": self.source_bundle_count,
            "publication_manifest_count": self.publication_manifest_count,
            "input_row_counts": dict(self.input_row_counts),
            "insert_row_counts": dict(self.insert_row_counts),
            "skipped_row_count": self.skipped_row_count,
            "error_row_count": self.error_row_count,
            "has_errors": self.has_errors,
        }


@dataclass(frozen=True)
class PublicationOrchestrationResult:
    """Validated insert rows and diagnostics; no database IO is performed."""

    insert_rows_by_table: Mapping[str, tuple[dict[str, Any], ...]]
    diagnostics: tuple[PublicationDiagnostic, ...]
    audit_summary: PublicationAuditSummary

    @property
    def skipped_rows(self) -> tuple[PublicationDiagnostic, ...]:
        return tuple(row for row in self.diagnostics if row.severity == "skipped")

    @property
    def error_rows(self) -> tuple[PublicationDiagnostic, ...]:
        return tuple(row for row in self.diagnostics if row.severity == "error")

    def to_insert_rows(self) -> dict[str, list[dict[str, Any]]]:
        return {
            table: [dict(row) for row in rows]
            for table, rows in self.insert_rows_by_table.items()
        }


def orchestrate_physics_publication(
    *,
    source_bundles: Iterable[Any] = (),
    publication_manifests: Iterable[Mapping[str, Any]] = (),
    artifact_bindings: Mapping[str, Mapping[str, Any] | ArtifactBinding] | None = None,
    snapshot_id_bindings: Mapping[str, str] | None = None,
) -> PublicationOrchestrationResult:
    """Validate source and symbolic publication rows for later DB insertion.

    ``source_bundles`` may be adapter dataclasses/objects with ``snapshot_row`` and
    ``candidate_rows`` attributes, or mappings with those keys. Candidate rows that
    do not already carry ``snapshot_id`` require an explicit ``snapshot_id_bindings``
    entry keyed by bundle key, source system, adapter name, or source URI.

    ``publication_manifests`` are delegated to
    :func:`load_symbolic_publication_manifest`, using the explicit
    ``artifact_bindings`` map for artifact/version resolution.
    """

    bindings = artifact_bindings or {}
    snapshot_bindings = snapshot_id_bindings or {}
    insert_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    diagnostics: list[PublicationDiagnostic] = []
    input_counts: dict[str, int] = defaultdict(int)
    source_count = 0
    manifest_count = 0

    for index, bundle in enumerate(source_bundles):
        source_count += 1
        _load_source_bundle(
            bundle,
            index=index,
            snapshot_id_bindings=snapshot_bindings,
            insert_rows=insert_rows,
            diagnostics=diagnostics,
            input_counts=input_counts,
        )

    for manifest in publication_manifests:
        manifest_count += 1
        publication_result = load_symbolic_publication_manifest(manifest, bindings)
        diagnostics.extend(publication_result.diagnostics)
        for table, rows in publication_result.to_insert_rows().items():
            input_counts[table] += _manifest_row_count(manifest, table)
            insert_rows[table].extend(rows)

    grouped_rows = {
        table: tuple(rows)
        for table, rows in sorted(insert_rows.items())
        if rows
    }
    summary = PublicationAuditSummary(
        source_bundle_count=source_count,
        publication_manifest_count=manifest_count,
        input_row_counts=dict(sorted(input_counts.items())),
        insert_row_counts={
            table: len(rows)
            for table, rows in sorted(grouped_rows.items())
        },
        skipped_row_count=sum(1 for row in diagnostics if row.severity == "skipped"),
        error_row_count=sum(1 for row in diagnostics if row.severity == "error"),
    )
    return PublicationOrchestrationResult(
        insert_rows_by_table=grouped_rows,
        diagnostics=tuple(diagnostics),
        audit_summary=summary,
    )


def _load_source_bundle(
    bundle: Any,
    *,
    index: int,
    snapshot_id_bindings: Mapping[str, str],
    insert_rows: dict[str, list[dict[str, Any]]],
    diagnostics: list[PublicationDiagnostic],
    input_counts: dict[str, int],
) -> None:
    snapshot_row = _bundle_value(bundle, "snapshot_row")
    candidate_rows = tuple(_bundle_value(bundle, "candidate_rows") or ())
    bundle_key = _bundle_key(bundle, snapshot_row, index)
    source_system = _text(snapshot_row, "source_system")

    if not isinstance(snapshot_row, Mapping):
        diagnostics.append(
            PublicationDiagnostic(
                table=SNAPSHOT_TABLE,
                reason="missing_snapshot_row",
                artifact_key=bundle_key,
                severity="error",
            )
        )
        return

    input_counts[SNAPSHOT_TABLE] += 1
    input_counts[CANDIDATE_TABLE] += len(candidate_rows)

    try:
        snapshot = validate_snapshot_row(snapshot_row)
    except ValueError as exc:
        diagnostics.append(
            PublicationDiagnostic(
                table=SNAPSHOT_TABLE,
                reason="validation_error",
                artifact_key=bundle_key,
                atom_name=source_system,
                severity="error",
                detail=str(exc),
            )
        )
        return

    insert_rows[SNAPSHOT_TABLE].append(snapshot.to_insert_dict())
    snapshot_id = _resolve_snapshot_id(bundle, snapshot_row, snapshot_id_bindings)
    rows_for_validation: Iterable[Mapping[str, Any]]
    if snapshot_id is None:
        rows_for_validation = candidate_rows
    else:
        try:
            rows_for_validation = attach_snapshot_id(candidate_rows, snapshot_id)
        except ValueError as exc:
            diagnostics.append(
                PublicationDiagnostic(
                    table=CANDIDATE_TABLE,
                    reason="snapshot_binding_error",
                    artifact_key=bundle_key,
                    atom_name=source_system,
                    severity="error",
                    detail=str(exc),
                )
            )
            return

    for ordinal, candidate_row in enumerate(rows_for_validation):
        if not isinstance(candidate_row, Mapping):
            diagnostics.append(
                PublicationDiagnostic(
                    table=CANDIDATE_TABLE,
                    reason="invalid_candidate_row",
                    artifact_key=bundle_key,
                    atom_name=source_system,
                    severity="error",
                    detail=f"candidate row {ordinal} is not a mapping",
                )
            )
            continue
        if candidate_row.get("snapshot_id") in (None, ""):
            diagnostics.append(
                PublicationDiagnostic(
                    table=CANDIDATE_TABLE,
                    reason="missing_snapshot_binding",
                    artifact_key=bundle_key,
                    atom_name=source_system,
                    detail=f"candidate row {ordinal} has no snapshot_id",
                )
            )
            continue
        try:
            candidate = validate_candidate_row(candidate_row)
        except ValueError as exc:
            diagnostics.append(
                PublicationDiagnostic(
                    table=CANDIDATE_TABLE,
                    reason="validation_error",
                    artifact_key=bundle_key,
                    atom_name=source_system,
                    severity="error",
                    detail=str(exc),
                )
            )
            continue
        insert_rows[CANDIDATE_TABLE].append(candidate.to_insert_dict())


def _bundle_value(bundle: Any, key: str) -> Any:
    if isinstance(bundle, Mapping):
        return bundle.get(key)
    value = getattr(bundle, key, None)
    return value() if callable(value) and key == "candidate_rows" else value


def _bundle_key(bundle: Any, snapshot_row: Any, index: int) -> str:
    if isinstance(bundle, Mapping):
        for key_name in ("bundle_key", "key", "name"):
            if bundle.get(key_name):
                return str(bundle[key_name])
    if isinstance(snapshot_row, Mapping):
        for key_name in ("source_system", "adapter_name", "source_uri"):
            if snapshot_row.get(key_name):
                return str(snapshot_row[key_name])
    return f"source_bundle:{index}"


def _resolve_snapshot_id(
    bundle: Any,
    snapshot_row: Mapping[str, Any],
    snapshot_id_bindings: Mapping[str, str],
) -> str | None:
    for key in _snapshot_binding_keys(bundle, snapshot_row):
        if key in snapshot_id_bindings:
            return snapshot_id_bindings[key]
    return None


def _snapshot_binding_keys(bundle: Any, snapshot_row: Mapping[str, Any]) -> tuple[str, ...]:
    keys: list[str] = []
    if isinstance(bundle, Mapping):
        for key_name in ("bundle_key", "key", "name"):
            _append_key(keys, bundle.get(key_name))
    for key_name in ("source_system", "adapter_name", "source_uri"):
        _append_key(keys, snapshot_row.get(key_name))
    return tuple(keys)


def _append_key(keys: list[str], value: Any) -> None:
    if value not in (None, "") and str(value) not in keys:
        keys.append(str(value))


def _manifest_row_count(manifest: Mapping[str, Any], table: str) -> int:
    rows = manifest.get(table, ())
    if isinstance(rows, (list, tuple)):
        return len(rows)
    return 0


def _text(row: Any, key: str) -> str:
    if not isinstance(row, Mapping):
        return ""
    value = row.get(key)
    return "" if value is None else str(value)
