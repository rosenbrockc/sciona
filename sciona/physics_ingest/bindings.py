"""Side-effect-free artifact/version binding resolver for physics publication."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from sciona.physics_ingest.publication import ArtifactBinding


_KEY_FIELDS = (
    "artifact_key",
    "local_artifact_key",
    "fqdn",
    "registry_name",
    "atom_name",
)


@dataclass(frozen=True)
class BindingDiagnostic:
    """One non-fatal artifact/version binding resolution diagnostic."""

    reason: str
    severity: str = "error"
    row_index: int | None = None
    key_name: str = ""
    key_value: str = ""
    artifact_key: str = ""
    atom_name: str = ""
    artifact_ids: tuple[str, ...] = ()
    version_ids: tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class BindingResolutionResult:
    """Resolved publication bindings keyed for manifest/source-bundle lookups."""

    bindings: Mapping[str, ArtifactBinding]
    diagnostics: tuple[BindingDiagnostic, ...] = ()

    @property
    def error_rows(self) -> tuple[BindingDiagnostic, ...]:
        return tuple(row for row in self.diagnostics if row.severity == "error")

    def to_publication_bindings(self) -> dict[str, ArtifactBinding]:
        """Return a concrete mapping accepted by publication loaders."""

        return dict(self.bindings)


def resolve_publication_artifact_bindings(
    binding_rows: Iterable[Mapping[str, Any]],
    artifact_rows: Iterable[Mapping[str, Any]],
    version_rows: Iterable[Mapping[str, Any]],
) -> BindingResolutionResult:
    """Resolve artifact/version IDs for manifest or source-bundle binding rows.

    All inputs are in-memory row mappings. The resolver performs no database IO.
    ``binding_rows`` are rows from atom manifests/source bundles containing any
    of ``artifact_key``, ``local_artifact_key``, ``fqdn``, ``registry_name``, or
    ``atom_name``. Successful rows produce ``ArtifactBinding`` values under every
    key present on the input row, so the result can be passed directly to
    ``publication.load_symbolic_publication_manifest``.
    """

    artifacts = tuple(artifact_rows)
    versions = tuple(version_rows)
    artifact_index = _index_artifacts(artifacts)
    versions_by_artifact = _versions_by_artifact(versions)

    bindings: dict[str, ArtifactBinding] = {}
    diagnostics: list[BindingDiagnostic] = []

    for row_index, row in enumerate(binding_rows):
        row_keys = _row_keys(row)
        if not row_keys:
            diagnostics.append(
                _diagnostic(
                    "missing_binding_key",
                    row,
                    row_index,
                    detail=(
                        "row has no artifact_key/local_artifact_key/fqdn/"
                        "registry_name/atom_name"
                    ),
                )
            )
            continue

        artifact_ids = _candidate_artifact_ids(row_keys, artifact_index)
        if not artifact_ids:
            diagnostics.append(
                _diagnostic(
                    "missing_artifact",
                    row,
                    row_index,
                    key_name=row_keys[0][0],
                    key_value=row_keys[0][1],
                )
            )
            continue
        if len(artifact_ids) > 1:
            diagnostics.append(
                _diagnostic(
                    "ambiguous_artifact",
                    row,
                    row_index,
                    key_name=row_keys[0][0],
                    key_value=row_keys[0][1],
                    artifact_ids=tuple(sorted(artifact_ids)),
                )
            )
            continue

        artifact_id = next(iter(artifact_ids))
        version_id, version_diagnostic = _resolve_version_id(
            row,
            row_index,
            artifact_id,
            versions_by_artifact.get(artifact_id, ()),
        )
        if version_diagnostic is not None:
            diagnostics.append(version_diagnostic)
            continue

        try:
            binding = ArtifactBinding.model_validate(
                {"artifact_id": artifact_id, "version_id": version_id}
            )
        except ValueError as exc:
            diagnostics.append(
                _diagnostic(
                    "invalid_binding",
                    row,
                    row_index,
                    artifact_ids=(artifact_id,),
                    version_ids=(version_id,),
                    detail=str(exc),
                )
            )
            continue

        for _key_name, key_value in row_keys:
            existing = bindings.get(key_value)
            if existing is not None and existing != binding:
                diagnostics.append(
                    _diagnostic(
                        "ambiguous_binding_key",
                        row,
                        row_index,
                        key_value=key_value,
                        artifact_ids=tuple(
                            sorted({existing.artifact_id, binding.artifact_id})
                        ),
                        version_ids=tuple(
                            sorted({existing.version_id, binding.version_id})
                        ),
                    )
                )
                continue
            bindings[key_value] = binding

    return BindingResolutionResult(bindings=bindings, diagnostics=tuple(diagnostics))


def _index_artifacts(
    artifact_rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], set[str]]:
    index: dict[tuple[str, str], set[str]] = {}
    for row in artifact_rows:
        artifact_id = _text(row, "artifact_id", "id")
        if not artifact_id:
            continue
        for key_name, key_value in _row_keys(row):
            index.setdefault((key_name, key_value), set()).add(artifact_id)
    return index


def _versions_by_artifact(
    version_rows: Iterable[Mapping[str, Any]],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in version_rows:
        artifact_id = _text(row, "artifact_id")
        if artifact_id:
            grouped.setdefault(artifact_id, []).append(row)
    return {artifact_id: tuple(rows) for artifact_id, rows in grouped.items()}


def _candidate_artifact_ids(
    row_keys: tuple[tuple[str, str], ...],
    artifact_index: Mapping[tuple[str, str], set[str]],
) -> set[str]:
    matches: set[str] = set()
    for key in row_keys:
        matches.update(artifact_index.get(key, set()))
    return matches


def _resolve_version_id(
    row: Mapping[str, Any],
    row_index: int,
    artifact_id: str,
    versions: tuple[Mapping[str, Any], ...],
) -> tuple[str, BindingDiagnostic | None]:
    if not versions:
        return "", _diagnostic(
            "missing_version",
            row,
            row_index,
            artifact_ids=(artifact_id,),
        )

    explicit_version_id = _text(row, "version_id")
    if explicit_version_id:
        matches = tuple(
            version
            for version in versions
            if _text(version, "version_id", "id") == explicit_version_id
        )
        if len(matches) == 1:
            return explicit_version_id, None
        return "", _diagnostic(
            "missing_version",
            row,
            row_index,
            artifact_ids=(artifact_id,),
            version_ids=(explicit_version_id,),
            detail="explicit version_id was not found for the resolved artifact",
        )

    selected = _select_versions(row, versions, "semver")
    if not selected:
        selected = _select_versions(row, versions, "content_hash")
    if not selected:
        selected = tuple(
            version for version in versions if _bool(version.get("is_latest"))
        )
    if not selected and len(versions) == 1:
        selected = versions

    if len(selected) == 1:
        version_id = _text(selected[0], "version_id", "id")
        if version_id:
            return version_id, None

    reason = "missing_version" if not versions else "ambiguous_version"
    candidates = selected or versions
    return "", _diagnostic(
        reason,
        row,
        row_index,
        artifact_ids=(artifact_id,),
        version_ids=tuple(
            sorted(_text(version, "version_id", "id") for version in candidates)
        ),
    )


def _select_versions(
    row: Mapping[str, Any],
    versions: tuple[Mapping[str, Any], ...],
    field_name: str,
) -> tuple[Mapping[str, Any], ...]:
    selector = _text(row, field_name)
    if not selector:
        return ()
    return tuple(version for version in versions if _text(version, field_name) == selector)


def _row_keys(row: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for key_name in _KEY_FIELDS:
        key_value = _text(row, key_name)
        key = (key_name, key_value)
        if key_value and key not in seen:
            seen.add(key)
            keys.append(key)
    return tuple(keys)


def _diagnostic(
    reason: str,
    row: Mapping[str, Any],
    row_index: int,
    *,
    key_name: str = "",
    key_value: str = "",
    artifact_ids: tuple[str, ...] = (),
    version_ids: tuple[str, ...] = (),
    detail: str = "",
) -> BindingDiagnostic:
    return BindingDiagnostic(
        reason=reason,
        row_index=row_index,
        key_name=key_name,
        key_value=key_value,
        artifact_key=_text(row, "artifact_key", "local_artifact_key"),
        atom_name=_text(row, "atom_name"),
        artifact_ids=tuple(item for item in artifact_ids if item),
        version_ids=tuple(item for item in version_ids if item),
        detail=detail,
    )


def _text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)
