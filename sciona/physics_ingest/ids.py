"""Deterministic identifiers for side-effect-free physics source ingestion."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import re
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

SOURCE_SNAPSHOT_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "sciona.physics_ingest.source_snapshot",
)
SOURCE_CANDIDATE_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "sciona.physics_ingest.source_candidate",
)


class DeterministicIdError(ValueError):
    """Raised when source rows cannot be assigned unambiguous deterministic IDs."""


def stable_payload_sha256(payload: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible source payloads."""

    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DeterministicIdError("payload must be JSON serializable") from exc
    return hashlib.sha256(encoded).hexdigest()


def source_snapshot_id(snapshot_row: Mapping[str, Any]) -> str:
    """Derive a UUIDv5 snapshot ID from stable source snapshot identity fields."""

    payload_sha256 = _snapshot_payload_sha256(snapshot_row)
    parts = (
        _text(snapshot_row.get("source_system")),
        _text(snapshot_row.get("source_version")),
        _text(snapshot_row.get("source_uri")),
        _text(snapshot_row.get("adapter_name")),
        _text(snapshot_row.get("adapter_version")),
        payload_sha256,
    )
    return str(uuid5(SOURCE_SNAPSHOT_NAMESPACE, "\x1f".join(parts)))


def source_candidate_id(snapshot_id: str, source_candidate_id: str) -> str:
    """Derive a UUIDv5 candidate ID scoped to a deterministic source snapshot ID."""

    valid_snapshot_id = _validate_uuid(snapshot_id, "snapshot_id")
    source_id = _text(source_candidate_id)
    if not source_id:
        raise DeterministicIdError("source_candidate_id is required")
    return str(uuid5(SOURCE_CANDIDATE_NAMESPACE, f"{valid_snapshot_id}\x1f{source_id}"))


def build_snapshot_id_bindings(source_bundles: Iterable[Any]) -> dict[str, str]:
    """Build orchestration snapshot bindings for source bundles without DB calls."""

    bindings: dict[str, str] = {}
    for index, bundle in enumerate(source_bundles):
        snapshot_row = _bundle_value(bundle, "snapshot_row")
        if not isinstance(snapshot_row, Mapping):
            raise DeterministicIdError(f"source bundle {index} has no snapshot_row mapping")
        snapshot_id = source_snapshot_id(snapshot_row)
        for key in _snapshot_binding_keys(bundle, snapshot_row, index):
            existing = bindings.get(key)
            if existing is not None and existing != snapshot_id:
                raise DeterministicIdError(
                    f"snapshot binding key {key!r} resolves to multiple snapshot IDs"
                )
            bindings[key] = snapshot_id
    return bindings


def attach_deterministic_candidate_ids(
    candidate_rows: Iterable[Mapping[str, Any]],
    snapshot_id: str,
    *,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Return copied candidate rows with deterministic ``candidate_id`` values."""

    valid_snapshot_id = _validate_uuid(snapshot_id, "snapshot_id")
    attached: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for ordinal, row in enumerate(candidate_rows):
        if not isinstance(row, Mapping):
            raise DeterministicIdError(f"candidate row {ordinal} is not a mapping")

        copied = dict(row)
        planned_id = source_candidate_id(
            valid_snapshot_id,
            _text(copied.get("source_candidate_id")),
        )
        existing = copied.get("candidate_id")
        if existing not in (None, ""):
            existing_id = _validate_uuid(str(existing), "candidate_id")
            if existing_id != planned_id and not overwrite:
                raise DeterministicIdError(
                    "candidate row already has a different candidate_id; "
                    "pass overwrite=True"
                )

        copied["candidate_id"] = planned_id
        if planned_id in seen:
            raise DeterministicIdError(
                "deterministic candidate_id collision for source_candidate_id "
                f"{copied.get('source_candidate_id')!r}"
            )
        seen[planned_id] = copied
        attached.append(copied)
    return attached


def plan_source_bundle_ids(
    source_bundles: Iterable[Any],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Plan snapshot bindings and candidate IDs for source bundle orchestration."""

    planned_bundles: list[dict[str, Any]] = []
    bindings: dict[str, str] = {}
    for index, bundle in enumerate(source_bundles):
        snapshot_row = _bundle_value(bundle, "snapshot_row")
        if not isinstance(snapshot_row, Mapping):
            raise DeterministicIdError(f"source bundle {index} has no snapshot_row mapping")

        snapshot_id = source_snapshot_id(snapshot_row)
        for key in _snapshot_binding_keys(bundle, snapshot_row, index):
            existing = bindings.get(key)
            if existing is not None and existing != snapshot_id:
                raise DeterministicIdError(
                    f"snapshot binding key {key!r} resolves to multiple snapshot IDs"
                )
            bindings[key] = snapshot_id

        candidate_rows = tuple(_bundle_value(bundle, "candidate_rows") or ())
        data_artifact_seeds = tuple(_bundle_value(bundle, "data_artifact_seeds") or ())
        planned_bundle = _copy_bundle_as_mapping(bundle)
        planned_snapshot = dict(snapshot_row)
        planned_snapshot["snapshot_id"] = snapshot_id
        planned_bundle["snapshot_row"] = planned_snapshot
        planned_bundle["candidate_rows"] = attach_deterministic_candidate_ids(
            candidate_rows,
            snapshot_id,
        )
        planned_bundle["data_artifact_seeds"] = [
            dict(seed) if isinstance(seed, Mapping) else seed
            for seed in data_artifact_seeds
        ]
        planned_bundles.append(planned_bundle)
    return bindings, planned_bundles


def _snapshot_payload_sha256(snapshot_row: Mapping[str, Any]) -> str:
    payload_sha256 = _text(snapshot_row.get("payload_sha256"))
    if payload_sha256:
        if not _SHA256_RE.fullmatch(payload_sha256):
            raise DeterministicIdError(
                "payload_sha256 must be a lowercase SHA-256 hex digest"
            )
        return payload_sha256
    if "payload" not in snapshot_row:
        raise DeterministicIdError("snapshot_row requires payload_sha256 or payload")
    return stable_payload_sha256(snapshot_row["payload"])


def _bundle_value(bundle: Any, key: str) -> Any:
    if isinstance(bundle, Mapping):
        return bundle.get(key)
    value = getattr(bundle, key, None)
    return value() if callable(value) and key == "candidate_rows" else value


def _copy_bundle_as_mapping(bundle: Any) -> dict[str, Any]:
    if isinstance(bundle, Mapping):
        return dict(bundle)
    copied: dict[str, Any] = {}
    for key in (
        "bundle_key",
        "key",
        "name",
        "snapshot_row",
        "candidate_rows",
        "data_artifact_seeds",
    ):
        if hasattr(bundle, key):
            copied[key] = _bundle_value(bundle, key)
    return copied


def _snapshot_binding_keys(
    bundle: Any,
    snapshot_row: Mapping[str, Any],
    index: int,
) -> tuple[str, ...]:
    keys: list[str] = []
    if isinstance(bundle, Mapping):
        for key_name in ("bundle_key", "key", "name"):
            _append_key(keys, bundle.get(key_name))
    else:
        for key_name in ("bundle_key", "key", "name"):
            _append_key(keys, getattr(bundle, key_name, None))
    for key_name in ("source_system", "adapter_name", "source_uri"):
        _append_key(keys, snapshot_row.get(key_name))
    if not keys:
        keys.append(f"source_bundle:{index}")
    return tuple(keys)


def _append_key(keys: list[str], value: Any) -> None:
    text = _text(value)
    if text and text not in keys:
        keys.append(text)


def _validate_uuid(value: str, field_name: str) -> str:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError) as exc:
        raise DeterministicIdError(f"{field_name} must be a UUID") from exc


def _text(value: Any) -> str:
    return "" if value is None else str(value)
