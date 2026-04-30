"""Shared side-effect-free helpers for physics source adapter manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence


JSONDict = dict[str, Any]


@dataclass(frozen=True)
class SourceAdapterBundle:
    """Rows emitted by a source adapter before DB insertion."""

    snapshot_row: Mapping[str, Any]
    candidate_rows: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    data_artifact_seeds: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_row": dict(self.snapshot_row),
            "candidate_rows": [dict(row) for row in self.candidate_rows],
            "data_artifact_seeds": [dict(seed) for seed in self.data_artifact_seeds],
        }


def build_snapshot_row(
    *,
    source_system: str,
    source_version: str,
    source_uri: str,
    adapter_name: str,
    adapter_version: str,
    payload: Mapping[str, Any],
    license_expression: str,
    provenance_summary: str,
    retrieved_at: str | None = None,
) -> JSONDict:
    row: JSONDict = {
        "source_system": source_system,
        "source_version": source_version,
        "source_uri": source_uri,
        "adapter_name": adapter_name,
        "adapter_version": adapter_version,
        "license_expression": license_expression,
        "provenance_summary": provenance_summary,
        "payload_sha256": stable_payload_sha256(payload),
        "payload": jsonable(payload),
    }
    if retrieved_at:
        row["retrieved_at"] = retrieved_at
    return row


def stable_payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_source_id(prefix: str, payload: Any, *, length: int = 16) -> str:
    return f"{prefix}:{stable_payload_sha256(payload)[:length]}"


def first_text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        text = stringify(value)
        if text:
            return text
    return ""


def first_mapping(row: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    for key in keys:
        value = row.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [stringify(item) for item in value if stringify(item)]
    return [stringify(value)] if stringify(value) else []


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("@id", "id", "value", "@value", "uri", "name", "label"):
            if key in value:
                return stringify(value[key])
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return str(value).strip()


def slug(text: str, *, default: str = "unknown") -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", text.strip().casefold()).strip("-")
    return normalized or default


def jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [jsonable(item) for item in value]
    return value


def normalize_raw_records(
    raw_records: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(dict(record) for record in raw_records)
