"""QUDT dimension-vector adapter for physics ingestion.

QUDT encodes SI base-dimension exponents in compact vectors such as
``A0E0L1I0M1H0T-2D0``.  This module maps those vectors into Sciona's existing
``DimensionalSignature`` compact strings and builds deterministic raw
snapshot/candidate-compatible records for the Wave 0 ingestion schema.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
import json
import re
from typing import Any

from sciona.ghost.dimensions import DimensionalSignature


ADAPTER_NAME = "sciona.physics_ingest.sources.qudt"
ADAPTER_VERSION = "wave1-qudt-dim-resolution-v2"

_EXPONENT_TOKEN_RE = r"-?(?:\d+/\d+|\d+(?:\.\d+)?|\.\d+)"
_VECTOR_RE = re.compile(rf"([AELIMHTD])({_EXPONENT_TOKEN_RE})")
_DIMENSION_VECTOR_URI_MARKER = "/dimensionvector/"

# QUDT dimension-vector axes:
# A = amount of substance, E = electric current, L = length,
# I = luminous intensity, M = mass, H = thermodynamic temperature,
# T = time, D = dimensionless marker.
_QUDT_TO_SCIONA_FIELD = {
    "A": "N",
    "E": "I",
    "L": "L",
    "I": "J",
    "M": "M",
    "H": "Theta",
    "T": "T",
}


class QudtDimensionError(ValueError):
    """Raised when a QUDT dimension vector cannot be mapped safely."""


@dataclass(frozen=True)
class QudtDimensionMapping:
    """Parsed QUDT dimension vector and its Sciona representation."""

    qudt_vector: str
    qudt_exponents: dict[str, Fraction]
    sciona_signature: DimensionalSignature
    qudt_exponent_payloads: dict[str, str] = field(default_factory=dict)

    @property
    def compact(self) -> str:
        return self.sciona_signature.to_compact()

    def as_payload(self) -> dict[str, Any]:
        return {
            "qudt_dimension_vector": self.qudt_vector,
            "qudt_exponents": {
                axis: _format_exponent(exponent)
                for axis, exponent in self.qudt_exponents.items()
            },
            "qudt_exponent_payloads": dict(self.qudt_exponent_payloads),
            "dim_signature": self.compact,
        }


@dataclass(frozen=True)
class QudtResourceRecord:
    """Normalized QUDT unit or quantity-kind record."""

    source_entity_uri: str
    source_label: str
    resource_kind: str
    dimension: QudtDimensionMapping | None
    source_payload: dict[str, Any]
    dimension_error: dict[str, Any] | None = None
    source_description: str = ""
    symbol: str = ""
    quantity_kind_uris: tuple[str, ...] = ()
    unit_uris: tuple[str, ...] = ()

    @property
    def source_candidate_id(self) -> str:
        if self.source_entity_uri:
            return self.source_entity_uri
        stable = {
            "label": self.source_label,
            "kind": self.resource_kind,
            "symbol": self.symbol,
        }
        return f"qudt:{_stable_sha256(stable)}"

    def as_candidate_row(self) -> dict[str, Any]:
        """Return a row-shaped dict for ``physics_equation_candidates``.

        QUDT records are not equations.  They are still emitted in candidate
        shape so the common Wave 1 loader can retain all external knowledge in
        one raw-review queue while later normalization links units/quantity
        kinds into symbolic variables.
        """
        payload: dict[str, Any] = {
            "source_system": "qudt",
            "resource_kind": self.resource_kind,
            "symbol": self.symbol,
            "quantity_kind_uris": list(self.quantity_kind_uris),
            "unit_uris": list(self.unit_uris),
            "raw_record": self.source_payload,
        }
        if self.dimension is not None:
            payload.update(self.dimension.as_payload())
        if self.dimension_error is not None:
            payload["dimension_error"] = dict(self.dimension_error)
            payload["dimension_status"] = "unresolved"
        return {
            "source_candidate_id": self.source_candidate_id,
            "source_entity_uri": self.source_entity_uri,
            "source_label": self.source_label,
            "source_description": self.source_description,
            "raw_formula": "",
            "raw_formula_format": "",
            "candidate_status": (
                "dimension_resolved" if self.dimension is not None else "raw_imported"
            ),
            "parse_confidence": 1.0 if self.dimension is not None else 0.0,
            "priority_score": 0.0,
            "mechanism_tags": [],
            "behavioral_archetypes": [],
            "source_payload": payload,
            "notes": (
                "QUDT dimensional metadata record; unresolved dimension reported."
                if self.dimension_error is not None
                else "QUDT dimensional metadata record; not a standalone equation."
            ),
        }


@dataclass(frozen=True)
class QudtSnapshotManifest:
    """Deterministic output envelope for a QUDT ingestion run."""

    snapshot_row: dict[str, Any]
    records: tuple[QudtResourceRecord, ...] = field(default_factory=tuple)

    @property
    def candidate_rows(self) -> list[dict[str, Any]]:
        return [record.as_candidate_row() for record in self.records]


def qudt_dimension_vector_to_compact(value: Any) -> str:
    """Map a QUDT dimension vector value to Sciona compact notation."""
    return parse_qudt_dimension_vector(value).compact


def parse_qudt_dimension_vector(value: Any) -> QudtDimensionMapping:
    """Parse a QUDT dimension vector string, URI, JSON-LD node, or singleton."""
    vector = _coerce_dimension_vector(value)
    matches = list(_VECTOR_RE.finditer(vector))
    if not matches:
        raise QudtDimensionError(f"not a QUDT dimension vector: {value!r}")

    consumed = "".join(match.group(0) for match in matches)
    if consumed != vector:
        raise QudtDimensionError(f"unsupported QUDT dimension vector syntax: {vector!r}")

    qudt_exponents = {axis: Fraction(0) for axis in "AELIMHTD"}
    qudt_exponent_payloads = {axis: "0" for axis in "AELIMHTD"}
    for match in matches:
        axis = match.group(1)
        token = match.group(2)
        qudt_exponents[axis] = Fraction(token)
        qudt_exponent_payloads[axis] = token

    if qudt_exponents.get("D", Fraction(0)) not in (0, 1):
        raise QudtDimensionError(
            f"unsupported dimensionless axis exponent in QUDT vector: {vector!r}"
        )

    kwargs: dict[str, int] = {}
    for qudt_axis, sciona_field in _QUDT_TO_SCIONA_FIELD.items():
        exp = qudt_exponents[qudt_axis]
        if exp:
            kwargs[sciona_field] = exp

    return QudtDimensionMapping(
        qudt_vector=vector,
        qudt_exponents=qudt_exponents,
        sciona_signature=DimensionalSignature(**kwargs),
        qudt_exponent_payloads=qudt_exponent_payloads,
    )


def build_qudt_symbolic_variable_dimension_updates(
    records: Iterable[QudtResourceRecord | Mapping[str, Any]],
    variables: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return variable-row updates resolved from QUDT records without side effects.

    Variables are matched by explicit unit/quantity-kind URI first, then by
    source symbol or display label.  Returned rows are shallow copies of input
    variables with QUDT dimension fields and evidence merged in.
    """
    resolved_records = tuple(_coerce_record(record) for record in records)
    updates: list[dict[str, Any]] = []
    for variable in variables:
        record = _match_qudt_record(variable, resolved_records)
        metadata = qudt_record_to_publication_variable_dimension_metadata(record)
        if metadata is None:
            continue
        update = dict(variable)
        evidence = dict(_mapping_or_empty(update.get("evidence_json")))
        evidence.update(metadata["evidence_json"])
        update["dim_signature"] = metadata["dim_signature"]
        update["dimension_source"] = metadata["dimension_source"]
        update["evidence_json"] = evidence
        if record is not None and record.resource_kind == "quantity_kind":
            if not update.get("quantity_kind_uri"):
                update["quantity_kind_uri"] = record.source_entity_uri
            if not update.get("quantity_kind_label"):
                update["quantity_kind_label"] = record.source_label
        if record is not None and record.resource_kind == "unit":
            if not update.get("unit_uri"):
                update["unit_uri"] = record.source_entity_uri
            if not update.get("unit_label"):
                update["unit_label"] = record.source_label
        updates.append(update)
    return updates


def qudt_record_to_publication_variable_dimension_metadata(
    record: QudtResourceRecord | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return side-effect-free publication variable dimension metadata.

    Unresolved QUDT records are left out of variable metadata so they remain
    reviewable as raw candidate rows instead of being treated as dimensionless.
    """
    if record is None:
        return None
    resolved = _coerce_record(record)
    if resolved.dimension is None:
        return None
    return {
        "dim_signature": resolved.dimension.compact,
        "dimension_source": "qudt",
        "evidence_json": {
            "qudt_dimension_resolution": {
                "source_entity_uri": resolved.source_entity_uri,
                "source_label": resolved.source_label,
                "resource_kind": resolved.resource_kind,
                **resolved.dimension.as_payload(),
            }
        },
    }


def extract_qudt_resource_record(raw: dict[str, Any]) -> QudtResourceRecord:
    """Normalize one QUDT JSON-LD-like unit or quantity-kind record."""
    uri = _first_text(raw, "@id", "id", "uri")
    label = _first_text(
        raw,
        "rdfs:label",
        "label",
        "http://www.w3.org/2000/01/rdf-schema#label",
    )
    description = _first_text(
        raw,
        "dcterms:description",
        "description",
        "comment",
        "rdfs:comment",
        "http://www.w3.org/2000/01/rdf-schema#comment",
    )
    symbol = _first_text(raw, "qudt:symbol", "symbol", "http://qudt.org/schema/qudt/symbol")
    resource_kind = _infer_resource_kind(raw)

    dim_raw = _first_value(
        raw,
        "qudt:hasDimensionVector",
        "hasDimensionVector",
        "dimensionVector",
        "http://qudt.org/schema/qudt/hasDimensionVector",
    )
    dimension: QudtDimensionMapping | None = None
    dimension_error: dict[str, Any] | None = None
    if dim_raw not in (None, "", []):
        try:
            dimension = parse_qudt_dimension_vector(dim_raw)
        except QudtDimensionError as exc:
            dimension_error = {
                "raw_dimension_vector": dim_raw,
                "message": str(exc),
            }

    return QudtResourceRecord(
        source_entity_uri=uri,
        source_label=label,
        source_description=description,
        resource_kind=resource_kind,
        symbol=symbol,
        quantity_kind_uris=tuple(
            _all_ids(raw, "qudt:hasQuantityKind", "hasQuantityKind")
        ),
        unit_uris=tuple(_all_ids(raw, "qudt:applicableUnit", "applicableUnit")),
        dimension=dimension,
        dimension_error=dimension_error,
        source_payload=raw,
    )


def build_qudt_snapshot_manifest(
    raw_records: list[dict[str, Any]],
    *,
    source_version: str,
    source_uri: str = "",
    retrieved_at: datetime | None = None,
    license_expression: str = "",
    provenance_summary: str = "QUDT unit and quantity-kind dimension-vector snapshot.",
) -> QudtSnapshotManifest:
    """Build a Wave 0-compatible snapshot manifest from QUDT records."""
    records = tuple(extract_qudt_resource_record(record) for record in raw_records)
    payload = {
        "record_count": len(raw_records),
        "dimension_record_count": sum(1 for record in records if record.dimension),
        "dimension_error_count": sum(1 for record in records if record.dimension_error),
        "resource_kinds": sorted({record.resource_kind for record in records}),
        "raw_records": raw_records,
    }
    retrieved = retrieved_at or datetime.now(timezone.utc)
    snapshot_row = {
        "source_system": "qudt",
        "source_version": source_version,
        "source_uri": source_uri,
        "retrieved_at": retrieved.astimezone(timezone.utc).isoformat(),
        "adapter_name": ADAPTER_NAME,
        "adapter_version": ADAPTER_VERSION,
        "license_expression": license_expression,
        "provenance_summary": provenance_summary,
        "payload_sha256": _stable_sha256(payload),
        "payload": payload,
    }
    return QudtSnapshotManifest(snapshot_row=snapshot_row, records=records)


def _coerce_dimension_vector(value: Any) -> str:
    if isinstance(value, list):
        if len(value) != 1:
            raise QudtDimensionError(
                f"expected one QUDT dimension vector value, got {len(value)}"
            )
        return _coerce_dimension_vector(value[0])
    if isinstance(value, dict):
        for key in ("@id", "id", "value", "@value", "uri"):
            if key in value:
                return _coerce_dimension_vector(value[key])
        raise QudtDimensionError(f"dimension vector node has no value: {value!r}")
    if not isinstance(value, str):
        raise QudtDimensionError(f"dimension vector must be text-like: {value!r}")

    text = value.strip()
    if not text:
        raise QudtDimensionError("empty QUDT dimension vector")
    if _DIMENSION_VECTOR_URI_MARKER in text:
        text = text.rsplit(_DIMENSION_VECTOR_URI_MARKER, 1)[1]
    if "#" in text:
        text = text.rsplit("#", 1)[1]
    if "://" in text and "/" in text:
        text = text.rsplit("/", 1)[1]
    return text


def _infer_resource_kind(raw: dict[str, Any]) -> str:
    type_text = " ".join(_stringify(item) for item in _as_list(raw.get("@type")))
    if "QuantityKind" in type_text:
        return "quantity_kind"
    if "Unit" in type_text:
        return "unit"
    uri = _first_text(raw, "@id", "id", "uri")
    if "/quantitykind/" in uri.lower():
        return "quantity_kind"
    if "/unit/" in uri.lower():
        return "unit"
    return "qudt_resource"


def _first_value(raw: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in raw:
            values = _as_list(raw[key])
            if values:
                return values[0]
    return None


def _first_text(raw: dict[str, Any], *keys: str) -> str:
    value = _first_value(raw, *keys)
    if value is None:
        return ""
    return _stringify(value)


def _all_ids(raw: dict[str, Any], *keys: str) -> list[str]:
    ids: list[str] = []
    for key in keys:
        for value in _as_list(raw.get(key)):
            text = _stringify(value)
            if text:
                ids.append(text)
    return ids


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _stringify(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("@id", "id", "@value", "value", "uri"):
            if key in value:
                return _stringify(value[key])
        return ""
    return str(value).strip()


def _format_exponent(exp: Fraction) -> str:
    if exp.denominator == 1:
        return str(exp.numerator)
    return f"{exp.numerator}/{exp.denominator}"


def _coerce_record(record: QudtResourceRecord | Mapping[str, Any]) -> QudtResourceRecord:
    if isinstance(record, QudtResourceRecord):
        return record
    return extract_qudt_resource_record(dict(record))


def _match_qudt_record(
    variable: Mapping[str, Any],
    records: tuple[QudtResourceRecord, ...],
) -> QudtResourceRecord | None:
    unit_uri = _casefolded(variable.get("unit_uri"))
    quantity_kind_uri = _casefolded(variable.get("quantity_kind_uri"))
    symbol_candidates = {
        _casefolded(variable.get(key))
        for key in ("source_symbol", "symbol_name", "unit_label", "quantity_kind_label")
    }
    symbol_candidates.discard("")

    for record in records:
        if record.dimension is None:
            continue
        record_uri = _casefolded(record.source_entity_uri)
        if unit_uri and unit_uri == record_uri:
            return record
        if quantity_kind_uri and quantity_kind_uri == record_uri:
            return record

    for record in records:
        if record.dimension is None:
            continue
        labels = {
            _casefolded(record.symbol),
            _casefolded(record.source_label),
            _casefolded(record.source_entity_uri.rsplit("/", 1)[-1]),
        }
        labels.discard("")
        if symbol_candidates & labels:
            return record
    return None


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _casefolded(value: Any) -> str:
    return str(value or "").strip().casefold()


def _stable_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(encoded.encode("utf-8")).hexdigest()
