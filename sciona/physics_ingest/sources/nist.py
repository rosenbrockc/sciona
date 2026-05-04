"""NIST source helpers for physics symbolic ingestion.

This module is intentionally side-effect free: it performs no network access
and no database writes.  It converts already-fetched NIST CODATA and DLMF
payloads into Wave 0 snapshot/candidate row dictionaries plus future
state-artifact seed dictionaries for constants.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from typing import Any, Iterable, Mapping

try:  # Keep the adapter usable in minimal environments.
    from sciona.ghost.dimensions import DimensionalSignature, parse_units_string
except Exception:  # pragma: no cover - exercised only without optional deps
    DimensionalSignature = None  # type: ignore[assignment]

    def parse_units_string(_s: str) -> None:  # type: ignore[no-redef]
        return None


ADAPTER_NAME = "sciona.physics_ingest.sources.nist"
ADAPTER_VERSION = "0.1.0"

NIST_CODATA_SOURCE_SYSTEM = "nist_codata"
NIST_DLMF_SOURCE_SYSTEM = "nist_dlmf"

NIST_CODATA_DEFAULT_URI = (
    "https://physics.nist.gov/cuu/Constants/Table/allascii.txt"
)
NIST_DLMF_DEFAULT_URI = "https://dlmf.nist.gov/"

CODATA_LICENSE_SUMMARY = (
    "NIST public-domain/government reference data; verify downstream citation "
    "and redistribution requirements before publication."
)
DLMF_LICENSE_SUMMARY = (
    "NIST DLMF reference metadata; preserve DLMF section/equation citations "
    "and verify downstream redistribution terms before publication."
)


@dataclass(frozen=True)
class CODATAConstant:
    """A parsed CODATA constant record."""

    source_id: str
    label: str
    value_text: str
    uncertainty_text: str
    unit_text: str
    source_version: str
    source_uri: str = NIST_CODATA_DEFAULT_URI
    symbol: str = ""
    description: str = ""
    quantity_kind_hint: str = ""
    dim_signature_hint: str = ""
    reference_ids: tuple[str, ...] = field(default_factory=tuple)
    raw_record: Mapping[str, Any] = field(default_factory=dict)

    @property
    def normalized_value(self) -> str:
        return normalize_codata_number(self.value_text)

    @property
    def normalized_uncertainty(self) -> str:
        if self.is_exact:
            return "0"
        return normalize_codata_number(self.uncertainty_text)

    @property
    def is_exact(self) -> bool:
        text = self.uncertainty_text.strip().lower()
        return text in {"exact", "(exact)", "0"} or self.value_text.strip().endswith("...")

    @property
    def dim_signature(self) -> str:
        if self.dim_signature_hint:
            return self.dim_signature_hint
        return infer_unit_dim_signature(self.unit_text)

    def raw_formula(self) -> str:
        lhs = self.symbol or self.label
        unit = f" {self.unit_text}" if self.unit_text else ""
        return f"{lhs} = {self.value_text}{unit}"

    def to_source_payload(self) -> dict[str, Any]:
        return {
            "source_kind": "constant",
            "source_system": NIST_CODATA_SOURCE_SYSTEM,
            "source_version": self.source_version,
            "source_uri": self.source_uri,
            "source_id": self.source_id,
            "ingestion_target_kind": "state_artifact",
            "symbolic_equation_candidate": False,
            "label": self.label,
            "symbol": self.symbol,
            "value_text": self.value_text,
            "normalized_value": self.normalized_value,
            "uncertainty_text": self.uncertainty_text,
            "normalized_uncertainty": self.normalized_uncertainty,
            "is_exact": self.is_exact,
            "unit_text": self.unit_text,
            "quantity_kind_hint": self.quantity_kind_hint,
            "dim_signature_hint": self.dim_signature,
            "reference_ids": list(self.reference_ids),
            "raw_record": dict(self.raw_record),
            "future_data_artifact": build_codata_data_artifact_seed(self),
        }


@dataclass(frozen=True)
class DLMFFunctionEntry:
    """A DLMF symbolic function/equation metadata record."""

    source_id: str
    label: str
    formula: str
    formula_format: str
    source_version: str
    source_uri: str = NIST_DLMF_DEFAULT_URI
    description: str = ""
    variables: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    constraints: tuple[str, ...] = field(default_factory=tuple)
    function_symbols: tuple[str, ...] = field(default_factory=tuple)
    reference_ids: tuple[str, ...] = field(default_factory=tuple)
    raw_record: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        record: Mapping[str, Any],
        *,
        source_version: str,
        default_source_uri: str = NIST_DLMF_DEFAULT_URI,
    ) -> DLMFFunctionEntry:
        return cls(
            source_id=str(record.get("source_id") or record.get("id") or ""),
            label=str(record.get("label") or record.get("name") or ""),
            formula=str(record.get("formula") or record.get("latex") or ""),
            formula_format=str(record.get("formula_format") or "latex"),
            source_version=source_version,
            source_uri=str(record.get("source_uri") or default_source_uri),
            description=str(record.get("description") or ""),
            variables=tuple(_mapping_list(record.get("variables"))),
            constraints=tuple(str(item) for item in record.get("constraints", ()) or ()),
            function_symbols=tuple(
                str(item) for item in record.get("function_symbols", ()) or ()
            ),
            reference_ids=tuple(str(item) for item in record.get("reference_ids", ()) or ()),
            raw_record=dict(record),
        )

    def to_source_payload(self) -> dict[str, Any]:
        return {
            "source_kind": "symbolic_function_metadata",
            "source_system": NIST_DLMF_SOURCE_SYSTEM,
            "source_version": self.source_version,
            "source_uri": self.source_uri,
            "source_id": self.source_id,
            "label": self.label,
            "formula": self.formula,
            "formula_format": self.formula_format,
            "variables": [dict(item) for item in self.variables],
            "constraints": list(self.constraints),
            "function_symbols": list(self.function_symbols),
            "reference_ids": list(self.reference_ids),
            "raw_record": dict(self.raw_record),
        }


@dataclass(frozen=True)
class Wave0SourceBundle:
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


def parse_codata_ascii(
    text: str,
    *,
    source_version: str,
    source_uri: str = NIST_CODATA_DEFAULT_URI,
    symbol_map: Mapping[str, str] | None = None,
    dim_signature_hints: Mapping[str, str] | None = None,
    quantity_kind_hints: Mapping[str, str] | None = None,
    reference_ids: Iterable[str] = (),
) -> tuple[CODATAConstant, ...]:
    """Parse NIST CODATA fixed-width or pipe-delimited constant records.

    The official ASCII table is fixed width.  Tests and hand-curated seed
    files may use pipe delimiters to avoid brittle spacing.
    """

    symbols = {key.casefold(): value for key, value in (symbol_map or {}).items()}
    dims = {key.casefold(): value for key, value in (dim_signature_hints or {}).items()}
    quantities = {
        key.casefold(): value for key, value in (quantity_kind_hints or {}).items()
    }
    refs = tuple(str(item) for item in reference_ids)

    constants: list[CODATAConstant] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        parsed = _parse_codata_line(raw_line)
        if parsed is None:
            continue
        label, value, uncertainty, unit, extras = parsed
        label_key = label.casefold()
        source_id = str(extras.get("source_id") or _slug(label))
        constant_refs = tuple(str(item) for item in extras.get("reference_ids", refs))
        constants.append(
            CODATAConstant(
                source_id=source_id,
                label=label,
                value_text=value,
                uncertainty_text=uncertainty,
                unit_text=unit,
                source_version=source_version,
                source_uri=source_uri,
                symbol=str(extras.get("symbol") or symbols.get(label_key, "")),
                description=str(extras.get("description") or ""),
                quantity_kind_hint=str(
                    extras.get("quantity_kind_hint")
                    or quantities.get(label_key, "")
                ),
                dim_signature_hint=str(
                    extras.get("dim_signature_hint") or dims.get(label_key, "")
                ),
                reference_ids=constant_refs,
                raw_record={
                    "line_number": line_no,
                    "raw_line": raw_line,
                    "parsed_extras": extras,
                },
            )
        )
    return tuple(constants)


def build_codata_wave0_bundle(
    constants: Iterable[CODATAConstant],
    *,
    source_version: str,
    source_uri: str = NIST_CODATA_DEFAULT_URI,
    retrieved_at: str | None = None,
    snapshot_id: str | None = None,
) -> Wave0SourceBundle:
    constants_tuple = tuple(constants)
    payload = {
        "source_kind": "codata_constants",
        "constants": [constant.to_source_payload() for constant in constants_tuple],
    }
    return Wave0SourceBundle(
        snapshot_row=build_snapshot_row(
            source_system=NIST_CODATA_SOURCE_SYSTEM,
            source_version=source_version,
            source_uri=source_uri,
            payload=payload,
            license_expression=CODATA_LICENSE_SUMMARY,
            provenance_summary=(
                "NIST CODATA constants parsed into immutable Wave 0 source "
                "snapshots; constants are candidates for future state artifacts."
            ),
            retrieved_at=retrieved_at,
        ),
        candidate_rows=tuple(
            build_codata_candidate_row(constant, snapshot_id=snapshot_id)
            for constant in constants_tuple
        ),
        data_artifact_seeds=tuple(
            build_codata_data_artifact_seed(constant) for constant in constants_tuple
        ),
    )


def build_dlmf_wave0_bundle(
    entries: Iterable[DLMFFunctionEntry],
    *,
    source_version: str,
    source_uri: str = NIST_DLMF_DEFAULT_URI,
    retrieved_at: str | None = None,
    snapshot_id: str | None = None,
) -> Wave0SourceBundle:
    entries_tuple = tuple(entries)
    payload = {
        "source_kind": "dlmf_symbolic_function_metadata",
        "entries": [entry.to_source_payload() for entry in entries_tuple],
    }
    return Wave0SourceBundle(
        snapshot_row=build_snapshot_row(
            source_system=NIST_DLMF_SOURCE_SYSTEM,
            source_version=source_version,
            source_uri=source_uri,
            payload=payload,
            license_expression=DLMF_LICENSE_SUMMARY,
            provenance_summary=(
                "NIST DLMF symbolic function metadata captured as Wave 0 "
                "equation candidates for later SymPy normalization."
            ),
            retrieved_at=retrieved_at,
        ),
        candidate_rows=tuple(
            build_dlmf_candidate_row(entry, snapshot_id=snapshot_id)
            for entry in entries_tuple
        ),
    )


def build_snapshot_row(
    *,
    source_system: str,
    source_version: str,
    source_uri: str,
    payload: Mapping[str, Any],
    license_expression: str,
    provenance_summary: str,
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    row = {
        "source_system": source_system,
        "source_version": source_version,
        "source_uri": source_uri,
        "adapter_name": ADAPTER_NAME,
        "adapter_version": ADAPTER_VERSION,
        "license_expression": license_expression,
        "provenance_summary": provenance_summary,
        "payload_sha256": stable_payload_sha256(payload),
        "payload": _jsonable(payload),
    }
    if retrieved_at:
        row["retrieved_at"] = retrieved_at
    return row


def build_codata_candidate_row(
    constant: CODATAConstant,
    *,
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    row = {
        "source_candidate_id": constant.source_id,
        "source_entity_uri": _append_fragment(constant.source_uri, constant.source_id),
        "source_label": constant.label,
        "source_description": constant.description,
        "raw_formula": constant.raw_formula(),
        "raw_formula_format": "plain_text",
        "candidate_status": "source_verified",
        "parse_confidence": 1.0,
        "priority_score": 0.35,
        "mechanism_tags": ["constant", "reference_data"],
        "behavioral_archetypes": ["calibration_reference"],
        "source_payload": constant.to_source_payload(),
        "notes": (
            "CODATA constants are represented as raw candidates for Wave 0 "
            "and should publish as state artifacts, not standalone symbolic "
            "equations, before executable use."
        ),
    }
    if snapshot_id:
        row["snapshot_id"] = snapshot_id
    return row


def build_dlmf_candidate_row(
    entry: DLMFFunctionEntry,
    *,
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    row = {
        "source_candidate_id": entry.source_id,
        "source_entity_uri": _append_fragment(entry.source_uri, entry.source_id),
        "source_label": entry.label,
        "source_description": entry.description,
        "raw_formula": entry.formula,
        "raw_formula_format": entry.formula_format,
        "candidate_status": "raw_imported",
        "parse_confidence": 0.0,
        "priority_score": 0.25,
        "mechanism_tags": ["special_function", "mathematical_reference"],
        "behavioral_archetypes": ["symbolic_transform"],
        "source_payload": entry.to_source_payload(),
        "notes": (
            "DLMF metadata requires formula parsing and convention review "
            "before symbolic artifact publication."
        ),
    }
    if snapshot_id:
        row["snapshot_id"] = snapshot_id
    return row


def build_codata_data_artifact_seed(constant: CODATAConstant) -> dict[str, Any]:
    symbol = constant.symbol or _slug(constant.label).replace("-", "_")
    return {
        "artifact_kind": "state_artifact",
        "fqdn": f"nist.codata.{symbol}",
        "source_system": NIST_CODATA_SOURCE_SYSTEM,
        "source_version": constant.source_version,
        "source_id": constant.source_id,
        "source_uri": constant.source_uri,
        "label": constant.label,
        "symbol": constant.symbol,
        "value_text": constant.value_text,
        "normalized_value": constant.normalized_value,
        "uncertainty_text": constant.uncertainty_text,
        "normalized_uncertainty": constant.normalized_uncertainty,
        "is_exact": constant.is_exact,
        "unit_text": constant.unit_text,
        "quantity_kind_hint": constant.quantity_kind_hint,
        "dim_signature_hint": constant.dim_signature,
        "reference_ids": list(constant.reference_ids),
    }


def stable_payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_codata_number(text: str) -> str:
    """Normalize CODATA numeric text while preserving exact/ellipsis semantics."""

    stripped = text.strip()
    if not stripped:
        return ""
    if stripped.lower() in {"exact", "(exact)"}:
        return "0"

    without_spaces = re.sub(r"(?<=\d)\s+(?=\d)", "", stripped)
    without_spaces = without_spaces.replace(" ", "")
    without_ellipsis = without_spaces.rstrip(".")
    try:
        return str(Decimal(without_ellipsis))
    except InvalidOperation:
        return without_spaces


def infer_unit_dim_signature(unit_text: str) -> str:
    """Best-effort compact dimension hint from a CODATA unit string."""

    normalized = _normalize_unit_text(unit_text)
    if not normalized:
        return "1"

    parsed = parse_units_string(normalized)
    if parsed is not None:
        return parsed.to_compact()

    if DimensionalSignature is None:
        return ""

    unit_dims = {
        "m": DimensionalSignature(L=1),
        "kg": DimensionalSignature(M=1),
        "s": DimensionalSignature(T=1),
        "a": DimensionalSignature(I=1),
        "k": DimensionalSignature(Theta=1),
        "mol": DimensionalSignature(N=1),
        "cd": DimensionalSignature(J=1),
        "hz": DimensionalSignature(T=-1),
        "n": DimensionalSignature(M=1, L=1, T=-2),
        "j": DimensionalSignature(M=1, L=2, T=-2),
        "w": DimensionalSignature(M=1, L=2, T=-3),
        "c": DimensionalSignature(T=1, I=1),
        "v": DimensionalSignature(M=1, L=2, T=-3, I=-1),
        "t": DimensionalSignature(M=1, T=-2, I=-1),
    }

    result = DimensionalSignature()
    matched = False
    for token in normalized.split():
        match = re.fullmatch(r"([a-z]+)(?:\^?(-?\d+))?", token)
        if match is None:
            return ""
        unit = match.group(1)
        exponent = int(match.group(2) or "1")
        dim = unit_dims.get(unit)
        if dim is None:
            return ""
        result = result.multiply(dim.power(exponent))
        matched = True
    return result.to_compact() if matched else ""


def _parse_codata_line(raw_line: str) -> tuple[str, str, str, str, dict[str, Any]] | None:
    line = raw_line.rstrip("\n")
    if not line.strip() or set(line.strip()) <= {"-", " "}:
        return None
    if _looks_like_header(line):
        return None

    if "|" in line:
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 4:
            return None
        extras: dict[str, Any] = {}
        for extra in parts[4:]:
            if not extra:
                continue
            if "=" in extra:
                key, value = extra.split("=", 1)
                if key.strip() == "reference_ids":
                    extras[key.strip()] = [
                        item.strip() for item in value.split(",") if item.strip()
                    ]
                else:
                    extras[key.strip()] = value.strip()
        return parts[0], parts[1], parts[2], parts[3], extras

    if len(line) < 92:
        return None

    label = line[:60].strip()
    value = line[60:85].strip()
    uncertainty = line[85:110].strip()
    unit = line[110:].strip()
    if not label or not value:
        return None
    return label, value, uncertainty, unit, {}


def _looks_like_header(line: str) -> bool:
    lowered = line.strip().casefold()
    return lowered.startswith("quantity") or lowered.startswith("source:")


def _normalize_unit_text(unit_text: str) -> str:
    normalized = unit_text.strip().lower()
    normalized = normalized.replace("^-", "-")
    normalized = normalized.replace("^", "")
    normalized = normalized.replace("·", " ")
    normalized = normalized.replace("*", " ")
    normalized = normalized.replace("/", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def _append_fragment(uri: str, fragment: str) -> str:
    if not uri or not fragment:
        return uri
    separator = "&" if "?" in uri else "#"
    return f"{uri}{separator}{fragment}"


def _slug(text: str) -> str:
    lowered = text.strip().casefold()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or "unknown"


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    result: list[Mapping[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            result.append(item)
    return result


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Decimal):
        return str(value)
    return value
