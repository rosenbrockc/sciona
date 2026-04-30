"""OPB source adapter scaffold.

OPB payloads are treated as already retrieved offline fixtures. The adapter
keeps every problem/equation record as a raw Wave 0 candidate and emits a
deterministic data-artifact seed for any accompanying benchmark/problem data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from sciona.physics_ingest.sources._manifest import (
    JSONDict,
    SourceAdapterBundle,
    build_snapshot_row,
    first_mapping,
    first_text,
    normalize_raw_records,
    stable_source_id,
    string_list,
)


SOURCE_SYSTEM = "opb"
ADAPTER_NAME = "sciona.physics_ingest.sources.opb"
ADAPTER_VERSION = "wave1.opb_scaffold.v1"
DEFAULT_SOURCE_URI = ""
LICENSE_SUMMARY = (
    "OPB offline payload metadata; preserve upstream benchmark/problem license "
    "and citation terms before redistribution."
)

_FORMULA_FORMATS = {
    "",
    "latex",
    "mathml",
    "content_mathml",
    "wikidata_math",
    "asciimath",
    "sympy",
    "plain_text",
}


@dataclass(frozen=True)
class OPBRecord:
    """Normalized OPB problem/equation payload."""

    source_id: str
    label: str
    source_uri: str
    formula: str = ""
    formula_format: str = ""
    variables: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    property_mappings: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    raw_record: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        record: Mapping[str, Any],
        *,
        default_source_uri: str = DEFAULT_SOURCE_URI,
    ) -> "OPBRecord":
        source_id = first_text(record, "source_id", "id", "problem_id", "equation_id")
        if not source_id:
            source_id = stable_source_id("opb", record)
        formula = first_text(
            record,
            "formula",
            "equation",
            "expression",
            "latex",
            "sympy",
            "asciimath",
        )
        formula_format = _formula_format(record, formula)
        return cls(
            source_id=source_id,
            label=first_text(record, "label", "name", "title") or source_id,
            source_uri=first_text(record, "source_uri", "uri", "url")
            or default_source_uri,
            formula=formula,
            formula_format=formula_format,
            variables=tuple(_mapping_list(record.get("variables"))),
            property_mappings=_extract_property_mappings(record),
            raw_record=dict(record),
        )

    def to_source_payload(self) -> JSONDict:
        return {
            "source_system": SOURCE_SYSTEM,
            "source_kind": "problem_or_equation",
            "source_id": self.source_id,
            "formula": self.formula,
            "formula_format": self.formula_format,
            "variables": [dict(variable) for variable in self.variables],
            "mechanism_tags": string_list(self.raw_record.get("mechanism_tags")),
            "behavioral_archetypes": string_list(
                self.raw_record.get("behavioral_archetypes")
            ),
            "property_mappings": {
                key: dict(value) for key, value in self.property_mappings.items()
            },
            "raw_record": dict(self.raw_record),
            "future_data_artifact": build_opb_data_artifact_seed(self),
        }

    def to_candidate_row(self, *, snapshot_id: str | None = None) -> JSONDict:
        row: JSONDict = {
            "source_candidate_id": self.source_id,
            "source_entity_uri": self.source_uri,
            "source_label": self.label,
            "source_description": first_text(
                self.raw_record, "description", "summary", "notes"
            ),
            "raw_formula": self.formula,
            "raw_formula_format": self.formula_format,
            "candidate_status": "raw_imported",
            "parse_confidence": 0.0,
            "priority_score": 0.25 if self.formula else 0.05,
            "mechanism_tags": string_list(self.raw_record.get("mechanism_tags")),
            "behavioral_archetypes": string_list(
                self.raw_record.get("behavioral_archetypes")
            ),
            "source_payload": self.to_source_payload(),
            "notes": (
                "OPB payload retained as a raw candidate; symbolic parsing, "
                "variable dimensions, and benchmark-data normalization pending."
            ),
        }
        if snapshot_id:
            row["snapshot_id"] = snapshot_id
        return row


def build_opb_wave0_bundle(
    raw_records: Iterable[Mapping[str, Any]],
    *,
    source_version: str,
    source_uri: str = DEFAULT_SOURCE_URI,
    retrieved_at: str | None = None,
    snapshot_id: str | None = None,
) -> SourceAdapterBundle:
    raw_records_tuple = normalize_raw_records(raw_records)
    records = tuple(
        OPBRecord.from_mapping(record, default_source_uri=source_uri)
        for record in raw_records_tuple
    )
    payload = {
        "source_kind": "opb_problem_records",
        "record_count": len(records),
        "formula_record_count": sum(1 for record in records if record.formula),
        "mapped_property_count": sum(
            len(record.property_mappings) for record in records
        ),
        "raw_records": raw_records_tuple,
    }
    return SourceAdapterBundle(
        snapshot_row=build_snapshot_row(
            source_system=SOURCE_SYSTEM,
            source_version=source_version,
            source_uri=source_uri,
            adapter_name=ADAPTER_NAME,
            adapter_version=ADAPTER_VERSION,
            payload=payload,
            license_expression=LICENSE_SUMMARY,
            provenance_summary=(
                "OPB offline problem/equation payloads captured as raw Wave 0 "
                "candidates and future data-artifact seeds."
            ),
            retrieved_at=retrieved_at,
        ),
        candidate_rows=tuple(
            record.to_candidate_row(snapshot_id=snapshot_id) for record in records
        ),
        data_artifact_seeds=tuple(
            build_opb_data_artifact_seed(record) for record in records
        ),
    )


def build_opb_data_artifact_seed(record: OPBRecord) -> JSONDict:
    return {
        "artifact_kind": "data_artifact",
        "fqdn": f"opb.record.{record.source_id.replace(':', '.')}",
        "source_system": SOURCE_SYSTEM,
        "source_id": record.source_id,
        "source_uri": record.source_uri,
        "label": record.label,
        "formula": record.formula,
        "formula_format": record.formula_format,
        "variables": [dict(variable) for variable in record.variables],
        "property_mappings": {
            key: dict(value) for key, value in record.property_mappings.items()
        },
        "data_payload": first_mapping(record.raw_record, "data", "dataset", "samples"),
    }


def _formula_format(record: Mapping[str, Any], formula: str) -> str:
    explicit = first_text(record, "formula_format", "raw_formula_format", "format")
    if explicit and explicit in _FORMULA_FORMATS:
        return explicit
    if first_text(record, "latex"):
        return "latex"
    if first_text(record, "sympy"):
        return "sympy"
    if first_text(record, "asciimath"):
        return "asciimath"
    return "plain_text" if formula else ""


def _extract_property_mappings(
    record: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    explicit = first_mapping(record, "property_mappings", "properties")
    mappings: dict[str, Mapping[str, Any]] = {}
    for key, value in explicit.items():
        if isinstance(value, Mapping):
            mappings[str(key)] = {
                "value": first_text(value, "value") or dict(value),
                "unit": first_text(value, "unit", "units"),
                "dim_signature": first_text(value, "dim_signature", "dimension"),
                "mapping_status": (
                    "dimension_mapped"
                    if first_text(value, "dim_signature", "dimension")
                    else "raw_unit_retained"
                ),
            }
        elif str(value):
            mappings[str(key)] = {
                "value": str(value),
                "unit": "",
                "dim_signature": "",
                "mapping_status": "raw_unit_retained",
            }
    return mappings


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
