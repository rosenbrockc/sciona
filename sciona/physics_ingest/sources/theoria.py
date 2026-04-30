"""TheorIA source adapter scaffold.

The adapter is offline-only. It accepts already retrieved TheorIA payloads and
preserves every theory/problem/evaluation record as a Wave 0 snapshot row, raw
candidate row, and deterministic benchmark/evaluation data-artifact seed. Rows
are retained even when formulas are absent or license review is still pending.
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


SOURCE_SYSTEM = "theoria"
ADAPTER_NAME = "sciona.physics_ingest.sources.theoria"
ADAPTER_VERSION = "wave1.theoria_scaffold.v1"
DEFAULT_SOURCE_URI = ""
LICENSE_SUMMARY = (
    "TheorIA offline payload metadata; license status may be pending. Preserve "
    "upstream citation, dataset, and redistribution terms before publishing."
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
class TheorIARecord:
    """Normalized TheorIA theory/problem/evaluation payload."""

    source_id: str
    label: str
    source_uri: str
    formula: str = ""
    formula_format: str = ""
    theory: str = ""
    task_kind: str = ""
    variables: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    evaluation_payload: Mapping[str, Any] = field(default_factory=dict)
    raw_record: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        record: Mapping[str, Any],
        *,
        default_source_uri: str = DEFAULT_SOURCE_URI,
    ) -> "TheorIARecord":
        source_id = first_text(
            record,
            "source_id",
            "id",
            "record_id",
            "problem_id",
            "theory_id",
        )
        if not source_id:
            source_id = stable_source_id("theoria", record)
        formula = first_text(
            record,
            "formula",
            "equation",
            "expression",
            "target_expression",
            "latex",
            "sympy",
            "asciimath",
        )
        return cls(
            source_id=source_id,
            label=first_text(record, "label", "name", "title") or source_id,
            source_uri=first_text(record, "source_uri", "uri", "url")
            or default_source_uri,
            formula=formula,
            formula_format=_formula_format(record, formula),
            theory=first_text(record, "theory", "domain", "category"),
            task_kind=first_text(record, "task_kind", "task", "problem_type", "type"),
            variables=tuple(_mapping_list(record.get("variables"))),
            evaluation_payload=_evaluation_payload(record),
            raw_record=dict(record),
        )

    def to_source_payload(self) -> JSONDict:
        return {
            "source_system": SOURCE_SYSTEM,
            "source_kind": "theory_problem_or_evaluation",
            "source_id": self.source_id,
            "theory": self.theory,
            "task_kind": self.task_kind,
            "formula": self.formula,
            "formula_format": self.formula_format,
            "variables": [dict(variable) for variable in self.variables],
            "mechanism_tags": string_list(self.raw_record.get("mechanism_tags")),
            "behavioral_archetypes": string_list(
                self.raw_record.get("behavioral_archetypes")
            ),
            "evaluation_payload": dict(self.evaluation_payload),
            "raw_record": dict(self.raw_record),
            "future_data_artifact": build_theoria_data_artifact_seed(self),
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
            "priority_score": 0.3 if self.formula else 0.05,
            "mechanism_tags": string_list(self.raw_record.get("mechanism_tags")),
            "behavioral_archetypes": string_list(
                self.raw_record.get("behavioral_archetypes")
            ),
            "source_payload": self.to_source_payload(),
            "notes": (
                "TheorIA payload retained as a raw candidate; symbolic parsing, "
                "license review, and benchmark/evaluation normalization pending."
            ),
        }
        if snapshot_id:
            row["snapshot_id"] = snapshot_id
        return row


def build_theoria_wave0_bundle(
    raw_records: Iterable[Mapping[str, Any]],
    *,
    source_version: str,
    source_uri: str = DEFAULT_SOURCE_URI,
    retrieved_at: str | None = None,
    snapshot_id: str | None = None,
    license_expression: str = LICENSE_SUMMARY,
) -> SourceAdapterBundle:
    raw_records_tuple = normalize_raw_records(raw_records)
    records = tuple(
        TheorIARecord.from_mapping(record, default_source_uri=source_uri)
        for record in raw_records_tuple
    )
    payload = {
        "source_kind": "theoria_records",
        "record_count": len(records),
        "formula_record_count": sum(1 for record in records if record.formula),
        "evaluation_record_count": sum(
            1 for record in records if record.evaluation_payload
        ),
        "license_review_status": (
            "pending" if "pending" in license_expression.casefold() else "provided"
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
            license_expression=license_expression,
            provenance_summary=(
                "TheorIA offline theory/problem payloads captured as raw Wave 0 "
                "candidates and future benchmark/evaluation data-artifact seeds."
            ),
            retrieved_at=retrieved_at,
        ),
        candidate_rows=tuple(
            record.to_candidate_row(snapshot_id=snapshot_id) for record in records
        ),
        data_artifact_seeds=tuple(
            build_theoria_data_artifact_seed(record) for record in records
        ),
    )


def build_theoria_data_artifact_seed(record: TheorIARecord) -> JSONDict:
    return {
        "artifact_kind": "data_artifact",
        "artifact_role": "benchmark_evaluation_seed",
        "fqdn": f"theoria.record.{record.source_id.replace(':', '.')}",
        "source_system": SOURCE_SYSTEM,
        "source_id": record.source_id,
        "source_uri": record.source_uri,
        "label": record.label,
        "theory": record.theory,
        "task_kind": record.task_kind,
        "formula": record.formula,
        "formula_format": record.formula_format,
        "variables": [dict(variable) for variable in record.variables],
        "evaluation_payload": dict(record.evaluation_payload),
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


def _evaluation_payload(record: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("evaluation", "evaluation_spec", "benchmark", "benchmark_data", "data"):
        payload = first_mapping(record, key)
        if payload:
            return dict(payload)
    return {}


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
