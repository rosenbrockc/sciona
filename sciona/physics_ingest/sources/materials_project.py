"""Materials Project source adapter scaffold.

This module is intentionally offline-only. It accepts already retrieved
Materials Project documents and preserves them as Wave 0 source snapshots,
raw candidates, and deterministic data-artifact seeds. Candidates are emitted
even when formulas or property mappings are incomplete.
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


SOURCE_SYSTEM = "materials_project"
ADAPTER_NAME = "sciona.physics_ingest.sources.materials_project"
ADAPTER_VERSION = "wave1.materials_project_scaffold.v1"
DEFAULT_SOURCE_URI = "https://materialsproject.org/"
LICENSE_SUMMARY = (
    "Materials Project computed materials metadata; preserve Materials Project "
    "citation, API terms, and downstream license notices."
)

_PROPERTY_ALIASES = {
    "band_gap": ("band_gap", "bandgap", "e_gap"),
    "formation_energy_per_atom": (
        "formation_energy_per_atom",
        "formation_energy",
        "e_form",
    ),
    "energy_above_hull": ("energy_above_hull", "e_above_hull"),
    "density": ("density", "rho"),
    "volume": ("volume", "cell_volume"),
    "total_magnetization": ("total_magnetization", "magnetization"),
}

_PROPERTY_UNITS = {
    "band_gap": "eV",
    "formation_energy_per_atom": "eV/atom",
    "energy_above_hull": "eV/atom",
    "density": "g cm^-3",
    "volume": "Angstrom^3",
    "total_magnetization": "mu_B",
}

_PROPERTY_DIMENSIONS = {
    "band_gap": "M1L2T-2",
    "formation_energy_per_atom": "M1L2T-2N-1",
    "energy_above_hull": "M1L2T-2N-1",
    "density": "M1L-3",
    "volume": "L3",
    "total_magnetization": "",
}


@dataclass(frozen=True)
class MaterialsProjectRecord:
    """Normalized Materials Project material document."""

    source_id: str
    label: str
    source_uri: str
    formula: str = ""
    composition: Mapping[str, Any] = field(default_factory=dict)
    property_mappings: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    raw_record: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        record: Mapping[str, Any],
        *,
        default_source_uri: str = DEFAULT_SOURCE_URI,
    ) -> "MaterialsProjectRecord":
        source_id = first_text(record, "material_id", "task_id", "source_id", "id")
        if not source_id:
            source_id = stable_source_id("materials_project", record)
        formula = first_text(
            record,
            "formula_pretty",
            "pretty_formula",
            "formula",
            "full_formula",
            "reduced_formula",
        )
        composition = first_mapping(record, "composition", "composition_reduced")
        label = first_text(record, "label", "name", "title") or formula or source_id
        source_uri = first_text(record, "source_uri", "uri", "url")
        if not source_uri and source_id.startswith("mp-"):
            source_uri = f"{default_source_uri.rstrip('/')}/materials/{source_id}/"
        return cls(
            source_id=source_id,
            label=label,
            source_uri=source_uri or default_source_uri,
            formula=formula,
            composition=dict(composition),
            property_mappings=_extract_property_mappings(record),
            raw_record=dict(record),
        )

    def to_source_payload(self) -> JSONDict:
        return {
            "source_system": SOURCE_SYSTEM,
            "source_kind": "computed_material",
            "source_id": self.source_id,
            "formula": self.formula,
            "composition": dict(self.composition),
            "elements": string_list(self.raw_record.get("elements")),
            "property_mappings": {
                key: dict(value) for key, value in self.property_mappings.items()
            },
            "raw_record": dict(self.raw_record),
            "future_data_artifact": build_materials_project_data_artifact_seed(self),
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
            "raw_formula_format": "plain_text" if self.formula else "",
            "candidate_status": "raw_imported",
            "parse_confidence": 0.0,
            "priority_score": 0.2,
            "mechanism_tags": ["materials", "reference_data"],
            "behavioral_archetypes": ["material_property"],
            "source_payload": self.to_source_payload(),
            "notes": (
                "Materials Project material payload retained as a raw "
                "candidate; formula and property normalization pending."
            ),
        }
        if snapshot_id:
            row["snapshot_id"] = snapshot_id
        return row


def build_materials_project_wave0_bundle(
    raw_records: Iterable[Mapping[str, Any]],
    *,
    source_version: str,
    source_uri: str = DEFAULT_SOURCE_URI,
    retrieved_at: str | None = None,
    snapshot_id: str | None = None,
) -> SourceAdapterBundle:
    raw_records_tuple = normalize_raw_records(raw_records)
    records = tuple(
        MaterialsProjectRecord.from_mapping(record, default_source_uri=source_uri)
        for record in raw_records_tuple
    )
    payload = {
        "source_kind": "materials_project_documents",
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
                "Materials Project payloads captured as raw Wave 0 candidates "
                "and future data-artifact seeds."
            ),
            retrieved_at=retrieved_at,
        ),
        candidate_rows=tuple(
            record.to_candidate_row(snapshot_id=snapshot_id) for record in records
        ),
        data_artifact_seeds=tuple(
            build_materials_project_data_artifact_seed(record) for record in records
        ),
    )


def build_materials_project_data_artifact_seed(
    record: MaterialsProjectRecord,
) -> JSONDict:
    return {
        "artifact_kind": "data_artifact",
        "fqdn": f"materials_project.material.{record.source_id.replace('-', '_')}",
        "source_system": SOURCE_SYSTEM,
        "source_id": record.source_id,
        "source_uri": record.source_uri,
        "label": record.label,
        "formula": record.formula,
        "composition": dict(record.composition),
        "property_mappings": {
            key: dict(value) for key, value in record.property_mappings.items()
        },
    }


def _extract_property_mappings(
    record: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    explicit = first_mapping(record, "property_mappings", "properties")
    mappings: dict[str, Mapping[str, Any]] = {}
    for property_name, aliases in _PROPERTY_ALIASES.items():
        value = _first_present(record, *aliases)
        if value == "":
            value = _first_present(explicit, property_name)
        if value == "":
            continue
        mappings[property_name] = {
            "value": value,
            "unit": _PROPERTY_UNITS[property_name],
            "dim_signature": _PROPERTY_DIMENSIONS[property_name],
            "mapping_status": (
                "dimension_mapped"
                if _PROPERTY_DIMENSIONS[property_name]
                else "raw_unit_retained"
            ),
        }
    return mappings


def _first_present(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        if key in row:
            value = first_text(row, key)
            if value:
                return value
    return ""
