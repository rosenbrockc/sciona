"""HITRAN source adapter scaffold.

The adapter performs no network access. It converts already retrieved HITRAN
line/property payloads into immutable Wave 0 snapshot rows, raw candidate rows,
and deterministic data-artifact seed dictionaries. Records are retained even
when transition labels, units, or property mappings are incomplete.
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
)


SOURCE_SYSTEM = "hitran"
ADAPTER_NAME = "sciona.physics_ingest.sources.hitran"
ADAPTER_VERSION = "wave1.hitran_scaffold.v1"
DEFAULT_SOURCE_URI = "https://hitran.org/"
LICENSE_SUMMARY = (
    "HITRAN spectroscopic database metadata; preserve source citation and "
    "verify license terms before redistribution."
)

_PROPERTY_ALIASES = {
    "wavenumber": ("nu", "wavenumber", "line_center", "transition_wavenumber"),
    "line_intensity": ("sw", "intensity", "line_intensity", "strength"),
    "einstein_a": ("a", "einstein_a", "einstein_a_coefficient"),
    "air_broadening": ("gamma_air", "air_broadening", "air_width"),
    "self_broadening": ("gamma_self", "self_broadening", "self_width"),
    "lower_state_energy": ("elower", "lower_state_energy", "lower_energy"),
    "temperature_exponent": ("n_air", "temperature_exponent"),
    "pressure_shift": ("delta_air", "pressure_shift"),
}

_PROPERTY_UNITS = {
    "wavenumber": "cm^-1",
    "line_intensity": "cm^-1/(molecule cm^-2)",
    "einstein_a": "s^-1",
    "air_broadening": "cm^-1 atm^-1",
    "self_broadening": "cm^-1 atm^-1",
    "lower_state_energy": "cm^-1",
    "temperature_exponent": "1",
    "pressure_shift": "cm^-1 atm^-1",
}

_PROPERTY_DIMENSIONS = {
    "wavenumber": "L-1",
    "line_intensity": "",
    "einstein_a": "T-1",
    "air_broadening": "",
    "self_broadening": "",
    "lower_state_energy": "L-1",
    "temperature_exponent": "1",
    "pressure_shift": "",
}


@dataclass(frozen=True)
class HITRANLineRecord:
    """Normalized HITRAN line or cross-section payload."""

    source_id: str
    label: str
    source_uri: str
    molecule: str = ""
    isotopologue: str = ""
    transition: str = ""
    wavenumber: str = ""
    property_mappings: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    raw_record: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        record: Mapping[str, Any],
        *,
        default_source_uri: str = DEFAULT_SOURCE_URI,
    ) -> "HITRANLineRecord":
        source_id = first_text(record, "source_id", "id", "line_id", "global_id")
        if not source_id:
            source_id = stable_source_id("hitran", record)
        molecule = first_text(record, "molecule", "molecule_name", "species")
        isotopologue = first_text(record, "isotopologue", "isotopologue_name", "iso")
        transition = first_text(
            record,
            "transition",
            "transition_label",
            "quantum_transition",
            "local_iso_id",
        )
        mappings = _extract_property_mappings(record)
        wavenumber = str(mappings.get("wavenumber", {}).get("value", ""))
        label = (
            first_text(record, "label", "name", "title")
            or " ".join(part for part in (molecule, isotopologue, transition) if part)
            or source_id
        )
        return cls(
            source_id=source_id,
            label=label,
            source_uri=first_text(record, "source_uri", "uri", "url")
            or default_source_uri,
            molecule=molecule,
            isotopologue=isotopologue,
            transition=transition,
            wavenumber=wavenumber,
            property_mappings=mappings,
            raw_record=dict(record),
        )

    def raw_formula(self) -> str:
        parts = [
            part
            for part in (self.molecule, self.isotopologue, self.transition)
            if part
        ]
        if self.wavenumber:
            parts.append(f"{self.wavenumber} cm^-1")
        return " | ".join(parts)

    def to_source_payload(self) -> JSONDict:
        return {
            "source_system": SOURCE_SYSTEM,
            "source_kind": "spectral_line",
            "source_id": self.source_id,
            "molecule": self.molecule,
            "isotopologue": self.isotopologue,
            "transition": self.transition,
            "property_mappings": {
                key: dict(value) for key, value in self.property_mappings.items()
            },
            "raw_record": dict(self.raw_record),
            "future_data_artifact": build_hitran_data_artifact_seed(self),
        }

    def to_candidate_row(self, *, snapshot_id: str | None = None) -> JSONDict:
        row: JSONDict = {
            "source_candidate_id": self.source_id,
            "source_entity_uri": self.source_uri,
            "source_label": self.label,
            "source_description": first_text(self.raw_record, "description", "comment"),
            "raw_formula": self.raw_formula(),
            "raw_formula_format": "plain_text" if self.raw_formula() else "",
            "candidate_status": "raw_imported",
            "parse_confidence": 0.0,
            "priority_score": 0.15,
            "mechanism_tags": ["spectroscopy", "reference_data"],
            "behavioral_archetypes": ["spectral_transition"],
            "source_payload": self.to_source_payload(),
            "notes": (
                "HITRAN spectral metadata retained as a raw candidate; "
                "formula and property normalization pending."
            ),
        }
        if snapshot_id:
            row["snapshot_id"] = snapshot_id
        return row


def build_hitran_wave0_bundle(
    raw_records: Iterable[Mapping[str, Any]],
    *,
    source_version: str,
    source_uri: str = DEFAULT_SOURCE_URI,
    retrieved_at: str | None = None,
    snapshot_id: str | None = None,
) -> SourceAdapterBundle:
    raw_records_tuple = normalize_raw_records(raw_records)
    records = tuple(
        HITRANLineRecord.from_mapping(record, default_source_uri=source_uri)
        for record in raw_records_tuple
    )
    payload = {
        "source_kind": "hitran_spectral_lines",
        "record_count": len(records),
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
                "HITRAN spectral line payloads captured as raw Wave 0 "
                "candidates and future data-artifact seeds."
            ),
            retrieved_at=retrieved_at,
        ),
        candidate_rows=tuple(
            record.to_candidate_row(snapshot_id=snapshot_id) for record in records
        ),
        data_artifact_seeds=tuple(
            build_hitran_data_artifact_seed(record) for record in records
        ),
    )


def build_hitran_data_artifact_seed(record: HITRANLineRecord) -> JSONDict:
    return {
        "artifact_kind": "data_artifact",
        "fqdn": f"hitran.line.{record.source_id.replace(':', '.')}",
        "source_system": SOURCE_SYSTEM,
        "source_id": record.source_id,
        "source_uri": record.source_uri,
        "label": record.label,
        "molecule": record.molecule,
        "isotopologue": record.isotopologue,
        "transition": record.transition,
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
