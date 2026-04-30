from __future__ import annotations

import re

from sciona.physics_ingest.staging import stage_source_rows
from sciona.physics_ingest.sources.hitran import build_hitran_wave0_bundle
from sciona.physics_ingest.sources.materials_project import (
    build_materials_project_wave0_bundle,
)
from sciona.physics_ingest.sources.opb import build_opb_wave0_bundle


SNAPSHOT_ID = "00000000-0000-0000-0000-000000000010"


def test_hitran_bundle_retains_raw_lines_and_builds_artifact_seeds() -> None:
    raw_records = [
        {
            "id": "HITRAN:H2O:0001",
            "molecule": "H2O",
            "isotopologue": "1H2-16O",
            "transition": "000-000 P(3)",
            "nu": "1234.56789",
            "sw": "1.2e-20",
        },
        {
            "id": "HITRAN:raw:missing-transition",
            "molecule": "CO2",
            "unmapped_column": "retained",
        },
    ]

    bundle = build_hitran_wave0_bundle(
        raw_records,
        source_version="HITRAN fixture",
        retrieved_at="2026-04-30T00:00:00Z",
        snapshot_id=SNAPSHOT_ID,
    )
    bundle_again = build_hitran_wave0_bundle(
        raw_records,
        source_version="HITRAN fixture",
        retrieved_at="2026-04-30T00:00:00Z",
        snapshot_id=SNAPSHOT_ID,
    )

    snapshot = bundle.snapshot_row
    assert snapshot["source_system"] == "hitran"
    assert snapshot["adapter_name"] == "sciona.physics_ingest.sources.hitran"
    assert snapshot["payload"]["record_count"] == 2
    assert snapshot["payload"]["mapped_property_count"] == 2
    assert snapshot["payload"]["raw_records"][1]["unmapped_column"] == "retained"
    assert re.fullmatch(r"[0-9a-f]{64}", str(snapshot["payload_sha256"]))
    assert snapshot["payload_sha256"] == bundle_again.snapshot_row["payload_sha256"]

    first = bundle.candidate_rows[0]
    assert first["snapshot_id"] == SNAPSHOT_ID
    assert first["raw_formula"] == "H2O | 1H2-16O | 000-000 P(3) | 1234.56789 cm^-1"
    assert first["source_payload"]["property_mappings"]["wavenumber"] == {
        "value": "1234.56789",
        "unit": "cm^-1",
        "dim_signature": "L-1",
        "mapping_status": "dimension_mapped",
    }

    second = bundle.candidate_rows[1]
    assert second["source_candidate_id"] == "HITRAN:raw:missing-transition"
    assert second["source_payload"]["raw_record"]["unmapped_column"] == "retained"
    assert len(bundle.data_artifact_seeds) == 2
    assert bundle.data_artifact_seeds[0]["artifact_kind"] == "data_artifact"
    assert bundle.data_artifact_seeds[0]["fqdn"] == "hitran.line.HITRAN.H2O.0001"

    staged_snapshot, staged_candidates = stage_source_rows(
        snapshot_row=bundle.snapshot_row,
        candidate_rows=bundle.candidate_rows,
        snapshot_id=SNAPSHOT_ID,
    )
    assert staged_snapshot.source_system == "hitran"
    assert staged_candidates[1].candidate_status == "raw_imported"


def test_materials_project_bundle_keeps_formula_gaps_as_raw_candidates() -> None:
    raw_records = [
        {
            "material_id": "mp-149",
            "formula_pretty": "Si",
            "elements": ["Si"],
            "band_gap": 1.17,
            "density": 2.329,
        },
        {
            "material_id": "mp-missing-formula",
            "properties": {"total_magnetization": {"value": 0.0, "unit": "mu_B"}},
            "source_note": "formula intentionally absent",
        },
    ]

    bundle = build_materials_project_wave0_bundle(
        raw_records,
        source_version="MP fixture",
        snapshot_id=SNAPSHOT_ID,
    )

    assert bundle.snapshot_row["source_system"] == "materials_project"
    assert bundle.snapshot_row["payload"]["formula_record_count"] == 1
    assert bundle.snapshot_row["payload"]["record_count"] == 2

    silicon = bundle.candidate_rows[0]
    assert silicon["raw_formula"] == "Si"
    assert silicon["raw_formula_format"] == "plain_text"
    assert silicon["source_payload"]["property_mappings"]["band_gap"] == {
        "value": "1.17",
        "unit": "eV",
        "dim_signature": "M1L2T-2",
        "mapping_status": "dimension_mapped",
    }
    assert bundle.data_artifact_seeds[0]["fqdn"] == "materials_project.material.mp_149"

    incomplete = bundle.candidate_rows[1]
    assert incomplete["raw_formula"] == ""
    assert incomplete["raw_formula_format"] == ""
    assert incomplete["source_label"] == "mp-missing-formula"
    assert (
        incomplete["source_payload"]["raw_record"]["source_note"]
        == "formula intentionally absent"
    )

    staged_snapshot, staged_candidates = stage_source_rows(
        snapshot_row=bundle.snapshot_row,
        candidate_rows=bundle.candidate_rows,
        snapshot_id=SNAPSHOT_ID,
    )
    assert staged_snapshot.source_system == "materials_project"
    assert staged_candidates[0].raw_formula_format == "plain_text"


def test_opb_bundle_preserves_formula_and_dataset_payloads() -> None:
    raw_records = [
        {
            "problem_id": "opb:newton-2",
            "title": "Newton second law",
            "latex": "F = m a",
            "variables": [
                {"symbol": "F", "role": "output", "dim_signature": "M1L1T-2"},
                {"symbol": "m", "role": "input", "dim_signature": "M1"},
            ],
            "property_mappings": {
                "force": {"value": "F", "unit": "N", "dim_signature": "M1L1T-2"}
            },
            "data": {"fixture_rows": [{"m": 2, "a": 3, "F": 6}]},
        },
        {
            "problem_id": "opb:raw-only",
            "title": "Unparsed benchmark row",
            "data": {"raw": [1, 2, 3]},
        },
    ]

    bundle = build_opb_wave0_bundle(
        raw_records,
        source_version="OPB fixture",
        source_uri="https://example.invalid/opb-fixture",
        snapshot_id=SNAPSHOT_ID,
    )
    bundle_again = build_opb_wave0_bundle(
        raw_records,
        source_version="OPB fixture",
        source_uri="https://example.invalid/opb-fixture",
        snapshot_id=SNAPSHOT_ID,
    )

    assert bundle.snapshot_row["source_system"] == "opb"
    assert bundle.snapshot_row["payload"]["formula_record_count"] == 1
    assert bundle.snapshot_row["payload_sha256"] == bundle_again.snapshot_row[
        "payload_sha256"
    ]

    equation = bundle.candidate_rows[0]
    assert equation["raw_formula"] == "F = m a"
    assert equation["raw_formula_format"] == "latex"
    assert equation["source_payload"]["variables"][0]["symbol"] == "F"
    assert equation["source_payload"]["property_mappings"]["force"] == {
        "value": "F",
        "unit": "N",
        "dim_signature": "M1L1T-2",
        "mapping_status": "dimension_mapped",
    }
    assert bundle.data_artifact_seeds[0]["data_payload"] == {
        "fixture_rows": [{"m": 2, "a": 3, "F": 6}]
    }

    raw_only = bundle.candidate_rows[1]
    assert raw_only["raw_formula"] == ""
    assert raw_only["raw_formula_format"] == ""
    assert raw_only["source_payload"]["raw_record"]["data"] == {"raw": [1, 2, 3]}

    staged_snapshot, staged_candidates = stage_source_rows(
        snapshot_row=bundle.snapshot_row,
        candidate_rows=bundle.candidate_rows,
        snapshot_id=SNAPSHOT_ID,
    )
    assert staged_snapshot.source_system == "opb"
    assert staged_candidates[1].source_candidate_id == "opb:raw-only"
