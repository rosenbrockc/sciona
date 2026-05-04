from __future__ import annotations

import re

from sciona.physics_ingest.sources.nist import (
    DLMFFunctionEntry,
    build_codata_wave0_bundle,
    build_dlmf_wave0_bundle,
    infer_unit_dim_signature,
    normalize_codata_number,
    parse_codata_ascii,
    stable_payload_sha256,
)


def _fixed_width_codata_line(
    label: str,
    value: str,
    uncertainty: str,
    unit: str,
) -> str:
    return f"{label:<60}{value:<25}{uncertainty:<25}{unit}"


def test_parse_codata_ascii_preserves_value_uncertainty_unit_and_refs() -> None:
    text = "\n".join(
        [
            "Quantity                                                    Value                    Uncertainty              Unit",
            _fixed_width_codata_line(
                "speed of light in vacuum",
                "299 792 458",
                "(exact)",
                "m s^-1",
            ),
            (
                "Planck constant | 6.626 070 15 e-34 | (exact) | J Hz^-1 | "
                "symbol=h | reference_ids=NIST-CODATA-2022,h"
            ),
        ]
    )

    constants = parse_codata_ascii(
        text,
        source_version="CODATA 2022",
        symbol_map={"speed of light in vacuum": "c"},
        reference_ids=("NIST-CODATA-2022",),
    )

    assert len(constants) == 2
    speed_of_light = constants[0]
    assert speed_of_light.source_id == "speed-of-light-in-vacuum"
    assert speed_of_light.symbol == "c"
    assert speed_of_light.value_text == "299 792 458"
    assert speed_of_light.normalized_value == "299792458"
    assert speed_of_light.uncertainty_text == "(exact)"
    assert speed_of_light.normalized_uncertainty == "0"
    assert speed_of_light.unit_text == "m s^-1"
    assert speed_of_light.dim_signature == "L1T-1"
    assert speed_of_light.reference_ids == ("NIST-CODATA-2022",)

    planck = constants[1]
    assert planck.symbol == "h"
    assert planck.normalized_value == "6.62607015E-34"
    assert planck.unit_text == "J Hz^-1"
    assert planck.dim_signature == "M1L2T-1"
    assert planck.reference_ids == ("NIST-CODATA-2022", "h")


def test_build_codata_wave0_bundle_matches_wave0_snapshot_and_candidate_shape() -> None:
    constants = parse_codata_ascii(
        "elementary charge | 1.602 176 634 e-19 | (exact) | C | symbol=e",
        source_version="CODATA 2022",
        reference_ids=("NIST-CODATA-2022",),
    )

    bundle = build_codata_wave0_bundle(
        constants,
        source_version="CODATA 2022",
        retrieved_at="2026-04-30T00:00:00Z",
        snapshot_id="00000000-0000-0000-0000-000000000001",
    )

    snapshot = bundle.snapshot_row
    assert snapshot["source_system"] == "nist_codata"
    assert snapshot["source_version"] == "CODATA 2022"
    assert snapshot["retrieved_at"] == "2026-04-30T00:00:00Z"
    assert re.fullmatch(r"[0-9a-f]{64}", str(snapshot["payload_sha256"]))
    assert snapshot["payload"]["constants"][0]["value_text"] == "1.602 176 634 e-19"

    candidate = bundle.candidate_rows[0]
    assert candidate["snapshot_id"] == "00000000-0000-0000-0000-000000000001"
    assert candidate["source_candidate_id"] == "elementary-charge"
    assert candidate["raw_formula"] == "e = 1.602 176 634 e-19 C"
    assert candidate["raw_formula_format"] == "plain_text"
    assert candidate["candidate_status"] == "source_verified"
    assert candidate["source_payload"]["normalized_value"] == "1.602176634E-19"
    assert candidate["source_payload"]["is_exact"] is True
    assert candidate["source_payload"]["ingestion_target_kind"] == "state_artifact"
    assert candidate["source_payload"]["symbolic_equation_candidate"] is False
    assert candidate["source_payload"]["dim_signature_hint"] == "T1I1"
    assert candidate["source_payload"]["reference_ids"] == ["NIST-CODATA-2022"]

    seed = bundle.data_artifact_seeds[0]
    assert seed["artifact_kind"] == "state_artifact"
    assert seed["fqdn"] == "nist.codata.e"
    assert seed["normalized_uncertainty"] == "0"


def test_dlmf_entry_builds_raw_symbolic_function_candidate() -> None:
    entry = DLMFFunctionEntry.from_mapping(
        {
            "source_id": "DLMF:10.2.E2",
            "label": "Bessel equation",
            "formula": r"z^2 w'' + z w' + (z^2-\nu^2)w = 0",
            "formula_format": "latex",
            "description": "Differential equation defining Bessel functions.",
            "variables": [
                {"symbol": "z", "role": "input"},
                {"symbol": r"\nu", "role": "parameter"},
                {"symbol": "w", "role": "state"},
            ],
            "constraints": ["z complex", r"\nu complex"],
            "function_symbols": ["J_nu", "Y_nu"],
            "reference_ids": ["DLMF-10.2.E2"],
        },
        source_version="DLMF 1.2.4",
    )

    bundle = build_dlmf_wave0_bundle(
        [entry],
        source_version="DLMF 1.2.4",
        snapshot_id="00000000-0000-0000-0000-000000000002",
    )

    snapshot = bundle.snapshot_row
    assert snapshot["source_system"] == "nist_dlmf"
    assert snapshot["payload"]["entries"][0]["function_symbols"] == ["J_nu", "Y_nu"]
    assert re.fullmatch(r"[0-9a-f]{64}", str(snapshot["payload_sha256"]))

    candidate = bundle.candidate_rows[0]
    assert candidate["snapshot_id"] == "00000000-0000-0000-0000-000000000002"
    assert candidate["source_candidate_id"] == "DLMF:10.2.E2"
    assert candidate["source_label"] == "Bessel equation"
    assert candidate["raw_formula_format"] == "latex"
    assert candidate["candidate_status"] == "raw_imported"
    assert candidate["mechanism_tags"] == [
        "special_function",
        "mathematical_reference",
    ]
    assert candidate["source_payload"]["reference_ids"] == ["DLMF-10.2.E2"]
    assert candidate["source_payload"]["variables"][1]["symbol"] == r"\nu"


def test_stable_payload_sha256_and_number_normalization_are_deterministic() -> None:
    payload_a = {"b": [2, 1], "a": {"value": "6.626 070 15 e-34"}}
    payload_b = {"a": {"value": "6.626 070 15 e-34"}, "b": [2, 1]}

    assert stable_payload_sha256(payload_a) == stable_payload_sha256(payload_b)
    assert normalize_codata_number("6.626 070 15 e-34") == "6.62607015E-34"
    assert normalize_codata_number("1.234...") == "1.234"


def test_infer_unit_dim_signature_handles_common_codata_compounds() -> None:
    assert infer_unit_dim_signature("m s^-1") == "L1T-1"
    assert infer_unit_dim_signature("J Hz^-1") == "M1L2T-1"
    assert infer_unit_dim_signature("") == "1"
    assert infer_unit_dim_signature("unparsed-unit") == ""
