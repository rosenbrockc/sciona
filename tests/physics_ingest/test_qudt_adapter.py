from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sciona.physics_ingest.sources.qudt import (
    QudtDimensionError,
    build_qudt_symbolic_variable_dimension_updates,
    build_qudt_snapshot_manifest,
    extract_qudt_resource_record,
    parse_qudt_dimension_vector,
    qudt_dimension_vector_to_compact,
)


def test_qudt_dimension_vector_maps_si_axes_to_sciona_compact() -> None:
    assert qudt_dimension_vector_to_compact("A0E0L1I0M1H0T-2D0") == "M1L1T-2"
    assert qudt_dimension_vector_to_compact("A0E-1L2I0M1H0T-3D0") == "M1L2T-3I-1"
    assert qudt_dimension_vector_to_compact("A0E0L0I0M0H1T0D0") == "Th1"
    assert qudt_dimension_vector_to_compact("A1E0L0I0M0H0T0D0") == "N1"
    assert qudt_dimension_vector_to_compact("A0E0L0I1M0H0T0D0") == "J1"
    assert qudt_dimension_vector_to_compact("A0E0L0I0M0H0T0D1") == "1"


def test_qudt_dimension_vector_accepts_uri_and_jsonld_node() -> None:
    uri = "http://qudt.org/vocab/dimensionvector/A0E0L2I0M1H0T-2D0"
    node = {"@id": uri}
    mapping = parse_qudt_dimension_vector(node)

    assert mapping.qudt_vector == "A0E0L2I0M1H0T-2D0"
    assert mapping.qudt_exponents["M"] == 1
    assert mapping.qudt_exponents["L"] == 2
    assert mapping.compact == "M1L2T-2"


def test_qudt_dimension_vector_preserves_fractional_exponent_payloads() -> None:
    decimal = parse_qudt_dimension_vector("A0E0L0.5I0M1H0T-1.5D0")
    fraction = parse_qudt_dimension_vector("A0E0L1/2I0M1H0T-3/2D0")

    assert decimal.qudt_exponents["L"] == fraction.qudt_exponents["L"]
    assert decimal.qudt_exponents["T"] == fraction.qudt_exponents["T"]
    assert decimal.qudt_exponent_payloads["L"] == "0.5"
    assert fraction.qudt_exponent_payloads["L"] == "1/2"
    assert decimal.compact == "M1L1/2T-3/2"
    assert fraction.as_payload()["qudt_exponents"]["T"] == "-3/2"
    assert fraction.as_payload()["qudt_exponent_payloads"]["T"] == "-3/2"


def test_invalid_dimension_vectors_fail_closed() -> None:
    with pytest.raises(QudtDimensionError):
        parse_qudt_dimension_vector("not-a-vector")
    with pytest.raises(QudtDimensionError):
        parse_qudt_dimension_vector(["A0E0L1I0M0H0T0D0", "A0E0L0I0M0H0T1D0"])
    with pytest.raises(QudtDimensionError):
        parse_qudt_dimension_vector("A0E0L1I0M0H0T0D2")


def test_extract_unit_record_reports_unparseable_dimension_without_dropping_record() -> None:
    raw = {
        "@id": "http://qudt.org/vocab/unit/BAD",
        "@type": ["qudt:Unit"],
        "rdfs:label": "Bad unit",
        "qudt:hasDimensionVector": "A0E0LbadI0M0H0T0D0",
    }

    record = extract_qudt_resource_record(raw)
    row = record.as_candidate_row()

    assert record.dimension is None
    assert record.dimension_error is not None
    assert row["candidate_status"] == "raw_imported"
    assert row["source_payload"]["dimension_status"] == "unresolved"
    assert "unsupported QUDT dimension vector syntax" in (
        row["source_payload"]["dimension_error"]["message"]
    )


def test_extract_unit_record_produces_candidate_compatible_row() -> None:
    raw = {
        "@id": "http://qudt.org/vocab/unit/N",
        "@type": ["qudt:Unit"],
        "rdfs:label": [{"@language": "en", "@value": "Newton"}],
        "qudt:symbol": "N",
        "qudt:hasDimensionVector": {
            "@id": "http://qudt.org/vocab/dimensionvector/A0E0L1I0M1H0T-2D0"
        },
        "qudt:hasQuantityKind": [
            {"@id": "http://qudt.org/vocab/quantitykind/Force"}
        ],
    }

    record = extract_qudt_resource_record(raw)
    row = record.as_candidate_row()

    assert record.resource_kind == "unit"
    assert record.dimension is not None
    assert record.dimension.compact == "M1L1T-2"
    assert row["source_candidate_id"] == "http://qudt.org/vocab/unit/N"
    assert row["source_label"] == "Newton"
    assert row["candidate_status"] == "dimension_resolved"
    assert row["raw_formula"] == ""
    assert row["raw_formula_format"] == ""
    assert row["source_payload"]["dim_signature"] == "M1L1T-2"
    assert row["source_payload"]["quantity_kind_uris"] == [
        "http://qudt.org/vocab/quantitykind/Force"
    ]
    assert "not a standalone equation" in row["notes"]


def test_qudt_records_build_side_effect_free_symbolic_variable_dimension_updates() -> None:
    records = [
        extract_qudt_resource_record(
            {
                "@id": "http://qudt.org/vocab/unit/SQRT-M",
                "@type": ["qudt:Unit"],
                "rdfs:label": "square root metre",
                "qudt:symbol": "sqrt_m",
                "qudt:hasDimensionVector": "A0E0L1/2I0M0H0T0D0",
            }
        )
    ]
    variables = [
        {
            "expression_id": "expr-1",
            "symbol_name": "x",
            "source_symbol": "sqrt_m",
            "variable_role": "input",
            "dim_signature": "",
            "dimension_source": "unknown",
            "evidence_json": {"source": "fixture"},
        }
    ]

    updates = build_qudt_symbolic_variable_dimension_updates(records, variables)

    assert variables[0]["dim_signature"] == ""
    assert updates == [
        {
            "expression_id": "expr-1",
            "symbol_name": "x",
            "source_symbol": "sqrt_m",
            "variable_role": "input",
            "dim_signature": "L1/2",
            "dimension_source": "qudt",
            "evidence_json": {
                "source": "fixture",
                "qudt_dimension_resolution": {
                    "source_entity_uri": "http://qudt.org/vocab/unit/SQRT-M",
                    "source_label": "square root metre",
                    "resource_kind": "unit",
                    "qudt_dimension_vector": "A0E0L1/2I0M0H0T0D0",
                    "qudt_exponents": {
                        "A": "0",
                        "E": "0",
                        "L": "1/2",
                        "I": "0",
                        "M": "0",
                        "H": "0",
                        "T": "0",
                        "D": "0",
                    },
                    "qudt_exponent_payloads": {
                        "A": "0",
                        "E": "0",
                        "L": "1/2",
                        "I": "0",
                        "M": "0",
                        "H": "0",
                        "T": "0",
                        "D": "0",
                    },
                    "dim_signature": "L1/2",
                },
            },
            "unit_uri": "http://qudt.org/vocab/unit/SQRT-M",
            "unit_label": "square root metre",
        }
    ]


def test_build_snapshot_manifest_is_wave0_snapshot_compatible_and_deterministic() -> None:
    raw_records = [
        {
            "@id": "http://qudt.org/vocab/quantitykind/Force",
            "@type": "qudt:QuantityKind",
            "rdfs:label": "Force",
            "qudt:hasDimensionVector": {
                "@id": "http://qudt.org/vocab/dimensionvector/A0E0L1I0M1H0T-2D0"
            },
            "qudt:applicableUnit": [{"@id": "http://qudt.org/vocab/unit/N"}],
        },
        {
            "@id": "http://qudt.org/vocab/unit/V",
            "@type": "qudt:Unit",
            "rdfs:label": "Volt",
            "qudt:symbol": "V",
            "qudt:hasDimensionVector": "A0E-1L2I0M1H0T-3D0",
        },
    ]
    retrieved_at = datetime(2026, 4, 30, 18, 0, tzinfo=timezone.utc)

    manifest = build_qudt_snapshot_manifest(
        raw_records,
        source_version="qudt-2.1-test",
        source_uri="https://qudt.org/test",
        retrieved_at=retrieved_at,
        license_expression="test-license",
    )
    manifest_again = build_qudt_snapshot_manifest(
        raw_records,
        source_version="qudt-2.1-test",
        source_uri="https://qudt.org/test",
        retrieved_at=retrieved_at,
        license_expression="test-license",
    )

    snapshot = manifest.snapshot_row
    assert snapshot["source_system"] == "qudt"
    assert snapshot["source_version"] == "qudt-2.1-test"
    assert snapshot["adapter_name"] == "sciona.physics_ingest.sources.qudt"
    assert len(snapshot["payload_sha256"]) == 64
    assert snapshot["payload_sha256"] == manifest_again.snapshot_row["payload_sha256"]
    assert snapshot["payload"]["record_count"] == 2
    assert snapshot["payload"]["dimension_record_count"] == 2
    assert snapshot["payload"]["dimension_error_count"] == 0

    rows = manifest.candidate_rows
    assert len(rows) == 2
    assert rows[0]["source_payload"]["resource_kind"] == "quantity_kind"
    assert rows[0]["source_payload"]["unit_uris"] == ["http://qudt.org/vocab/unit/N"]
    assert rows[1]["source_payload"]["dim_signature"] == "M1L2T-3I-1"
