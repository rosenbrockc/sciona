"""Tests for physics ingest symbolic normalization drafts."""

from __future__ import annotations

import copy

import sympy as sp

from sciona.physics_ingest.normalization import (
    normalize_candidate_expression_draft,
    normalize_candidate_expression_drafts,
    normalize_candidate_expression_draft_with_qudt_dimensions,
    resolve_candidate_variable_dimensions_from_qudt,
)


ARTIFACT_ID = "00000000-0000-0000-0000-000000000101"
VERSION_ID = "00000000-0000-0000-0000-000000000102"
CANDIDATE_ID = "00000000-0000-0000-0000-000000000103"


def test_plain_text_candidate_normalizes_to_expression_row_draft() -> None:
    draft = normalize_candidate_expression_draft(
        {
            "candidate_id": CANDIDATE_ID,
            "source_candidate_id": "fixture-force",
            "raw_formula": "F = m a",
            "raw_formula_format": "plain_text",
            "variables": {
                "F": {"role": "output", "dim_signature": "M1L1T-2"},
                "m": {"role": "input", "dim_signature": "M1"},
                "a": {"role": "input", "dim_signature": "L1T-2"},
            },
            "mechanism_tags": ["newtonian"],
        },
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
        require_dimensions=True,
    )

    row = draft.row

    assert row.candidate_id == CANDIDATE_ID
    assert row.source_expression_id == "fixture-force"
    assert row.parse_status == "normalized"
    assert row.parse_confidence == 0.95
    assert row.review_status == "automated_pass"
    assert row.sympy_srepr.startswith("Equality(")
    assert row.canonical_expr_hash
    assert row.topology_hash
    assert row.dimensional_hash
    assert row.mechanism_tags == ["newtonian"]
    assert row.evidence_json["parse_roundtrip"]["status"] == "passed"
    assert row.evidence_json["normalization"]["review_tasks"] == []


def test_latex_candidate_uses_local_parser_and_preserves_raw_formula() -> None:
    draft = normalize_candidate_expression_draft(
        {
            "source_candidate_id": "fixture-latex-energy",
            "raw_formula": r"E = m c^2",
            "raw_formula_format": "latex",
        },
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
    )

    assert draft.row.parse_status == "normalized"
    assert draft.row.raw_formula == r"E = m c^2"
    assert draft.row.raw_formula_format == "latex"
    assert draft.row.evidence_json["parse_roundtrip"]["status"] == "passed"
    assert "latex_parsed_locally" in {
        diagnostic.code for diagnostic in draft.diagnostics
    }


def test_sympy_expression_candidate_normalizes_without_text_preparse() -> None:
    draft = normalize_candidate_expression_draft(
        {
            "source_candidate_id": "fixture-sympy",
            "sympy_expr": sp.Eq(sp.Symbol("v"), sp.Symbol("x") / sp.Symbol("t")),
            "raw_formula": "Eq(v, x/t)",
            "raw_formula_format": "sympy",
        },
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
    )

    assert draft.row.parse_status == "normalized"
    assert draft.row.review_status == "needs_human"
    assert draft.row.canonical_expr_hash
    assert draft.row.topology_hash
    assert {diagnostic.code for diagnostic in draft.diagnostics} == {
        "missing_dimension"
    }
    dimensions = draft.row.evidence_json["normalization"]["dimensions"]
    assert dimensions["unknown_dimensions"] == {
        "symbols": ["t", "v", "x"],
        "count": 3,
        "review_task_codes": [
            "missing_dimension",
            "missing_dimension",
            "missing_dimension",
        ],
        "review_task_code_counts": {"missing_dimension": 3},
    }
    assert draft.row.evidence_json["normalization"]["review_task_codes"] == [
        "missing_dimension",
        "missing_dimension",
        "missing_dimension",
    ]


def test_fractional_dimension_signatures_are_reflected_in_evidence() -> None:
    draft = normalize_candidate_expression_draft(
        {
            "source_candidate_id": "fixture-rational-dimensions",
            "raw_formula": "v = x",
            "raw_formula_format": "plain_text",
            "variables": {
                "v": {"role": "output", "dim_signature": "M1L1/2T-1"},
                "x": {"role": "input", "dim_signature": "L1/2"},
            },
        },
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
        require_dimensions=True,
    )

    dimensions = draft.row.evidence_json["normalization"]["dimensions"]

    assert draft.row.parse_status == "normalized"
    assert dimensions["unknown_dimensions"]["count"] == 0
    assert dimensions["provided_dimensions"]["signatures"] == [
        {
            "symbol": "v",
            "dim_signature": "M1L1/2T-1",
            "is_unknown": False,
            "is_rational": True,
        },
        {
            "symbol": "x",
            "dim_signature": "L1/2",
            "is_unknown": False,
            "is_rational": True,
        },
    ]
    assert dimensions["rational_dimensions"] == {
        "symbols": ["v", "x"],
        "count": 2,
        "signatures": dimensions["provided_dimensions"]["signatures"],
    }


def test_explicit_unknown_dimension_signature_is_reflected_in_evidence() -> None:
    draft = normalize_candidate_expression_draft(
        {
            "source_candidate_id": "fixture-explicit-unknown-dimension",
            "raw_formula": "u = v",
            "raw_formula_format": "plain_text",
            "variables": {
                "u": {"role": "output", "dim_signature": "?"},
                "v": {"role": "input", "dim_signature": "1"},
            },
        },
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
        require_dimensions=True,
    )

    dimensions = draft.row.evidence_json["normalization"]["dimensions"]

    assert dimensions["unknown_dimensions"] == {
        "symbols": ["u"],
        "count": 1,
        "review_task_codes": [],
        "review_task_code_counts": {},
    }
    assert dimensions["provided_dimensions"]["signatures"] == [
        {
            "symbol": "u",
            "dim_signature": "?",
            "is_unknown": True,
            "is_rational": False,
        },
        {
            "symbol": "v",
            "dim_signature": "1",
            "is_unknown": False,
            "is_rational": False,
        },
    ]


def test_qudt_assisted_normalization_resolves_candidate_dimensions() -> None:
    candidate = {
        "source_candidate_id": "fixture-qudt-force",
        "raw_formula": "F = m * a",
        "raw_formula_format": "plain_text",
        "variables": {
            "F": {
                "role": "output",
                "quantity_kind_uri": "http://qudt.org/vocab/quantitykind/Force",
                "dim_signature": "",
            },
            "m": {"role": "input", "dim_signature": "M1"},
            "a": {"role": "input", "dim_signature": "L1T-2"},
        },
    }
    draft = normalize_candidate_expression_draft_with_qudt_dimensions(
        candidate,
        qudt_records=[
            {
                "@id": "http://qudt.org/vocab/quantitykind/Force",
                "@type": ["qudt:QuantityKind"],
                "rdfs:label": "Force",
                "qudt:hasDimensionVector": "A0E0L1I0M1H0T-2D0",
            }
        ],
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
        require_dimensions=True,
    )

    dimensions = draft.row.evidence_json["normalization"]["dimensions"]

    assert candidate["variables"]["F"]["dim_signature"] == ""
    assert draft.row.parse_status == "normalized"
    assert draft.row.review_status == "automated_pass"
    assert "qudt_dimension_resolved" in {
        diagnostic.code for diagnostic in draft.diagnostics
    }
    assert dimensions["unknown_dimensions"]["count"] == 0
    assert {
        entry["symbol"]: entry["dim_signature"]
        for entry in dimensions["provided_dimensions"]["signatures"]
    } == {"F": "M1L1T-2", "a": "L1T-2", "m": "M1"}


def test_qudt_assisted_normalization_resolves_unique_unit_code_alias() -> None:
    draft = normalize_candidate_expression_draft_with_qudt_dimensions(
        {
            "source_candidate_id": "fixture-qudt-unit-code-alias",
            "raw_formula": "F = m * a",
            "raw_formula_format": "plain_text",
            "variables": {
                "F": {"role": "output", "unit_code": "new", "dim_signature": ""},
                "m": {"role": "input", "dim_signature": "M1"},
                "a": {"role": "input", "dim_signature": "L1T-2"},
            },
        },
        qudt_records=[
            {
                "@id": "http://qudt.org/vocab/unit/N",
                "@type": ["qudt:Unit"],
                "rdfs:label": "Newton",
                "qudt:symbol": "N",
                "qudt:uneceCommonCode": "NEW",
                "qudt:hasDimensionVector": "A0E0L1I0M1H0T-2D0",
            }
        ],
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
        require_dimensions=True,
    )

    dimensions = draft.row.evidence_json["normalization"]["dimensions"]

    assert draft.row.parse_status == "normalized"
    assert draft.row.review_status == "automated_pass"
    assert "qudt_dimension_resolved" in {
        diagnostic.code for diagnostic in draft.diagnostics
    }
    assert {
        entry["symbol"]: entry["dim_signature"]
        for entry in dimensions["provided_dimensions"]["signatures"]
    } == {"F": "M1L1T-2", "a": "L1T-2", "m": "M1"}


def test_qudt_assisted_normalization_resolves_quantity_kind_label_alias() -> None:
    draft = normalize_candidate_expression_draft_with_qudt_dimensions(
        {
            "source_candidate_id": "fixture-qudt-quantity-kind-label-alias",
            "raw_formula": "J = x",
            "raw_formula_format": "plain_text",
            "variables": {
                "J": {
                    "role": "output",
                    "quantity_kind_name": "angular momentum",
                    "dim_signature": "",
                },
                "x": {"role": "input", "dim_signature": "M1L2T-1"},
            },
        },
        qudt_records=[
            {
                "@id": "http://qudt.org/vocab/quantitykind/AngularMomentum",
                "@type": ["qudt:QuantityKind"],
                "rdfs:label": "Angular-Momentum",
                "qudt:hasDimensionVector": "A0E0L2I0M1H0T-1D0",
            }
        ],
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
        require_dimensions=True,
    )

    dimensions = draft.row.evidence_json["normalization"]["dimensions"]

    assert draft.row.parse_status == "normalized"
    assert draft.row.review_status == "automated_pass"
    assert {
        entry["symbol"]: entry["dim_signature"]
        for entry in dimensions["provided_dimensions"]["signatures"]
    } == {"J": "M1L2T-1", "x": "M1L2T-1"}


def test_qudt_assisted_normalization_resolves_unit_uri_node_alias() -> None:
    draft = normalize_candidate_expression_draft_with_qudt_dimensions(
        {
            "source_candidate_id": "fixture-qudt-unit-node-alias",
            "raw_formula": "a = x",
            "raw_formula_format": "plain_text",
            "variables": {
                "a": {
                    "role": "output",
                    "unit": {"@id": "http://qudt.org/vocab/unit/M-PER-SEC2"},
                    "dim_signature": "",
                },
                "x": {"role": "input", "dim_signature": "L1T-2"},
            },
        },
        qudt_records=[
            {
                "@id": "http://qudt.org/vocab/unit/M-PER-SEC2",
                "@type": ["qudt:Unit"],
                "rdfs:label": "Metre per Square Second",
                "qudt:hasDimensionVector": "A0E0L1I0M0H0T-2D0",
            }
        ],
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
        require_dimensions=True,
    )

    dimensions = draft.row.evidence_json["normalization"]["dimensions"]

    assert draft.row.parse_status == "normalized"
    assert draft.row.review_status == "automated_pass"
    assert {
        entry["symbol"]: entry["dim_signature"]
        for entry in dimensions["provided_dimensions"]["signatures"]
    } == {"a": "L1T-2", "x": "L1T-2"}


def test_qudt_assisted_normalization_resolves_long_tail_unit_code_aliases() -> None:
    draft = normalize_candidate_expression_draft_with_qudt_dimensions(
        {
            "source_candidate_id": "fixture-qudt-long-tail-unit-code-alias",
            "raw_formula": "P = F",
            "raw_formula_format": "plain_text",
            "variables": {
                "P": {
                    "role": "output",
                    "unit_ucum_code": "N.m/s",
                    "dim_signature": "",
                },
                "F": {"role": "input", "dim_signature": "M1L2T-3"},
            },
        },
        qudt_records=[
            {
                "@id": "http://qudt.org/vocab/unit/W",
                "@type": ["qudt:Unit"],
                "rdfs:label": "Watt",
                "qudt:symbol": "W",
                "qudt:ucumCode": [{"@value": "N.m/s"}],
                "qudt:abbreviation": "W",
                "qudt:hasDimensionVector": "A0E0L2I0M1H0T-3D0",
            }
        ],
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
        require_dimensions=True,
    )

    dimensions = draft.row.evidence_json["normalization"]["dimensions"]

    assert draft.row.parse_status == "normalized"
    assert draft.row.review_status == "automated_pass"
    assert {
        entry["symbol"]: entry["dim_signature"]
        for entry in dimensions["provided_dimensions"]["signatures"]
    } == {"F": "M1L2T-3", "P": "M1L2T-3"}


def test_qudt_assisted_normalization_resolves_source_unit_text_aliases() -> None:
    draft = normalize_candidate_expression_draft_with_qudt_dimensions(
        {
            "source_candidate_id": "fixture-qudt-source-unit-text-alias",
            "raw_formula": "v = x",
            "raw_formula_format": "plain_text",
            "variables": {
                "v": {
                    "role": "output",
                    "unit_text": "m s^-1",
                    "dim_signature": "",
                },
                "x": {"role": "input", "dim_signature": "L1T-1"},
            },
        },
        qudt_records=[
            {
                "@id": "http://qudt.org/vocab/unit/M-PER-SEC",
                "@type": ["qudt:Unit"],
                "rdfs:label": "Metre per Second",
                "qudt:symbol": "m/s",
                "qudt:hasDimensionVector": "A0E0L1I0M0H0T-1D0",
            }
        ],
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
        require_dimensions=True,
    )

    dimensions = draft.row.evidence_json["normalization"]["dimensions"]

    assert draft.row.parse_status == "normalized"
    assert draft.row.review_status == "automated_pass"
    assert {
        entry["symbol"]: entry["dim_signature"]
        for entry in dimensions["provided_dimensions"]["signatures"]
    } == {"v": "L1T-1", "x": "L1T-1"}


def test_qudt_assisted_normalization_resolves_source_quantity_hint_alias() -> None:
    draft = normalize_candidate_expression_draft_with_qudt_dimensions(
        {
            "source_candidate_id": "fixture-qudt-source-quantity-hint-alias",
            "raw_formula": "h = x",
            "raw_formula_format": "plain_text",
            "variables": {
                "h": {
                    "role": "output",
                    "quantity_kind_hint": "Planck constant",
                    "dim_signature": "",
                },
                "x": {"role": "input", "dim_signature": "M1L2T-1"},
            },
        },
        qudt_records=[
            {
                "@id": "http://qudt.org/vocab/quantitykind/PlanckConstant",
                "@type": ["qudt:QuantityKind"],
                "rdfs:label": "PlanckConstant",
                "qudt:hasDimensionVector": "A0E0L2I0M1H0T-1D0",
            }
        ],
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
        require_dimensions=True,
    )

    dimensions = draft.row.evidence_json["normalization"]["dimensions"]

    assert draft.row.parse_status == "normalized"
    assert draft.row.review_status == "automated_pass"
    assert {
        entry["symbol"]: entry["dim_signature"]
        for entry in dimensions["provided_dimensions"]["signatures"]
    } == {"h": "M1L2T-1", "x": "M1L2T-1"}


def test_qudt_assisted_normalization_resolves_quantity_kind_alt_label() -> None:
    draft = normalize_candidate_expression_draft_with_qudt_dimensions(
        {
            "source_candidate_id": "fixture-qudt-quantity-kind-alt-label",
            "raw_formula": "c_p = y",
            "raw_formula_format": "plain_text",
            "variables": {
                "c_p": {
                    "role": "output",
                    "physical_quantity": "specific heat capacity at constant pressure",
                    "dim_signature": "",
                },
                "y": {"role": "input", "dim_signature": "L2T-2Th-1"},
            },
        },
        qudt_records=[
            {
                "@id": "http://qudt.org/vocab/quantitykind/SpecificHeatCapacity",
                "@type": ["qudt:QuantityKind"],
                "rdfs:label": "Specific Heat Capacity",
                "skos:altLabel": [
                    {"@value": "Specific Heat Capacity at Constant Pressure"}
                ],
                "qudt:hasDimensionVector": "A0E0L2I0M0H-1T-2D0",
            }
        ],
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
        require_dimensions=True,
    )

    dimensions = draft.row.evidence_json["normalization"]["dimensions"]

    assert draft.row.parse_status == "normalized"
    assert draft.row.review_status == "automated_pass"
    assert {
        entry["symbol"]: entry["dim_signature"]
        for entry in dimensions["provided_dimensions"]["signatures"]
    } == {"c_p": "L2T-2Th-1", "y": "L2T-2Th-1"}


def test_unresolved_qudt_dimensions_remain_reviewable_not_dimensionless() -> None:
    draft = normalize_candidate_expression_draft_with_qudt_dimensions(
        {
            "source_candidate_id": "fixture-unresolved-qudt",
            "raw_formula": "F = m * a",
            "raw_formula_format": "plain_text",
            "variables": {
                "F": {
                    "role": "output",
                    "quantity_kind_uri": "http://qudt.org/vocab/quantitykind/Force",
                    "dim_signature": "",
                },
                "m": {"role": "input", "dim_signature": "M1"},
                "a": {"role": "input", "dim_signature": "L1T-2"},
            },
        },
        qudt_records=[
            {
                "@id": "http://qudt.org/vocab/quantitykind/Force",
                "@type": ["qudt:QuantityKind"],
                "rdfs:label": "Force",
                "qudt:hasDimensionVector": "A0E0LbadI0M0H0T0D0",
            }
        ],
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
        require_dimensions=True,
    )

    dimensions = draft.row.evidence_json["normalization"]["dimensions"]

    assert draft.row.parse_status == "normalized"
    assert draft.row.review_status == "needs_human"
    assert "qudt_dimension_unresolved" in {
        diagnostic.code for diagnostic in draft.diagnostics
    }
    assert "missing_required_dimension" in {
        diagnostic.code for diagnostic in draft.diagnostics
    }
    assert dimensions["unknown_dimensions"]["symbols"] == ["F"]
    assert {
        entry["symbol"]: entry["dim_signature"]
        for entry in dimensions["provided_dimensions"]["signatures"]
    } == {"a": "L1T-2", "m": "M1"}


def test_qudt_assisted_normalization_preserves_rational_exponents() -> None:
    draft = normalize_candidate_expression_draft_with_qudt_dimensions(
        {
            "source_candidate_id": "fixture-qudt-rational",
            "raw_formula": "y = x",
            "raw_formula_format": "plain_text",
            "variables": {
                "y": {"role": "output", "dim_signature": "L1/2"},
                "x": {
                    "role": "input",
                    "unit_uri": "http://qudt.org/vocab/unit/SQRT-M",
                    "dim_signature": "",
                },
            },
        },
        qudt_records=[
            {
                "@id": "http://qudt.org/vocab/unit/SQRT-M",
                "@type": ["qudt:Unit"],
                "rdfs:label": "square root metre",
                "qudt:symbol": "sqrt_m",
                "qudt:hasDimensionVector": "A0E0L1/2I0M0H0T0D0",
            }
        ],
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
        require_dimensions=True,
    )

    dimensions = draft.row.evidence_json["normalization"]["dimensions"]

    assert draft.row.parse_status == "normalized"
    assert dimensions["rational_dimensions"] == {
        "symbols": ["x", "y"],
        "count": 2,
        "signatures": dimensions["provided_dimensions"]["signatures"],
    }
    assert {
        entry["symbol"]: entry["dim_signature"]
        for entry in dimensions["provided_dimensions"]["signatures"]
    } == {"x": "L1/2", "y": "L1/2"}


def test_ambiguous_qudt_dimensions_are_not_applied_to_candidate_copy() -> None:
    assisted = resolve_candidate_variable_dimensions_from_qudt(
        {
            "source_candidate_id": "fixture-ambiguous-qudt",
            "raw_formula": "F = m * a",
            "variables": {
                "F": {"role": "output", "source_symbol": "Force", "dim_signature": ""}
            },
        },
        qudt_records=[
            {
                "@id": "http://qudt.org/vocab/quantitykind/Force",
                "@type": ["qudt:QuantityKind"],
                "rdfs:label": "Force",
                "qudt:hasDimensionVector": "A0E0L1I0M1H0T-2D0",
            },
            {
                "@id": "http://qudt.org/vocab/unit/Force",
                "@type": ["qudt:Unit"],
                "rdfs:label": "Force",
                "qudt:hasDimensionVector": "A0E0L2I0M1H0T-2D0",
            },
        ],
    )

    assert assisted.candidate["variables"]["F"] == {
        "role": "output",
        "source_symbol": "Force",
        "symbol": "F",
    }
    assert [diagnostic.code for diagnostic in assisted.diagnostics] == [
        "qudt_dimension_ambiguous"
    ]


def test_ambiguous_qudt_alias_remains_reviewable_not_dimensionless() -> None:
    draft = normalize_candidate_expression_draft_with_qudt_dimensions(
        {
            "source_candidate_id": "fixture-ambiguous-qudt-alias",
            "raw_formula": "F = m * a",
            "raw_formula_format": "plain_text",
            "variables": {
                "F": {
                    "role": "output",
                    "unit_label": "force unit",
                    "dim_signature": "",
                },
                "m": {"role": "input", "dim_signature": "M1"},
                "a": {"role": "input", "dim_signature": "L1T-2"},
            },
        },
        qudt_records=[
            {
                "@id": "http://qudt.org/vocab/unit/FORCE-A",
                "@type": ["qudt:Unit"],
                "rdfs:label": "Force-Unit",
                "qudt:hasDimensionVector": "A0E0L1I0M1H0T-2D0",
            },
            {
                "@id": "http://qudt.org/vocab/unit/FORCE-B",
                "@type": ["qudt:Unit"],
                "rdfs:label": "Force_Unit",
                "qudt:hasDimensionVector": "A0E0L2I0M1H0T-2D0",
            },
        ],
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
        require_dimensions=True,
    )

    dimensions = draft.row.evidence_json["normalization"]["dimensions"]

    assert draft.row.parse_status == "normalized"
    assert draft.row.review_status == "needs_human"
    assert "qudt_dimension_ambiguous" in {
        diagnostic.code for diagnostic in draft.diagnostics
    }
    assert dimensions["unknown_dimensions"]["symbols"] == ["F"]
    assert {
        entry["symbol"]: entry["dim_signature"]
        for entry in dimensions["provided_dimensions"]["signatures"]
    } == {"a": "L1T-2", "m": "M1"}


def test_ambiguous_normalized_source_unit_alias_remains_reviewable() -> None:
    draft = normalize_candidate_expression_draft_with_qudt_dimensions(
        {
            "source_candidate_id": "fixture-ambiguous-normalized-unit-alias",
            "raw_formula": "v = x",
            "raw_formula_format": "plain_text",
            "variables": {
                "v": {
                    "role": "output",
                    "unit_text": "m s^-1",
                    "dim_signature": "",
                },
                "x": {"role": "input", "dim_signature": "L1T-1"},
            },
        },
        qudt_records=[
            {
                "@id": "http://qudt.org/vocab/unit/M-PER-SEC-A",
                "@type": ["qudt:Unit"],
                "rdfs:label": "Metre per Second A",
                "qudt:symbol": "m/s",
                "qudt:hasDimensionVector": "A0E0L1I0M0H0T-1D0",
            },
            {
                "@id": "http://qudt.org/vocab/unit/M-PER-SEC-B",
                "@type": ["qudt:Unit"],
                "rdfs:label": "Metre per Second B",
                "qudt:symbol": "m/s",
                "qudt:hasDimensionVector": "A0E0L1I0M1H0T-1D0",
            },
        ],
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
        require_dimensions=True,
    )

    dimensions = draft.row.evidence_json["normalization"]["dimensions"]

    assert draft.row.parse_status == "normalized"
    assert draft.row.review_status == "needs_human"
    assert "qudt_dimension_ambiguous" in {
        diagnostic.code for diagnostic in draft.diagnostics
    }
    assert dimensions["unknown_dimensions"]["symbols"] == ["v"]
    assert {
        entry["symbol"]: entry["dim_signature"]
        for entry in dimensions["provided_dimensions"]["signatures"]
    } == {"x": "L1T-1"}


def test_qudt_alias_resolution_does_not_mutate_input_candidate() -> None:
    candidate = {
        "source_candidate_id": "fixture-qudt-non-mutating-alias",
        "raw_formula": "x = y",
        "variables": {
            "x": {
                "role": "output",
                "unit_label": "square-root metre",
                "dim_signature": "",
            },
            "y": {"role": "input", "dim_signature": "L1/2"},
        },
    }
    original = copy.deepcopy(candidate)

    assisted = resolve_candidate_variable_dimensions_from_qudt(
        candidate,
        qudt_records=[
            {
                "@id": "http://qudt.org/vocab/unit/SQRT-M",
                "@type": ["qudt:Unit"],
                "rdfs:label": "Square Root Metre",
                "qudt:hasDimensionVector": "A0E0L1/2I0M0H0T0D0",
            }
        ],
    )

    assert candidate == original
    assert assisted.candidate["variables"]["x"]["dim_signature"] == "L1/2"


def test_parse_failure_returns_needs_human_row_without_hashes() -> None:
    draft = normalize_candidate_expression_draft(
        {
            "candidate_id": "not-a-uuid-source-id",
            "raw_formula": "F =",
            "raw_formula_format": "plain_text",
        },
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
    )

    row = draft.row

    assert row.candidate_id is None
    assert row.source_expression_id == "not-a-uuid-source-id"
    assert row.parse_status == "parse_failed"
    assert row.parse_confidence == 0.0
    assert row.review_status == "needs_human"
    assert row.raw_formula == "F ="
    assert row.sympy_srepr == ""
    assert row.canonical_expr_hash == ""
    assert row.topology_hash == ""
    assert row.evidence_json["parse_roundtrip"]["status"] == "failed"
    assert "parse_failed" in {diagnostic.code for diagnostic in draft.diagnostics}
    assert row.evidence_json["normalization"]["review_task_code_counts"] == {
        "parse_failed": 1
    }
    assert row.evidence_json["normalization"]["dimensions"]["unknown_dimensions"] == {
        "symbols": [],
        "count": 0,
        "review_task_codes": [],
        "review_task_code_counts": {},
    }


def test_batch_normalization_keeps_failed_candidates() -> None:
    drafts = normalize_candidate_expression_drafts(
        [
            {
                "source_candidate_id": "ok",
                "raw_formula": "x = y",
                "raw_formula_format": "plain_text",
            },
            {
                "source_candidate_id": "bad",
                "raw_formula": "x =",
                "raw_formula_format": "plain_text",
            },
        ],
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
    )

    assert [draft.row.source_expression_id for draft in drafts] == ["ok", "bad"]
    assert [draft.row.parse_status for draft in drafts] == [
        "normalized",
        "parse_failed",
    ]
