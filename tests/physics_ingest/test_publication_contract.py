from __future__ import annotations

from sciona.physics_ingest.publication import load_symbolic_publication_manifest


ARTIFACT_A = "20000000-0000-0000-0000-000000000001"
VERSION_A = "30000000-0000-0000-0000-000000000001"
ARTIFACT_B = "20000000-0000-0000-0000-000000000002"
VERSION_B = "30000000-0000-0000-0000-000000000002"


def test_load_symbolic_publication_manifest_resolves_and_validates_insert_rows() -> None:
    manifest = _manifest()

    result = load_symbolic_publication_manifest(
        manifest,
        {
            "local:sciona-atoms-physics:fixture.force": {
                "artifact_id": ARTIFACT_A,
                "version_id": VERSION_A,
            },
            "angle_atom": {
                "artifact_id": ARTIFACT_B,
                "version_id": VERSION_B,
            },
        },
    )

    rows = result.to_insert_rows()
    expressions = rows["artifact_symbolic_expressions"]
    variables = rows["artifact_symbolic_variables"]
    bounds = rows["artifact_validity_bounds"]

    assert result.diagnostics == ()
    assert len(expressions) == 2
    assert expressions[0]["artifact_id"] == ARTIFACT_A
    assert expressions[0]["version_id"] == VERSION_A
    assert expressions[0]["sympy_srepr"] == "Equality(Symbol('F'), Mul(Symbol('m'), Symbol('a')))"
    assert expressions[0]["source_expression_id"] == "local:sciona-atoms-physics:fixture.force"
    assert expressions[0]["parse_status"] == "normalized"
    assert expressions[0]["review_status"] == "automated_pass"
    assert expressions[0]["validation_status"] == "passed"
    assert expressions[0]["evidence_json"]["publication_manifest"]["provider"] == (
        "sciona-atoms-physics"
    )
    assert expressions[0]["evidence_json"]["publication_manifest"]["constants"] == {
        "unit": 1.0
    }

    assert {row["symbol_name"] for row in variables} == {"F", "m", "a", "theta"}
    force_variable = next(row for row in variables if row["symbol_name"] == "F")
    assert force_variable["expression_id"] == expressions[0]["expression_id"]
    assert force_variable["variable_role"] == "output"
    assert force_variable["dim_signature"] == "M1L1T-2"
    assert force_variable["dimension_source"] == "source"
    assert "artifact_id" not in force_variable
    assert force_variable["evidence_json"]["publication_manifest"]["artifact_id"] == (
        ARTIFACT_A
    )

    mass_bound = next(row for row in bounds if row["variable_name"] == "m")
    assert mass_bound["expression_id"] == expressions[0]["expression_id"]
    assert mass_bound["artifact_id"] == ARTIFACT_A
    assert mass_bound["lower_value"] == 0.0
    assert "upper_value" not in mass_bound
    assert mass_bound["validity_statement"] == "m >= 0.0"


def test_publication_loader_reports_missing_bindings_without_db_calls() -> None:
    result = load_symbolic_publication_manifest(
        _manifest(),
        {
            "angle_atom": {
                "artifact_id": ARTIFACT_B,
                "version_id": VERSION_B,
            },
        },
    )

    rows = result.to_insert_rows()
    assert [row["artifact_id"] for row in rows["artifact_symbolic_expressions"]] == [
        ARTIFACT_B
    ]
    assert {row.reason for row in result.skipped_rows} == {
        "missing_artifact_binding",
        "missing_expression_binding",
    }
    assert [
        (row.table, row.artifact_key, row.atom_name)
        for row in result.skipped_rows
        if row.reason == "missing_artifact_binding"
    ] == [
        (
            "artifact_symbolic_expressions",
            "local:sciona-atoms-physics:fixture.force",
            "force_atom",
        )
    ]
    assert result.error_rows == ()


def test_publication_loader_reports_validation_errors_and_excludes_bad_rows() -> None:
    manifest = _manifest()
    manifest["artifact_symbolic_variables"] = [
        *manifest["artifact_symbolic_variables"],
        {
            "artifact_key": "local:sciona-atoms-physics:fixture.force",
            "provider": "sciona-atoms-physics",
            "atom_name": "force_atom",
            "symbol": "bad",
            "role": "not_a_role",
            "dim_signature": "",
        },
    ]

    result = load_symbolic_publication_manifest(
        manifest,
        {
            "local:sciona-atoms-physics:fixture.force": {
                "artifact_id": ARTIFACT_A,
                "version_id": VERSION_A,
            },
            "angle_atom": {
                "artifact_id": ARTIFACT_B,
                "version_id": VERSION_B,
            },
        },
    )

    assert len(result.to_insert_rows()["artifact_symbolic_variables"]) == 4
    assert [(row.table, row.reason, row.severity) for row in result.error_rows] == [
        ("artifact_symbolic_variables", "validation_error", "error")
    ]
    assert "variable_role" in result.error_rows[0].detail


def _manifest() -> dict[str, object]:
    return {
        "provider": "sciona-atoms-physics",
        "modules": ["fixture"],
        "artifact_symbolic_expressions": [
            {
                "artifact_key": "local:sciona-atoms-physics:fixture.force",
                "local_artifact_key": "local:sciona-atoms-physics:fixture.force",
                "provider": "sciona-atoms-physics",
                "atom_name": "force_atom",
                "atom_module": "fixture.force",
                "registry_name": "force_atom",
                "expression_srepr": "Equality(Symbol('F'), Mul(Symbol('m'), Symbol('a')))",
                "expression_text": "Eq(F, a*m)",
                "variables": {"F": "output", "m": "input", "a": "input"},
                "dim_signature": {"F": "M1L1T-2", "m": "M1", "a": "L1T-2"},
                "symbolic_dim_signature": {
                    "F": "M1L1T-2",
                    "m": "M1",
                    "a": "L1T-2",
                },
                "constants": {"unit": 1.0},
                "bibliography": [{"title": "fixture"}],
                "artifact_uuid": None,
            },
            {
                "artifact_key": "local:sciona-atoms-physics:fixture.angle",
                "local_artifact_key": "local:sciona-atoms-physics:fixture.angle",
                "provider": "sciona-atoms-physics",
                "atom_name": "angle_atom",
                "atom_module": "fixture.angle",
                "registry_name": "angle_atom",
                "expression_srepr": "Symbol('theta')",
                "expression_text": "theta",
                "variables": {"theta": "output"},
                "dim_signature": {"theta": ""},
                "symbolic_dim_signature": {"theta": ""},
                "constants": {},
                "bibliography": [],
                "artifact_uuid": None,
            },
        ],
        "artifact_symbolic_variables": [
            {
                "artifact_key": "local:sciona-atoms-physics:fixture.force",
                "provider": "sciona-atoms-physics",
                "atom_name": "force_atom",
                "symbol": "F",
                "role": "output",
                "dim_signature": "M1L1T-2",
            },
            {
                "artifact_key": "local:sciona-atoms-physics:fixture.force",
                "provider": "sciona-atoms-physics",
                "atom_name": "force_atom",
                "symbol": "m",
                "role": "input",
                "dim_signature": "M1",
            },
            {
                "artifact_key": "local:sciona-atoms-physics:fixture.force",
                "provider": "sciona-atoms-physics",
                "atom_name": "force_atom",
                "symbol": "a",
                "role": "input",
                "dim_signature": "L1T-2",
            },
            {
                "artifact_key": "local:sciona-atoms-physics:fixture.angle",
                "provider": "sciona-atoms-physics",
                "atom_name": "angle_atom",
                "symbol": "theta",
                "role": "output",
                "dim_signature": "",
            },
        ],
        "artifact_validity_bounds": [
            {
                "artifact_key": "local:sciona-atoms-physics:fixture.force",
                "provider": "sciona-atoms-physics",
                "atom_name": "force_atom",
                "symbol": "m",
                "min_value": 0.0,
                "max_value": None,
            },
            {
                "artifact_key": "local:sciona-atoms-physics:fixture.angle",
                "provider": "sciona-atoms-physics",
                "atom_name": "angle_atom",
                "symbol": "theta",
                "min_value": 0.0,
                "max_value": 3.14159,
            },
        ],
    }
