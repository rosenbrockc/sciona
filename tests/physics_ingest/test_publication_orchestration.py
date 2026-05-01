from __future__ import annotations

from sciona.physics_ingest.orchestration import orchestrate_physics_publication


SNAPSHOT_ID = "00000000-0000-0000-0000-000000000001"
ARTIFACT_ID = "20000000-0000-0000-0000-000000000001"
VERSION_ID = "30000000-0000-0000-0000-000000000001"


def test_orchestrator_groups_validated_source_and_publication_insert_rows() -> None:
    bundle = _source_bundle()
    original_candidate = dict(bundle["candidate_rows"][0])

    result = orchestrate_physics_publication(
        source_bundles=[bundle],
        publication_manifests=[_publication_manifest()],
        artifact_bindings={
            "local:fixture.force": {
                "artifact_id": ARTIFACT_ID,
                "version_id": VERSION_ID,
            }
        },
        snapshot_id_bindings={"fixture-bundle": SNAPSHOT_ID},
    )

    rows = result.to_insert_rows()
    assert result.diagnostics == ()
    assert set(rows) == {
        "artifact_symbolic_expressions",
        "artifact_symbolic_variables",
        "physics_equation_candidates",
        "physics_ingest_snapshots",
    }
    assert rows["physics_ingest_snapshots"][0]["source_system"] == "manual"
    assert rows["physics_equation_candidates"][0]["snapshot_id"] == SNAPSHOT_ID
    assert rows["artifact_symbolic_expressions"][0]["artifact_id"] == ARTIFACT_ID
    assert rows["artifact_symbolic_variables"][0]["symbol_name"] == "F"
    assert result.audit_summary.to_dict() == {
        "source_bundle_count": 1,
        "publication_manifest_count": 1,
        "input_row_counts": {
            "artifact_symbolic_expressions": 1,
            "artifact_symbolic_variables": 1,
            "artifact_validity_bounds": 0,
            "physics_equation_candidates": 1,
            "physics_ingest_snapshots": 1,
        },
        "insert_row_counts": {
            "artifact_symbolic_expressions": 1,
            "artifact_symbolic_variables": 1,
            "physics_equation_candidates": 1,
            "physics_ingest_snapshots": 1,
        },
        "skipped_row_count": 0,
        "error_row_count": 0,
        "has_errors": False,
    }
    assert bundle["candidate_rows"][0] == original_candidate


def test_orchestrator_skips_candidates_without_explicit_snapshot_binding() -> None:
    result = orchestrate_physics_publication(source_bundles=[_source_bundle()])

    rows = result.to_insert_rows()
    assert "physics_ingest_snapshots" in rows
    assert "physics_equation_candidates" not in rows
    assert [(row.table, row.reason, row.severity) for row in result.skipped_rows] == [
        ("physics_equation_candidates", "missing_snapshot_binding", "skipped")
    ]
    assert result.audit_summary.skipped_row_count == 1
    assert result.audit_summary.error_row_count == 0


def test_orchestrator_keeps_publication_validation_errors_non_fatal() -> None:
    manifest = _publication_manifest()
    manifest["artifact_symbolic_variables"] = [
        {
            "artifact_key": "local:fixture.force",
            "atom_name": "force_atom",
            "symbol": "F",
            "role": "unsupported",
        }
    ]

    result = orchestrate_physics_publication(
        publication_manifests=[manifest],
        artifact_bindings={
            "local:fixture.force": {
                "artifact_id": ARTIFACT_ID,
                "version_id": VERSION_ID,
            }
        },
    )

    rows = result.to_insert_rows()
    assert len(rows["artifact_symbolic_expressions"]) == 1
    assert "artifact_symbolic_variables" not in rows
    assert [(row.table, row.reason, row.severity) for row in result.error_rows] == [
        ("artifact_symbolic_variables", "validation_error", "error")
    ]
    assert "variable_role" in result.error_rows[0].detail
    assert result.audit_summary.has_errors is True


def _source_bundle() -> dict[str, object]:
    return {
        "bundle_key": "fixture-bundle",
        "snapshot_row": {
            "source_system": "manual",
            "source_version": "fixture-v1",
            "adapter_name": "fixture.adapter",
            "payload_sha256": "a" * 64,
            "payload": {"record_count": 1},
        },
        "candidate_rows": [
            {
                "source_candidate_id": "fixture:eq:force",
                "source_label": "Newton second law",
                "raw_formula": "F = m a",
                "raw_formula_format": "plain_text",
                "candidate_status": "raw_imported",
                "parse_confidence": 0.5,
                "source_payload": {"fixture": True},
            }
        ],
    }


def _publication_manifest() -> dict[str, object]:
    return {
        "provider": "fixture",
        "artifact_symbolic_expressions": [
            {
                "artifact_key": "local:fixture.force",
                "local_artifact_key": "local:fixture.force",
                "atom_name": "force_atom",
                "registry_name": "force_atom",
                "expression_srepr": "Equality(Symbol('F'), Mul(Symbol('m'), Symbol('a')))",
                "expression_text": "Eq(F, a*m)",
            }
        ],
        "artifact_symbolic_variables": [
            {
                "artifact_key": "local:fixture.force",
                "atom_name": "force_atom",
                "symbol": "F",
                "role": "output",
            }
        ],
        "artifact_validity_bounds": [],
    }
