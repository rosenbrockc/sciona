from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import sciona.physics_ingest.validation as validation_module
from sciona.physics_ingest.validation import (
    VALIDATION_REPORT_KIND,
    VALIDATION_REPORT_VERSION,
    build_physics_ingestion_validation_report,
    discover_changed_pdg_payload_fixture_paths,
    discover_changed_symbolic_fixture_paths,
    discover_pdg_payload_fixture_paths,
    validate_pdg_payload,
    validate_pdg_payload_file,
    validate_source_adapter_coverage,
    validate_source_adapter_data_artifact_seed_quality,
    validate_source_execution_readiness,
    validate_symbolic_publication_fixture,
)
from sciona.physics_ingest.sources import build_physics_source_retrieval_run_plan_dict


REPO_ROOT = Path(__file__).resolve().parents[2]
PDG_FIXTURE_DIR = REPO_ROOT / "tests" / "physics_ingest" / "fixtures" / "pdg_payloads"


def test_symbolic_publication_fixture_validator_accepts_complete_fixture(
    tmp_path,
) -> None:
    fixture_path = tmp_path / "fixture.publication_manifest.json"
    fixture_path.write_text(json.dumps(_symbolic_manifest()), encoding="utf-8")

    check = validate_symbolic_publication_fixture(fixture_path)

    assert check.ok is True
    assert check.issues == ()
    assert check.metadata == {
        "expression_count": 1,
        "variable_count": 3,
        "validity_bound_count": 1,
    }


def test_symbolic_publication_fixture_validator_reports_metadata_gaps(tmp_path) -> None:
    manifest = _symbolic_manifest()
    manifest["artifact_symbolic_expressions"][0]["mechanism_tags"] = []
    manifest["artifact_symbolic_variables"][1]["dim_signature"] = ""
    fixture_path = tmp_path / "fixture.publication_manifest.json"
    fixture_path.write_text(json.dumps(manifest), encoding="utf-8")

    check = validate_symbolic_publication_fixture(fixture_path)

    assert check.ok is False
    assert [issue.reason for issue in check.issues] == [
        "missing_mechanism_tags",
        "missing_dim_signature",
    ]


def test_pdg_payload_validator_accepts_graph_ready_derivation_fixture() -> None:
    check = validate_pdg_payload(_pdg_payload(), subject="fixture-pdg")

    assert check.ok is True
    assert check.issues == ()
    assert check.metadata == {
        "equation_count": 3,
        "inference_edge_count": 2,
        "cdg_candidate_manifest_count": 1,
        "artifact_row_count": 1,
        "artifact_version_row_count": 1,
        "relationship_row_count": 2,
        "cdg_node_count": 2,
        "cdg_edge_count": 1,
        "cdg_binding_count": 4,
    }


def test_pdg_payload_validator_reports_missing_cdg_artifact_envelope_rows() -> None:
    check = validate_pdg_payload(
        _pdg_payload(),
        subject="fixture-pdg",
        cdg_artifact_envelope=None,
    )

    assert check.ok is False
    assert [issue.reason for issue in check.issues] == [
        "pdg_cdg_artifact_envelope_artifacts_missing",
        "pdg_cdg_artifact_envelope_artifact_versions_missing",
    ]
    assert check.metadata["cdg_candidate_manifest_count"] == 1
    assert check.metadata["artifact_row_count"] == 0
    assert check.metadata["artifact_version_row_count"] == 0


def test_pdg_payload_validator_reports_nondeterministic_cdg_artifact_envelope_rows(
    monkeypatch,
) -> None:
    original = validation_module.build_pdg_publication_write_rows
    call_count = 0

    def nondeterministic_publication_rows(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        rows = original(*args, **kwargs)
        if call_count != 2:
            return rows
        insert_rows = rows.to_insert_rows()
        insert_rows["artifacts"][0]["fqdn"] += ".changed"
        return type(rows)(
            insert_rows_by_table={
                table: tuple(table_rows)
                for table, table_rows in insert_rows.items()
            },
            diagnostics=rows.diagnostics,
        )

    monkeypatch.setattr(
        validation_module,
        "build_pdg_publication_write_rows",
        nondeterministic_publication_rows,
    )

    check = validate_pdg_payload(_pdg_payload(), subject="fixture-pdg")

    assert check.ok is False
    assert [issue.reason for issue in check.issues] == [
        "pdg_publication_rows_nondeterministic",
        "pdg_cdg_artifact_envelope_rows_nondeterministic",
    ]


def test_discovers_default_pdg_payload_fixtures() -> None:
    paths = discover_pdg_payload_fixture_paths(REPO_ROOT)

    assert paths == (
        PDG_FIXTURE_DIR / "conservation_pde_chain.pdg.json",
        PDG_FIXTURE_DIR / "differentiate_integrate_chain.pdg.json",
        PDG_FIXTURE_DIR / "limit_nondimensionalization_chain.pdg.json",
        PDG_FIXTURE_DIR / "nondimensionalize_approximate_chain.pdg.json",
        PDG_FIXTURE_DIR / "scaling_symmetry_chain.pdg.json",
        PDG_FIXTURE_DIR / "solve_substitute_chain.pdg.json",
        PDG_FIXTURE_DIR / "variational_principle_chain.pdg.json",
    )


def test_changed_only_pdg_discovery_filters_git_changed_fixture_paths(
    tmp_path,
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    pdg_dir = repo / "tests" / "physics_ingest" / "fixtures" / "pdg_payloads"
    docs_pdg_dir = repo / "docs" / "physics_ingest" / "fixtures" / "pdg_payloads"
    pdg_dir.mkdir(parents=True)
    docs_pdg_dir.mkdir(parents=True)
    (pdg_dir / "changed.pdg.json").write_text("{}", encoding="utf-8")
    (docs_pdg_dir / "docs_changed.pdg.json").write_text("{}", encoding="utf-8")
    (pdg_dir / "skipped_missing_endpoint.json").write_text("{}", encoding="utf-8")
    docs_dir = repo / "docs"
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "symbolic_math.pdf").write_text("not a fixture", encoding="utf-8")

    paths = discover_changed_pdg_payload_fixture_paths(repo)

    assert paths == (
        docs_pdg_dir / "docs_changed.pdg.json",
        pdg_dir / "changed.pdg.json",
    )


def test_changed_only_symbolic_discovery_filters_git_changed_fixture_paths(
    tmp_path,
) -> None:
    atoms_repo = tmp_path / "atoms"
    _init_git_repo(atoms_repo)
    fixture_dir = atoms_repo / "data" / "publication_fixtures"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "force.publication_manifest.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (fixture_dir / "not_a_publication_fixture.json").write_text(
        "{}",
        encoding="utf-8",
    )
    docs_dir = atoms_repo / "docs"
    docs_dir.mkdir()
    (docs_dir / "symbolic_math.pdf").write_text("not a fixture", encoding="utf-8")

    paths = discover_changed_symbolic_fixture_paths(atoms_repo)

    assert paths == (fixture_dir / "force.publication_manifest.json",)


def test_pdg_payload_file_uses_stable_fixture_subject() -> None:
    check = validate_pdg_payload_file(
        PDG_FIXTURE_DIR / "solve_substitute_chain.pdg.json"
    )

    assert check.ok is True
    assert check.subject == "pdg_fixture:solve_substitute_chain"
    assert check.metadata["fixture_path"].endswith("solve_substitute_chain.pdg.json")


def test_pdg_payload_file_reports_deterministic_skipped_edge_reason() -> None:
    check = validate_pdg_payload_file(PDG_FIXTURE_DIR / "skipped_missing_endpoint.json")

    assert check.ok is False
    assert check.subject == "pdg_fixture:skipped_missing_endpoint"
    assert [issue.reason for issue in check.issues] == [
        "pdg_relationship_edge_skipped_missing_expression_binding"
    ]
    assert [issue.subject for issue in check.issues] == [
        "pdg_fixture:skipped_missing_endpoint:edge:missing_limit_case"
    ]


def test_validation_report_is_json_safe_and_fails_strict_without_fixtures() -> None:
    report = build_physics_ingestion_validation_report(
        fixture_paths=(),
        include_default_pdg=True,
        strict=True,
    )

    assert report["report_kind"] == VALIDATION_REPORT_KIND
    assert report["report_version"] == VALIDATION_REPORT_VERSION
    assert report["ok"] is False
    assert report["summary"] == {
        "check_count": 6,
        "failed_check_count": 2,
        "error_count": 2,
        "check_ids": [
            "symbolic_fixture_inventory",
            "pdg_payload_fixture_inventory",
            "pdg_publication_graph",
            "source_execution_readiness",
            "source_adapter_coverage",
            "source_adapter_data_artifact_seeds",
        ],
        "failed_check_ids": [
            "symbolic_fixture_inventory",
            "pdg_payload_fixture_inventory",
        ],
        "error_count_by_reason": {
            "missing_pdg_payload_fixture_inventory": 1,
            "missing_symbolic_fixture_inventory": 1,
        },
        "failed_count_by_check_id": {
            "pdg_payload_fixture_inventory": 1,
            "symbolic_fixture_inventory": 1,
        },
        "issue_count_by_table": {"unscoped": 2},
    }
    json.dumps(report, sort_keys=True)


def test_validation_report_includes_explicit_pdg_fixture_path() -> None:
    report = build_physics_ingestion_validation_report(
        pdg_payload_paths=(PDG_FIXTURE_DIR / "solve_substitute_chain.pdg.json",),
        include_default_pdg=False,
    )

    assert report["ok"] is True
    checks = report["checks"]
    assert len(checks) == 4
    assert checks[0]["subject"] == "pdg_fixture:solve_substitute_chain"
    assert checks[0]["metadata"]["fixture_path"].endswith(
        "solve_substitute_chain.pdg.json"
    )
    assert checks[1]["check_id"] == "source_execution_readiness"
    assert checks[2]["check_id"] == "source_adapter_coverage"
    assert checks[3]["check_id"] == "source_adapter_data_artifact_seeds"


def test_validation_report_includes_source_execution_by_default() -> None:
    report = build_physics_ingestion_validation_report(
        include_default_pdg=False,
        source_max_jobs=1,
    )

    assert report["ok"] is True
    checks = report["checks"]
    assert [check["check_id"] for check in checks] == [
        "source_execution_readiness",
        "source_adapter_coverage",
        "source_adapter_data_artifact_seeds",
    ]
    source_check = checks[0]
    assert source_check["metadata"]["total_steps"] == 1
    assert source_check["metadata"]["diagnostic_count"] == 0
    assert source_check["metadata"]["report"]["summary"]["total_steps"] == 1
    seed_check = checks[2]
    assert seed_check["metadata"]["seed_count"] == 6
    assert seed_check["metadata"]["diagnostic_count"] == 0
    json.dumps(report, sort_keys=True)


def test_validation_report_filters_source_execution_by_phase7_ring() -> None:
    report = build_physics_ingestion_validation_report(
        include_default_pdg=False,
        source_phase7_ring="ring_5_reference_datasets",
    )

    assert report["ok"] is True
    source_check = report["checks"][0]
    source_report = source_check["metadata"]["report"]
    assert source_check["check_id"] == "source_execution_readiness"
    assert source_check["metadata"]["total_steps"] == 4
    assert all(
        "ring_5_reference_datasets" in step["phase7_rings"]
        for step in source_report["steps"]
    )
    assert {step["phase7_ring"] for step in source_report["steps"]} == {
        "ring_1_foundational",
        "ring_2_existing_sciona_domains",
    }
    json.dumps(report, sort_keys=True)


def test_validation_report_can_skip_source_execution() -> None:
    report = build_physics_ingestion_validation_report(
        include_default_pdg=False,
        include_source_execution=False,
        include_source_adapter_coverage=False,
        include_source_adapter_data_artifact_seeds=False,
    )

    assert report["ok"] is True
    assert report["summary"] == {
        "check_count": 0,
        "failed_check_count": 0,
        "error_count": 0,
        "check_ids": [],
        "failed_check_ids": [],
        "error_count_by_reason": {},
        "failed_count_by_check_id": {},
        "issue_count_by_table": {},
    }
    assert report["checks"] == []


def test_validation_report_includes_source_adapter_coverage_by_default() -> None:
    report = build_physics_ingestion_validation_report(
        include_default_pdg=False,
        include_source_execution=False,
    )

    assert report["ok"] is True
    checks = report["checks"]
    assert [check["check_id"] for check in checks] == [
        "source_adapter_coverage",
        "source_adapter_data_artifact_seeds",
    ]
    coverage_check = checks[0]
    assert coverage_check["metadata"]["total_jobs"] == 11
    assert coverage_check["metadata"]["diagnostic_count"] == 0
    assert coverage_check["metadata"]["report"]["summary"]["total_jobs"] == 11
    seed_check = checks[1]
    assert seed_check["metadata"]["bundle_count"] == 6
    assert seed_check["metadata"]["seed_count"] == 6
    json.dumps(report, sort_keys=True)


def test_validation_report_can_skip_source_adapter_coverage() -> None:
    report = build_physics_ingestion_validation_report(
        include_default_pdg=False,
        include_source_execution=False,
        include_source_adapter_coverage=False,
    )

    assert report["ok"] is True
    assert [check["check_id"] for check in report["checks"]] == [
        "source_adapter_data_artifact_seeds"
    ]


def test_validation_report_can_skip_source_adapter_seed_quality() -> None:
    report = build_physics_ingestion_validation_report(
        include_default_pdg=False,
        include_source_execution=False,
        include_source_adapter_data_artifact_seeds=False,
    )

    assert report["ok"] is True
    assert [check["check_id"] for check in report["checks"]] == [
        "source_adapter_coverage"
    ]


def test_validation_report_summary_includes_dashboard_rollups() -> None:
    report = build_physics_ingestion_validation_report(
        include_default_pdg=False,
        include_source_execution=False,
        include_source_adapter_data_artifact_seeds=False,
        source_retrieval_manifest={
            "manifest_version": "test",
            "snapshot_key_prefix": "test",
            "jobs": [
                {
                    "job_id": "synthetic_missing_module.backfill",
                    "adapter_name": "sciona.physics_ingest.sources.does_not_exist",
                    "adapter_version": "0.0.1",
                    "target_adapter_input": "raw_records",
                }
            ],
        },
    )

    assert report["report_version"] == VALIDATION_REPORT_VERSION
    assert report["summary"] == {
        "check_count": 1,
        "failed_check_count": 1,
        "error_count": 2,
        "check_ids": ["source_adapter_coverage"],
        "failed_check_ids": ["source_adapter_coverage"],
        "error_count_by_reason": {
            "source_adapter_coverage_missing_adapter_module": 1,
            "source_adapter_coverage_missing_builder_readiness_contract": 1,
        },
        "failed_count_by_check_id": {"source_adapter_coverage": 1},
        "issue_count_by_table": {"source_adapter_coverage": 2},
    }
    json.dumps(report, sort_keys=True)


def test_source_adapter_data_artifact_seed_quality_accepts_default_bundles() -> None:
    check = validate_source_adapter_data_artifact_seed_quality()

    assert check.ok is True
    assert check.issues == ()
    assert check.metadata["bundle_count"] == 6
    assert check.metadata["seed_count"] == 6
    assert check.metadata["diagnostic_count"] == 0
    json.dumps(check.to_dict(), sort_keys=True)


def test_source_adapter_data_artifact_seed_quality_reports_stable_issues() -> None:
    check = validate_source_adapter_data_artifact_seed_quality(
        {
            "synthetic.bad": {
                "data_artifact_seeds": [
                    {
                        "artifact_kind": "",
                        "fqdn": "fixture.bad",
                        "source_system": "fixture",
                        "payload": object(),
                    }
                ]
            }
        }
    )

    assert check.ok is False
    assert [issue.reason for issue in check.issues] == [
        "source_adapter_data_artifact_seed_missing_artifact_kind",
        "source_adapter_data_artifact_seed_missing_source_id",
        "source_adapter_data_artifact_seed_json_unsafe",
    ]
    assert all(
        issue.table == "source_adapter_data_artifact_seeds"
        for issue in check.issues
    )


def test_source_adapter_data_artifact_seed_quality_reports_duplicate_fqdns() -> None:
    check = validate_source_adapter_data_artifact_seed_quality(
        (
            (
                "synthetic.first",
                {
                    "data_artifact_seeds": [
                        {
                            "artifact_kind": "data_artifact",
                            "fqdn": "fixture.duplicate",
                            "source_system": "fixture_a",
                            "source_id": "a-1",
                        }
                    ]
                },
            ),
            (
                "synthetic.second",
                {
                    "data_artifact_seeds": [
                        {
                            "artifact_kind": "data_artifact",
                            "fqdn": "fixture.duplicate",
                            "source_system": "fixture_b",
                            "source_id": "b-1",
                        }
                    ]
                },
            ),
        )
    )

    assert check.ok is False
    assert [issue.reason for issue in check.issues] == [
        "source_adapter_data_artifact_seed_duplicate_fqdn"
    ]
    assert [issue.subject for issue in check.issues] == [
        "source_adapter_data_artifact_seeds:synthetic.second:fixture.duplicate"
    ]
    assert check.issues[0].detail == (
        "duplicates source_adapter_data_artifact_seeds:"
        "synthetic.first:fixture.duplicate"
    )
    assert check.metadata["seed_count"] == 2
    assert check.metadata["diagnostic_count"] == 1


def test_seed_quality_reports_duplicate_source_identity() -> None:
    check = validate_source_adapter_data_artifact_seed_quality(
        {
            "synthetic.duplicates": {
                "data_artifact_seeds": [
                    {
                        "artifact_kind": "data_artifact",
                        "fqdn": "fixture.first",
                        "source_system": "fixture",
                        "source_id": "same-source-row",
                    },
                    {
                        "artifact_kind": "data_artifact",
                        "fqdn": "fixture.second",
                        "source_system": "fixture",
                        "source_id": "same-source-row",
                    },
                ]
            }
        }
    )

    assert check.ok is False
    assert [issue.reason for issue in check.issues] == [
        "source_adapter_data_artifact_seed_duplicate_source_identity"
    ]
    assert [issue.subject for issue in check.issues] == [
        "source_adapter_data_artifact_seeds:synthetic.duplicates:fixture.second"
    ]
    assert check.issues[0].detail == (
        "duplicates source_adapter_data_artifact_seeds:"
        "synthetic.duplicates:fixture.first"
    )
    assert check.metadata["diagnostic_count"] == 1


def test_source_execution_diagnostics_convert_to_validation_issues() -> None:
    plan = build_physics_source_retrieval_run_plan_dict(max_jobs=1)
    plan["dry_run"] = False
    plan["steps"][0]["adapter_module"] = ""
    plan["steps"][0]["adapter_name"] = ""

    check = validate_source_execution_readiness(plan)

    assert check.ok is False
    assert [issue.reason for issue in check.issues] == [
        "source_execution_non_dry_run_plan",
        "source_execution_non_dry_run_plan",
        "source_execution_missing_adapter_name",
    ]
    assert all(issue.table == "source_execution_readiness" for issue in check.issues)
    assert check.metadata["diagnostic_count"] == 3


def test_source_adapter_coverage_diagnostics_convert_to_validation_issues() -> None:
    manifest = {
        "manifest_version": "test",
        "snapshot_key_prefix": "test",
        "jobs": [
            {
                "job_id": "synthetic_missing_module.backfill",
                "adapter_name": "sciona.physics_ingest.sources.does_not_exist",
                "adapter_version": "0.0.1",
                "target_adapter_input": "raw_records",
            }
        ],
    }

    check = validate_source_adapter_coverage(manifest)

    assert check.ok is False
    assert [issue.reason for issue in check.issues] == [
        "source_adapter_coverage_missing_adapter_module",
        "source_adapter_coverage_missing_builder_readiness_contract",
    ]
    assert all(issue.table == "source_adapter_coverage" for issue in check.issues)
    assert [issue.subject for issue in check.issues] == [
        "source_adapter_coverage:synthetic_missing_module.backfill",
        "source_adapter_coverage:synthetic_missing_module.backfill",
    ]
    assert check.metadata["diagnostic_count"] == 2


def test_validation_script_discovers_default_pdg_fixtures_in_json_mode() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_physics_ingestion.py",
            "--skip-atoms",
            "--source-max-jobs",
            "1",
            "--json",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert report["report_version"] == VALIDATION_REPORT_VERSION
    subjects = [check["subject"] for check in report["checks"]]
    assert "pdg_fixture:limit_nondimensionalization_chain" in subjects
    assert "pdg_fixture:solve_substitute_chain" in subjects
    assert "default_pdg_validation_fixture" in subjects
    source_checks = [
        check
        for check in report["checks"]
        if check["check_id"] == "source_execution_readiness"
    ]
    assert len(source_checks) == 1
    assert source_checks[0]["metadata"]["report"]["summary"]["total_steps"] == 1
    coverage_checks = [
        check
        for check in report["checks"]
        if check["check_id"] == "source_adapter_coverage"
    ]
    assert len(coverage_checks) == 1
    assert coverage_checks[0]["metadata"]["report"]["summary"]["total_jobs"] == 11
    seed_checks = [
        check
        for check in report["checks"]
        if check["check_id"] == "source_adapter_data_artifact_seeds"
    ]
    assert len(seed_checks) == 1
    assert seed_checks[0]["metadata"]["seed_count"] == 6


def test_validation_script_strict_accepts_discovered_pdg_fixtures(tmp_path) -> None:
    fixture_path = tmp_path / "fixture.publication_manifest.json"
    fixture_path.write_text(json.dumps(_symbolic_manifest()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_physics_ingestion.py",
            "--atoms-repo",
            str(tmp_path / "missing-atoms-repo"),
            "--fixture",
            str(fixture_path),
            "--strict",
            "--skip-source-execution",
            "--skip-source-adapter-coverage",
            "--skip-source-adapter-data-artifact-seeds",
            "--json",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert "pdg_payload_fixture_inventory" not in report["summary"]["check_ids"]
    assert (
        "missing_pdg_payload_fixture_inventory"
        not in report["summary"]["error_count_by_reason"]
    )


def test_validation_script_changed_only_keeps_source_checks_in_json_mode() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_physics_ingestion.py",
            "--changed-only",
            "--skip-atoms",
            "--skip-pdg",
            "--source-max-jobs",
            "1",
            "--json",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert [check["check_id"] for check in report["checks"]] == [
        "source_execution_readiness",
        "source_adapter_coverage",
        "source_adapter_data_artifact_seeds",
    ]
    assert report["checks"][0]["metadata"]["report"]["summary"]["total_steps"] == 1


def test_validation_script_json_filters_source_execution_by_phase7_ring() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_physics_ingestion.py",
            "--changed-only",
            "--skip-atoms",
            "--skip-pdg",
            "--source-phase7-ring",
            "ring_5_reference_datasets",
            "--json",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    source_check = report["checks"][0]
    source_report = source_check["metadata"]["report"]
    assert report["ok"] is True
    assert source_check["check_id"] == "source_execution_readiness"
    assert source_report["summary"]["total_steps"] == 4
    assert all(
        "ring_5_reference_datasets" in step["phase7_rings"]
        for step in source_report["steps"]
    )


def test_validation_script_changed_only_json_accepts_explicit_pdg_fixture() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_physics_ingestion.py",
            "--changed-only",
            "--skip-atoms",
            "--skip-source-execution",
            "--skip-source-adapter-coverage",
            "--pdg-json",
            str(PDG_FIXTURE_DIR / "solve_substitute_chain.pdg.json"),
            "--json",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    subjects = [check["subject"] for check in report["checks"]]
    assert "pdg_fixture:solve_substitute_chain" in subjects
    assert "default_pdg_validation_fixture" in subjects


def test_validation_script_can_skip_source_checks_in_json_mode() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_physics_ingestion.py",
            "--skip-atoms",
            "--skip-pdg",
            "--skip-source-execution",
            "--skip-source-adapter-coverage",
            "--skip-source-adapter-data-artifact-seeds",
            "--json",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["checks"] == []


def _init_git_repo(path: Path) -> None:
    path.mkdir()
    subprocess.run(
        ["git", "init"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )


def _symbolic_manifest() -> dict[str, object]:
    return {
        "provider": "fixture",
        "modules": ["fixture.physics"],
        "artifact_symbolic_expressions": [
            {
                "artifact_key": "local:fixture.force",
                "local_artifact_key": "local:fixture.force",
                "provider": "fixture",
                "atom_name": "force_atom",
                "atom_module": "fixture.physics",
                "registry_name": "force_atom",
                "expression_id": "10000000-0000-0000-0000-000000000001",
                "expression_srepr": "Equality(Symbol('F'), Mul(Symbol('m'), Symbol('a')))",
                "sympy_srepr": "Equality(Symbol('F'), Mul(Symbol('m'), Symbol('a')))",
                "expression_text": "Eq(F, a*m)",
                "raw_formula": "Eq(F, a*m)",
                "raw_formula_format": "plain_text",
                "expression_kind": "equation",
                "expression_role": "primary",
                "canonical_expr_hash": "a" * 64,
                "topology_hash": "b" * 64,
                "dimensional_hash": "c" * 64,
                "parse_status": "normalized",
                "parse_confidence": 1.0,
                "review_status": "automated_pass",
                "validation_status": "passed",
                "mechanism_tags": ["conservation"],
                "behavioral_archetypes": ["source"],
                "variables": {"F": "output", "m": "input", "a": "input"},
                "dim_signature": {"F": "M1L1T-2", "m": "M1", "a": "L1T-2"},
                "symbolic_dim_signature": {
                    "F": "M1L1T-2",
                    "m": "M1",
                    "a": "L1T-2",
                },
                "constants": {},
                "bibliography": ["fixture"],
                "artifact_uuid": None,
            }
        ],
        "artifact_symbolic_variables": [
            {
                "artifact_key": "local:fixture.force",
                "local_artifact_key": "local:fixture.force",
                "provider": "fixture",
                "atom_name": "force_atom",
                "atom_module": "fixture.physics",
                "registry_name": "force_atom",
                "expression_id": "10000000-0000-0000-0000-000000000001",
                "symbol": "F",
                "symbol_name": "F",
                "source_symbol": "F",
                "source_variable_id": "fixture:F",
                "role": "output",
                "variable_role": "output",
                "dim_signature": "M1L1T-2",
                "dimension_source": "source",
                "assumptions_json": {"dim_signature": "M1L1T-2"},
                "evidence_json": {"source_symbol": "F"},
                "ordinal": 0,
            },
            {
                "artifact_key": "local:fixture.force",
                "local_artifact_key": "local:fixture.force",
                "provider": "fixture",
                "atom_name": "force_atom",
                "atom_module": "fixture.physics",
                "registry_name": "force_atom",
                "expression_id": "10000000-0000-0000-0000-000000000001",
                "symbol": "m",
                "symbol_name": "m",
                "source_symbol": "m",
                "source_variable_id": "fixture:m",
                "role": "input",
                "variable_role": "input",
                "dim_signature": "M1",
                "dimension_source": "source",
                "assumptions_json": {"dim_signature": "M1"},
                "evidence_json": {"source_symbol": "m"},
                "ordinal": 1,
            },
            {
                "artifact_key": "local:fixture.force",
                "local_artifact_key": "local:fixture.force",
                "provider": "fixture",
                "atom_name": "force_atom",
                "atom_module": "fixture.physics",
                "registry_name": "force_atom",
                "expression_id": "10000000-0000-0000-0000-000000000001",
                "symbol": "a",
                "symbol_name": "a",
                "source_symbol": "a",
                "source_variable_id": "fixture:a",
                "role": "input",
                "variable_role": "input",
                "dim_signature": "L1T-2",
                "dimension_source": "source",
                "assumptions_json": {"dim_signature": "L1T-2"},
                "evidence_json": {"source_symbol": "a"},
                "ordinal": 2,
            },
        ],
        "artifact_validity_bounds": [
            {
                "artifact_key": "local:fixture.force",
                "local_artifact_key": "local:fixture.force",
                "provider": "fixture",
                "atom_name": "force_atom",
                "atom_module": "fixture.physics",
                "registry_name": "force_atom",
                "expression_id": "10000000-0000-0000-0000-000000000001",
                "symbol": "m",
                "variable_name": "m",
                "source_symbol": "m",
                "source_bound_id": "fixture:m:bound",
                "scope": "variable",
                "bound_kind": "domain",
                "min_value": 0.0,
                "max_value": None,
                "lower_value": 0.0,
                "upper_value": None,
                "lower_inclusive": True,
                "upper_inclusive": True,
                "dim_signature": "M1",
                "validity_statement": "m >= 0.0",
                "evidence_ref_key": "fixture:m:bound",
                "confidence": "high",
                "review_status": "automated_pass",
                "metadata": {"provider": "fixture"},
                "ordinal": 0,
            }
        ],
    }


def _pdg_payload() -> dict[str, object]:
    return {
        "equations": [
            {"id": "eq:base", "label": "Newton second law", "latex": "F = m a"},
            {"id": "eq:solved", "label": "Acceleration", "latex": "a = F / m"},
            {
                "id": "eq:force",
                "label": "Constant mass force",
                "latex": "F(t) = m d^2x/dt^2",
            },
        ],
        "inference_edges": [
            {
                "id": "edge:solve",
                "source": "eq:base",
                "target": "eq:solved",
                "rule": "solve for acceleration",
                "confidence": 0.93,
                "bindings": {"solve_for": "a"},
            },
            {
                "id": "edge:substitute",
                "source": "eq:solved",
                "target": "eq:force",
                "rule": "substitution",
                "confidence": 0.81,
            },
        ],
    }
