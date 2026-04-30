from __future__ import annotations

import re

from sciona.physics_ingest.staging import stage_source_rows
from sciona.physics_ingest.sources.phy_srbench import (
    build_phy_srbench_wave0_bundle,
)
from sciona.physics_ingest.sources.theoria import build_theoria_wave0_bundle


SNAPSHOT_ID = "00000000-0000-0000-0000-000000000020"


def test_theoria_bundle_retains_formula_gaps_and_pending_license_rows() -> None:
    raw_records = [
        {
            "problem_id": "theoria:lagrangian:oscillator",
            "title": "Harmonic oscillator Euler-Lagrange equation",
            "latex": r"\frac{d}{dt}\frac{\partial L}{\partial \dot{x}}-\frac{\partial L}{\partial x}=0",
            "theory": "classical mechanics",
            "variables": [{"symbol": "x", "role": "coordinate"}],
            "evaluation": {"metric": "symbolic_equivalence", "tolerance": 0},
            "mechanism_tags": ["mechanics"],
        },
        {
            "problem_id": "theoria:license-pending:no-formula",
            "title": "Retained raw benchmark row",
            "benchmark": {"split": "test", "expected_status": "pending"},
            "license_status": "pending",
        },
    ]

    bundle = build_theoria_wave0_bundle(
        raw_records,
        source_version="TheorIA fixture",
        retrieved_at="2026-04-30T00:00:00Z",
        snapshot_id=SNAPSHOT_ID,
        license_expression="PENDING",
    )
    bundle_again = build_theoria_wave0_bundle(
        raw_records,
        source_version="TheorIA fixture",
        retrieved_at="2026-04-30T00:00:00Z",
        snapshot_id=SNAPSHOT_ID,
        license_expression="PENDING",
    )

    snapshot = bundle.snapshot_row
    assert snapshot["source_system"] == "theoria"
    assert snapshot["adapter_name"] == "sciona.physics_ingest.sources.theoria"
    assert snapshot["license_expression"] == "PENDING"
    assert snapshot["payload"]["record_count"] == 2
    assert snapshot["payload"]["formula_record_count"] == 1
    assert snapshot["payload"]["evaluation_record_count"] == 2
    assert snapshot["payload"]["license_review_status"] == "pending"
    assert re.fullmatch(r"[0-9a-f]{64}", str(snapshot["payload_sha256"]))
    assert snapshot["payload_sha256"] == bundle_again.snapshot_row["payload_sha256"]

    equation = bundle.candidate_rows[0]
    assert equation["snapshot_id"] == SNAPSHOT_ID
    assert equation["raw_formula_format"] == "latex"
    assert equation["source_payload"]["evaluation_payload"] == {
        "metric": "symbolic_equivalence",
        "tolerance": 0,
    }
    assert bundle.data_artifact_seeds[0]["artifact_role"] == (
        "benchmark_evaluation_seed"
    )

    pending = bundle.candidate_rows[1]
    assert pending["raw_formula"] == ""
    assert pending["raw_formula_format"] == ""
    assert pending["source_payload"]["raw_record"]["license_status"] == "pending"
    assert bundle.data_artifact_seeds[1]["evaluation_payload"] == {
        "split": "test",
        "expected_status": "pending",
    }

    staged_snapshot, staged_candidates = stage_source_rows(
        snapshot_row=bundle.snapshot_row,
        candidate_rows=bundle.candidate_rows,
        snapshot_id=SNAPSHOT_ID,
    )
    assert staged_snapshot.source_system == "theoria"
    assert staged_candidates[1].source_candidate_id == (
        "theoria:license-pending:no-formula"
    )


def test_phy_srbench_bundle_preserves_dataset_and_evaluation_artifacts() -> None:
    raw_records = [
        {
            "task_id": "phy-srbench:kepler",
            "name": "Kepler third law synthetic task",
            "sympy": "Eq(T**2, 4*pi**2*a**3/(G*M))",
            "variables": [
                {"symbol": "T", "role": "target", "dim_signature": "T1"},
                {"symbol": "a", "role": "input", "dim_signature": "L1"},
            ],
            "dataset": {"train_rows": [{"a": 1.0, "T": 1.0}]},
            "evaluation_spec": {"metric": "r2", "threshold": 0.99},
        },
        {
            "task_id": "phy-srbench:raw-no-target",
            "title": "Dataset without disclosed target equation",
            "data": {"test_rows": [{"x": 1.0, "y": 2.0}]},
            "metrics": {"metric": "mse"},
            "license_status": "pending",
        },
    ]

    bundle = build_phy_srbench_wave0_bundle(
        raw_records,
        source_version="Phy-SRBench fixture",
        source_uri="https://example.invalid/phy-srbench-fixture",
        snapshot_id=SNAPSHOT_ID,
        license_expression="PENDING",
    )
    bundle_again = build_phy_srbench_wave0_bundle(
        raw_records,
        source_version="Phy-SRBench fixture",
        source_uri="https://example.invalid/phy-srbench-fixture",
        snapshot_id=SNAPSHOT_ID,
        license_expression="PENDING",
    )

    assert bundle.snapshot_row["source_system"] == "phy_srbench"
    assert bundle.snapshot_row["payload"]["record_count"] == 2
    assert bundle.snapshot_row["payload"]["formula_record_count"] == 1
    assert bundle.snapshot_row["payload"]["dataset_record_count"] == 2
    assert bundle.snapshot_row["payload"]["evaluation_record_count"] == 2
    assert bundle.snapshot_row["payload_sha256"] == bundle_again.snapshot_row[
        "payload_sha256"
    ]

    kepler = bundle.candidate_rows[0]
    assert kepler["raw_formula"] == "Eq(T**2, 4*pi**2*a**3/(G*M))"
    assert kepler["raw_formula_format"] == "sympy"
    assert kepler["mechanism_tags"] == ["symbolic_regression", "benchmark"]
    assert kepler["source_payload"]["dataset_payload"] == {
        "train_rows": [{"a": 1.0, "T": 1.0}]
    }
    assert bundle.data_artifact_seeds[0]["evaluation_payload"] == {
        "metric": "r2",
        "threshold": 0.99,
    }

    raw_only = bundle.candidate_rows[1]
    assert raw_only["raw_formula"] == ""
    assert raw_only["raw_formula_format"] == ""
    assert raw_only["source_payload"]["raw_record"]["license_status"] == "pending"
    assert bundle.data_artifact_seeds[1]["dataset_payload"] == {
        "test_rows": [{"x": 1.0, "y": 2.0}]
    }

    staged_snapshot, staged_candidates = stage_source_rows(
        snapshot_row=bundle.snapshot_row,
        candidate_rows=bundle.candidate_rows,
        snapshot_id=SNAPSHOT_ID,
    )
    assert staged_snapshot.source_system == "phy_srbench"
    assert staged_candidates[0].raw_formula_format == "sympy"
