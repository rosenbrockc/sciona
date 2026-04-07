"""Phase 7 policy tests for benchmark and e2e discipline."""

from __future__ import annotations

import json
from pathlib import Path

from sciona.benchmark_validation import flow_execution_path_summary
from sciona.principal.e2e_benchmark_policy import evaluate_e2e_benchmark_report


def test_e2e_benchmark_scripts_enforce_full_framework_mode() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    benchmark_script = (repo_root / "benchmarks" / "e2e_benchmark.sh").read_text()
    benchmark_all_script = (repo_root / "benchmarks" / "e2e_benchmark_all.sh").read_text()

    assert "SCIONA_DISABLE_CURATED_SIGNAL_EVENT_RATE_SHORTCUTS" in benchmark_script
    assert "SCIONA_SEMANTIC_INDEX_BACKEND=faiss" in benchmark_script
    assert "postprocess.json" in benchmark_script
    assert "summary.json" in benchmark_script
    assert "summary_table.txt" in benchmark_script
    assert "evaluate_e2e_benchmark_report" in benchmark_script
    assert "variant | artifacts | anti-shortcut | behavioral" in benchmark_script
    assert "e2e_goals/" in benchmark_all_script
    assert "e2e_benchmark.sh" in benchmark_all_script
    assert "for config in" in benchmark_all_script


def test_flow_execution_path_summary_flags_collapsed_execution_paths() -> None:
    class _Agg:
        def __init__(self, variant: str, execution_paths: list[str]) -> None:
            self.variant = variant
            self.execution_paths = execution_paths

    summary = flow_execution_path_summary(
        [
            _Agg("rapid", ["verified_orchestration"]),
            _Agg("structured", ["structured_single_pass"]),
            _Agg("verified", ["verified_orchestration"]),
        ]
    )

    assert summary["violations"]
    assert any("expected rapid_direct" in violation for violation in summary["violations"])
    assert any(
        violation.startswith("mode_paths_not_distinct:")
        for violation in summary["violations"]
    )


def test_e2e_benchmark_policy_reads_list_trial_history_and_runtime_evidence(
    tmp_path: Path,
) -> None:
    mode_dir = tmp_path / "pipeline_verified"
    mode_dir.mkdir()
    (mode_dir / "cdg.json").write_text(
        json.dumps(
            {
                "planning_artifact": {
                    "family_hint": "signal_event_rate",
                    "paradigm": "signal_event_rate",
                },
                "metadata": {
                    "skeleton_asset": {"asset_id": "skeleton.signal_event_rate.ecg.v1"}
                },
            }
        )
    )
    (mode_dir / "trial_history.json").write_text(
        json.dumps([{"trial": 1, "admissibility": {"decision_count": 2}}])
    )
    (mode_dir / "runtime_evidence.json").write_text(
        json.dumps({"runtime_context": {"stream_count": 1}})
    )

    report = evaluate_e2e_benchmark_report(
        goal="Detect heart rate from raw ECG signal",
        prover="python",
        llm_provider="codex_shim",
        total_ground_truth=3,
        variants={
            "verified": {
                "mode_dir": mode_dir,
                "latency_ms": 1234,
                "matches_total": 3,
                "matches_verified": 3,
                "ground_truth_hits": 3,
                "executable": True,
            }
        },
        shortcut_flags={"curated_signal_event_rate_shortcut": False},
        declared_shortcuts=[],
    )

    verified = report["results"]["verified"]
    assert verified["ground_truth_coverage"] == 1.0
    assert verified["artifact_inventory"]["search_trace"] is True
    assert verified["search_trace_summary"]["entry_count"] == 1
    assert verified["policy"]["required_artifacts"]["passed"] is True
    assert verified["policy"]["behavioral"]["passed"] is True
    assert report["benchmark_policy"]["anti_shortcut"]["passed"] is True
