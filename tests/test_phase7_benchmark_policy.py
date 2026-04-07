"""Phase 7 policy tests for benchmark and e2e discipline."""

from __future__ import annotations

from pathlib import Path

from sciona.benchmark_validation import flow_execution_path_summary


def test_e2e_benchmark_scripts_enforce_full_framework_mode() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    benchmark_script = (repo_root / "benchmarks" / "e2e_benchmark.sh").read_text()
    benchmark_all_script = (repo_root / "benchmarks" / "e2e_benchmark_all.sh").read_text()

    assert "SCIONA_DISABLE_CURATED_SIGNAL_EVENT_RATE_SHORTCUTS" in benchmark_script
    assert "SCIONA_SEMANTIC_INDEX_BACKEND=faiss" in benchmark_script
    assert "postprocess.json" in benchmark_script
    assert "summary.json" in benchmark_script
    assert "summary_table.txt" in benchmark_script
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
