from __future__ import annotations

from pathlib import Path

from sciona.benchmarks.provider_results import generate_provider_benchmark_results


def test_generate_provider_benchmark_results_groups_rows_by_provider_path() -> None:
    grouped = generate_provider_benchmark_results()

    assert len(grouped) == 2
    signal_path = next(path for path in grouped if "sciona-atoms-signal" in str(path))
    core_path = next(path for path in grouped if str(path).endswith("sciona-atoms/data/benchmarks/benchmark_results.json"))

    assert signal_path.name == "benchmark_results.json"
    assert core_path.name == "benchmark_results.json"
    assert {row["suite_id"] for row in grouped[signal_path]} == {"signal.event_rate.ecg.v1"}
    assert {row["suite_id"] for row in grouped[core_path]} == {
        "state_estimation.kalman.synthetic_tracking.v1",
        "state_estimation.particle.synthetic_tracking.v1",
    }


def test_generated_rows_are_cdg_benchmarks_with_deterministic_fields() -> None:
    grouped = generate_provider_benchmark_results()
    rows = [row for group in grouped.values() for row in group]

    assert rows
    assert {row["artifact_kind"] for row in rows} == {"cdg"}
    assert {row["measured_at"] for row in rows} == {"2026-04-14T00:00:00Z"}
    assert all(Path(path).name == "benchmark_results.json" for path in map(str, grouped))
    assert all(len(str(row["content_hash"])) == 64 for row in rows)
    assert all(row["run_config_hash"] for row in rows)
