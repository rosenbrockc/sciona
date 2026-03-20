"""Soft deprecation — supersession detection for atoms."""

from __future__ import annotations

from sciona.ecosystem.models import BenchmarkRecord


def detect_supersession(
    old_benchmarks: list[BenchmarkRecord],
    new_benchmarks: list[BenchmarkRecord],
    *,
    margin_pct: float = 5.0,
    direction: str = "minimize",
) -> bool:
    """Detect if the newer atom version supersedes the older one.

    The newer version must be strictly better on ALL shared benchmarks
    (same benchmark_id + metric_name) with at least *margin_pct*
    improvement.

    Parameters
    ----------
    old_benchmarks
        Benchmark records for the older atom version.
    new_benchmarks
        Benchmark records for the newer atom version.
    margin_pct
        Minimum improvement percentage (default 5%).
    direction
        ``"minimize"`` or ``"maximize"``.

    Returns
    -------
    bool
        True if the newer version supersedes the older one.
    """
    if not old_benchmarks or not new_benchmarks:
        return False

    # Index benchmarks by (benchmark_id, metric_name)
    old_by_key = {
        (b.benchmark_id, b.metric_name): b.metric_value for b in old_benchmarks
    }
    new_by_key = {
        (b.benchmark_id, b.metric_name): b.metric_value for b in new_benchmarks
    }

    # Find shared benchmark keys
    shared = set(old_by_key.keys()) & set(new_by_key.keys())
    if not shared:
        return False

    margin = margin_pct / 100.0

    for key in shared:
        old_val = old_by_key[key]
        new_val = new_by_key[key]

        if old_val == 0:
            continue

        if direction == "minimize":
            improvement = (old_val - new_val) / abs(old_val)
        else:
            improvement = (new_val - old_val) / abs(old_val)

        if improvement < margin:
            return False

    return True


def apply_supersession_penalty(
    ucb_score: float,
    atom_status: str,
    penalty_factor: float = 0.5,
) -> float:
    """Apply a penalty to the UCB score of superseded atoms.

    Superseded atoms are deprioritized but never excluded.
    """
    if atom_status == "superseded":
        return ucb_score * penalty_factor
    return ucb_score
