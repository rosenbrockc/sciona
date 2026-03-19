"""Benchmark loading and management."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from ageom.ecosystem.models import BenchmarkRecord

logger = logging.getLogger(__name__)


def load_benchmarks_sqlite(
    db_path: Path,
) -> dict[str, list[BenchmarkRecord]]:
    """Load per-atom benchmark records from manifest.sqlite.

    Returns a mapping of atom FQDN to benchmark records.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        logger.warning("Benchmark SQLite not found: %s", db_path)
        return {}

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        # Check if benchmarks table exists
        tables = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "benchmarks" not in tables:
            return {}

        rows = con.execute(
            """SELECT atom_fqdn, content_hash, benchmark_id,
                      metric_name, metric_value, dataset_tag, measured_at
               FROM benchmarks
               ORDER BY atom_fqdn, benchmark_id"""
        ).fetchall()
    finally:
        con.close()

    result: dict[str, list[BenchmarkRecord]] = {}
    for row in rows:
        record = BenchmarkRecord(
            atom_fqdn=row["atom_fqdn"],
            content_hash=row["content_hash"],
            benchmark_id=row["benchmark_id"],
            metric_name=row["metric_name"],
            metric_value=row["metric_value"],
            dataset_tag=row["dataset_tag"],
            measured_at=row["measured_at"],
        )
        result.setdefault(row["atom_fqdn"], []).append(record)

    return result


def normalize_metric_to_reward(
    value: float,
    metric_name: str,
    *,
    direction: str = "minimize",
    best_known: float | None = None,
    worst_known: float | None = None,
) -> float:
    """Normalize a raw metric value to a [0, 1] reward.

    Parameters
    ----------
    value
        The raw metric value.
    metric_name
        Name of the metric (for future per-metric strategies).
    direction
        ``"minimize"`` (lower is better) or ``"maximize"``.
    best_known / worst_known
        Optional range anchors.  If provided, uses min-max normalization.
        Otherwise uses ``1 / (1 + value)`` for minimize or ``value / (1 + value)``
        for maximize.
    """
    if best_known is not None and worst_known is not None:
        span = abs(worst_known - best_known)
        if span == 0:
            return 0.5
        if direction == "minimize":
            return max(0.0, min(1.0, (worst_known - value) / span))
        return max(0.0, min(1.0, (value - worst_known) / span))

    if direction == "minimize":
        return 1.0 / (1.0 + abs(value))
    return abs(value) / (1.0 + abs(value))


def compute_atom_prior(
    benchmarks: list[BenchmarkRecord],
    direction: str = "minimize",
) -> float:
    """Compute a single [0, 1] prior reward from benchmark records.

    Averages normalized rewards across all benchmark measurements.
    """
    if not benchmarks:
        return 0.0

    rewards = [
        normalize_metric_to_reward(b.metric_value, b.metric_name, direction=direction)
        for b in benchmarks
    ]
    return sum(rewards) / len(rewards)
