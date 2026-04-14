"""Tests for benchmark loading and prior computation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sciona.ecosystem.benchmarks import (
    compute_atom_prior,
    load_benchmarks_sqlite,
    normalize_metric_to_reward,
)
from sciona.ecosystem.models import BenchmarkRecord
from sciona.principal.benchmark_priors import (
    apply_benchmark_prior,
    load_benchmark_priors,
    score_untried_with_prior,
)


class TestBenchmarkSQLiteLoader:
    def test_load_benchmarks(self, tmp_path: Path):
        db_path = tmp_path / "manifest.sqlite"
        con = sqlite3.connect(str(db_path))
        con.execute("""CREATE TABLE benchmarks (
            atom_fqdn TEXT, content_hash TEXT, benchmark_id TEXT,
            metric_name TEXT, metric_value REAL, dataset_tag TEXT, measured_at TEXT,
            PRIMARY KEY (atom_fqdn, content_hash, benchmark_id, metric_name)
        )""")
        con.execute(
            "INSERT INTO benchmarks VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("pkg.filter", "abc", "signal_v1", "loss", 0.42, "v1", "2025-01-01"),
        )
        con.execute(
            "INSERT INTO benchmarks VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("pkg.filter", "abc", "signal_v1", "latency_ms", 15.0, "v1", "2025-01-01"),
        )
        con.execute(
            "INSERT INTO benchmarks VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("pkg.sort", "def", "sort_v1", "time_ms", 2.0, "v1", "2025-01-01"),
        )
        con.commit()
        con.close()

        result = load_benchmarks_sqlite(db_path)
        assert "pkg.filter" in result
        assert len(result["pkg.filter"]) == 2
        assert "pkg.sort" in result
        assert result["pkg.sort"][0].metric_value == 2.0

    def test_missing_file(self, tmp_path: Path):
        result = load_benchmarks_sqlite(tmp_path / "nonexistent.sqlite")
        assert result == {}

    def test_no_benchmarks_table(self, tmp_path: Path):
        db_path = tmp_path / "manifest.sqlite"
        con = sqlite3.connect(str(db_path))
        con.execute("CREATE TABLE atoms (fqdn TEXT)")
        con.commit()
        con.close()
        result = load_benchmarks_sqlite(db_path)
        assert result == {}

    def test_warns_when_manifest_is_stale(self, tmp_path: Path):
        db_path = tmp_path / "manifest.sqlite"
        con = sqlite3.connect(str(db_path))
        con.execute("CREATE TABLE atoms (fqdn TEXT)")
        con.execute(
            """CREATE TABLE benchmarks (
                atom_fqdn TEXT, content_hash TEXT, benchmark_id TEXT,
                metric_name TEXT, metric_value REAL, dataset_tag TEXT, measured_at TEXT,
                PRIMARY KEY (atom_fqdn, content_hash, benchmark_id, metric_name)
            )"""
        )
        con.execute("CREATE TABLE manifest_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        con.executemany(
            "INSERT INTO manifest_metadata (key, value) VALUES (?, ?)",
            [
                ("generated_at", "2020-01-01T00:00:00Z"),
                ("content_hash", "bogus"),
            ],
        )
        con.commit()
        con.close()

        with pytest.warns(UserWarning, match="Run 'sciona catalog sync' to update"):
            load_benchmarks_sqlite(db_path)


class TestMetricNormalization:
    def test_minimize_small_value(self):
        # Small loss → high reward
        r = normalize_metric_to_reward(0.1, "loss", direction="minimize")
        assert 0.8 < r <= 1.0

    def test_minimize_large_value(self):
        # Large loss → low reward
        r = normalize_metric_to_reward(100.0, "loss", direction="minimize")
        assert r < 0.02

    def test_maximize_high_value(self):
        r = normalize_metric_to_reward(0.9, "accuracy", direction="maximize")
        assert r > 0.4

    def test_with_known_range(self):
        r = normalize_metric_to_reward(
            0.3, "loss", direction="minimize", best_known=0.0, worst_known=1.0
        )
        assert abs(r - 0.7) < 0.01

    def test_zero_span(self):
        r = normalize_metric_to_reward(
            5.0, "loss", direction="minimize", best_known=5.0, worst_known=5.0
        )
        assert r == 0.5


class TestAtomPrior:
    def test_basic(self):
        records = [
            BenchmarkRecord("a", "h", "b1", "loss", 0.1, "v1", "2025"),
            BenchmarkRecord("a", "h", "b2", "loss", 0.2, "v1", "2025"),
        ]
        prior = compute_atom_prior(records, direction="minimize")
        assert 0 < prior <= 1.0

    def test_empty(self):
        assert compute_atom_prior([]) == 0.0


class TestBenchmarkPriors:
    def test_load_priors(self, tmp_path: Path):
        db_path = tmp_path / "manifest.sqlite"
        con = sqlite3.connect(str(db_path))
        con.execute("""CREATE TABLE benchmarks (
            atom_fqdn TEXT, content_hash TEXT, benchmark_id TEXT,
            metric_name TEXT, metric_value REAL, dataset_tag TEXT, measured_at TEXT,
            PRIMARY KEY (atom_fqdn, content_hash, benchmark_id, metric_name)
        )""")
        con.execute(
            "INSERT INTO benchmarks VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("pkg.filter", "abc", "b1", "loss", 0.1, "v1", "2025"),
        )
        con.commit()
        con.close()

        priors = load_benchmark_priors(db_path)
        assert "pkg.filter" in priors
        assert 0 < priors["pkg.filter"] <= 1.0

    def test_apply_prior_no_observations(self):
        effective = apply_benchmark_prior(0.0, 0, 0.8, prior_strength=2)
        assert abs(effective - 0.8) < 0.01

    def test_apply_prior_many_observations(self):
        effective = apply_benchmark_prior(0.5, 100, 0.8, prior_strength=2)
        # Prior should be washed out
        assert abs(effective - 0.5) < 0.01

    def test_apply_prior_balanced(self):
        effective = apply_benchmark_prior(0.5, 2, 0.8, prior_strength=2)
        assert abs(effective - 0.65) < 0.01

    def test_score_untried_with_prior(self):
        s1 = score_untried_with_prior(0.9)
        s2 = score_untried_with_prior(0.5)
        assert s1 > s2  # better benchmark → higher score
        assert s1 > 1e5  # still above any tried atom

    def test_score_untried_without_prior(self):
        s = score_untried_with_prior(0.0)
        assert s > 1e5  # still above any tried atom
