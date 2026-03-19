"""Tests for soft deprecation — supersession detection."""

from __future__ import annotations

import pytest

from ageom.ecosystem.models import BenchmarkRecord
from ageom.ecosystem.soft_deprecation import (
    apply_supersession_penalty,
    detect_supersession,
)


def _record(benchmark_id: str, metric_name: str, value: float) -> BenchmarkRecord:
    return BenchmarkRecord(
        atom_fqdn="pkg.filter",
        content_hash="abc",
        benchmark_id=benchmark_id,
        metric_name=metric_name,
        metric_value=value,
        dataset_tag="v1",
        measured_at="2025-01-01",
    )


class TestSupersessionDetection:
    def test_strictly_better_supersedes(self):
        old = [_record("b1", "loss", 1.0), _record("b2", "latency", 100)]
        new = [_record("b1", "loss", 0.5), _record("b2", "latency", 50)]
        assert detect_supersession(old, new, margin_pct=5.0) is True

    def test_same_performance_no_supersession(self):
        old = [_record("b1", "loss", 1.0)]
        new = [_record("b1", "loss", 1.0)]
        assert detect_supersession(old, new) is False

    def test_marginal_improvement_no_supersession(self):
        old = [_record("b1", "loss", 1.0)]
        new = [_record("b1", "loss", 0.96)]  # only 4% better
        assert detect_supersession(old, new, margin_pct=5.0) is False

    def test_mixed_results_no_supersession(self):
        old = [_record("b1", "loss", 1.0), _record("b2", "latency", 100)]
        new = [_record("b1", "loss", 0.5), _record("b2", "latency", 110)]  # worse on latency
        assert detect_supersession(old, new) is False

    def test_empty_benchmarks(self):
        assert detect_supersession([], [_record("b1", "loss", 0.5)]) is False
        assert detect_supersession([_record("b1", "loss", 1.0)], []) is False

    def test_no_shared_benchmarks(self):
        old = [_record("b1", "loss", 1.0)]
        new = [_record("b2", "latency", 50)]
        assert detect_supersession(old, new) is False

    def test_maximize_direction(self):
        old = [_record("b1", "accuracy", 0.80)]
        new = [_record("b1", "accuracy", 0.90)]
        assert detect_supersession(old, new, direction="maximize") is True

    def test_maximize_no_improvement(self):
        old = [_record("b1", "accuracy", 0.80)]
        new = [_record("b1", "accuracy", 0.81)]  # only 1.25%
        assert detect_supersession(old, new, margin_pct=5.0, direction="maximize") is False

    def test_custom_margin(self):
        old = [_record("b1", "loss", 1.0)]
        new = [_record("b1", "loss", 0.85)]  # 15% better
        assert detect_supersession(old, new, margin_pct=10.0) is True
        assert detect_supersession(old, new, margin_pct=20.0) is False


class TestSupersessionPenalty:
    def test_no_penalty_for_approved(self):
        assert apply_supersession_penalty(1.0, "approved") == 1.0

    def test_penalty_for_superseded(self):
        assert apply_supersession_penalty(1.0, "superseded") == 0.5

    def test_custom_factor(self):
        assert apply_supersession_penalty(1.0, "superseded", penalty_factor=0.3) == 0.3

    def test_zero_score(self):
        assert apply_supersession_penalty(0.0, "superseded") == 0.0
