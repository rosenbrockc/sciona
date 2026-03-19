"""Tests for Pareto front computation."""

from __future__ import annotations

import pytest

from ageom.clearinghouse.models import ObjectiveSpec, WinningCDG
from ageom.clearinghouse.pareto import (
    compute_pareto_front,
    dominates,
    split_architect_payout,
)


def _cdg(sid: str, **metrics: float) -> WinningCDG:
    return WinningCDG(
        submission_id=sid,
        architect_id=f"arch_{sid}",
        cdg_hash=f"hash_{sid}",
        metric_values=metrics,
    )


class TestDominance:
    def test_dominates_minimize(self):
        objs = [ObjectiveSpec(metric="loss", direction="minimize")]
        assert dominates({"loss": 0.1}, {"loss": 0.5}, objs)

    def test_not_dominates_equal(self):
        objs = [ObjectiveSpec(metric="loss", direction="minimize")]
        assert not dominates({"loss": 0.5}, {"loss": 0.5}, objs)

    def test_dominates_maximize(self):
        objs = [ObjectiveSpec(metric="acc", direction="maximize")]
        assert dominates({"acc": 0.9}, {"acc": 0.8}, objs)

    def test_two_objectives_dominates(self):
        objs = [
            ObjectiveSpec(metric="loss", direction="minimize"),
            ObjectiveSpec(metric="latency", direction="minimize"),
        ]
        assert dominates(
            {"loss": 0.1, "latency": 10},
            {"loss": 0.5, "latency": 50},
            objs,
        )

    def test_two_objectives_no_dominance(self):
        objs = [
            ObjectiveSpec(metric="loss", direction="minimize"),
            ObjectiveSpec(metric="latency", direction="minimize"),
        ]
        # a is better on loss, b is better on latency
        assert not dominates(
            {"loss": 0.1, "latency": 50},
            {"loss": 0.5, "latency": 10},
            objs,
        )


class TestParetoFront:
    def test_single_submission(self):
        objs = [ObjectiveSpec(metric="loss", direction="minimize")]
        subs = [_cdg("s1", loss=0.5)]
        front = compute_pareto_front(subs, objs)
        assert len(front) == 1
        assert front[0].submission_id == "s1"

    def test_all_dominated_by_one(self):
        objs = [ObjectiveSpec(metric="loss", direction="minimize")]
        subs = [_cdg("s1", loss=0.1), _cdg("s2", loss=0.5), _cdg("s3", loss=0.9)]
        front = compute_pareto_front(subs, objs, max_winners=5)
        assert len(front) == 1
        assert front[0].submission_id == "s1"

    def test_two_objective_pareto(self):
        objs = [
            ObjectiveSpec(metric="loss", direction="minimize"),
            ObjectiveSpec(metric="latency", direction="minimize"),
        ]
        subs = [
            _cdg("s1", loss=0.1, latency=100),  # best loss
            _cdg("s2", loss=0.5, latency=10),   # best latency
            _cdg("s3", loss=0.8, latency=80),   # dominated
        ]
        front = compute_pareto_front(subs, objs, max_winners=5)
        ids = {s.submission_id for s in front}
        assert "s1" in ids
        assert "s2" in ids
        assert "s3" not in ids

    def test_all_pareto_optimal_capped(self):
        objs = [
            ObjectiveSpec(metric="loss", direction="minimize"),
            ObjectiveSpec(metric="latency", direction="minimize"),
        ]
        # Each is Pareto-optimal (trade-off between loss and latency)
        subs = [
            _cdg("s1", loss=0.1, latency=100),
            _cdg("s2", loss=0.3, latency=50),
            _cdg("s3", loss=0.5, latency=30),
            _cdg("s4", loss=0.7, latency=10),
        ]
        front = compute_pareto_front(subs, objs, max_winners=2)
        assert len(front) == 2

    def test_empty_submissions(self):
        objs = [ObjectiveSpec(metric="loss", direction="minimize")]
        front = compute_pareto_front([], objs)
        assert front == []

    def test_empty_objectives(self):
        subs = [_cdg("s1", loss=0.5)]
        front = compute_pareto_front(subs, [])
        assert front == []

    def test_three_objectives(self):
        objs = [
            ObjectiveSpec(metric="loss", direction="minimize"),
            ObjectiveSpec(metric="latency", direction="minimize"),
            ObjectiveSpec(metric="memory", direction="minimize"),
        ]
        subs = [
            _cdg("s1", loss=0.1, latency=100, memory=500),
            _cdg("s2", loss=0.5, latency=10, memory=100),
            _cdg("s3", loss=0.3, latency=50, memory=50),   # Pareto-optimal
            _cdg("s4", loss=0.9, latency=90, memory=400),   # dominated
        ]
        front = compute_pareto_front(subs, objs, max_winners=5)
        ids = {s.submission_id for s in front}
        assert "s4" not in ids
        assert len(front) >= 2


class TestArchitectPayoutSplit:
    def test_single_winner(self):
        winners = [_cdg("s1", loss=0.1)]
        weights = split_architect_payout(winners, [])
        assert weights == {"s1": 1.0}

    def test_multiple_winners(self):
        objs = [ObjectiveSpec(metric="loss", direction="minimize")]
        winners = [_cdg("s1", loss=0.1), _cdg("s2", loss=0.5)]
        weights = split_architect_payout(winners, objs)
        assert len(weights) == 2
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_empty_winners(self):
        weights = split_architect_payout([], [])
        assert weights == {}
