"""Tests for AtomLedger UCB1 with benchmark priors and supersession penalty."""

from __future__ import annotations

import math

import pytest

from sciona.principal.atom_ledger import AtomLedger, SlotSignature


def _slot() -> SlotSignature:
    return SlotSignature(
        parent_name="root",
        concept_type="filter",
        input_types=("np.ndarray",),
        output_types=("np.ndarray",),
    )


class TestBenchmarkPriorRanking:
    def test_untried_without_prior_ranked_above_untried_with_prior(self):
        ledger = AtomLedger()
        slot = _slot()
        priors = {"atom_a": 0.9}
        ranked = ledger.rank_candidates(
            slot, ["atom_a", "atom_b"], benchmark_priors=priors
        )
        # atom_b has no prior → inf (explore first); atom_a has prior → 1e18 + 0.9
        scores = {name: score for name, score in ranked}
        assert scores["atom_b"] > scores["atom_a"]
        assert math.isinf(scores["atom_b"])

    def test_untried_with_better_prior_ranked_higher(self):
        ledger = AtomLedger()
        slot = _slot()
        priors = {"atom_a": 0.9, "atom_b": 0.3}
        ranked = ledger.rank_candidates(
            slot, ["atom_a", "atom_b"], benchmark_priors=priors
        )
        scores = {name: score for name, score in ranked}
        assert scores["atom_a"] > scores["atom_b"]

    def test_all_untried_with_priors_above_tried(self):
        ledger = AtomLedger()
        slot = _slot()
        # Record some observations for tried_atom
        for i in range(5):
            ledger.record(slot, "tried_atom", 50.0, i)

        priors = {"untried_a": 0.5}
        ranked = ledger.rank_candidates(
            slot, ["tried_atom", "untried_a"], benchmark_priors=priors
        )
        scores = {name: score for name, score in ranked}
        assert scores["untried_a"] > scores["tried_atom"]

    def test_prior_mixes_with_observations(self):
        ledger = AtomLedger()
        slot = _slot()
        # Record 2 observations with mean reward ~0.5
        ledger.record(slot, "atom_a", 50.0, 0)
        ledger.record(slot, "atom_a", 50.0, 1)

        # Without prior
        ranked_no_prior = ledger.rank_candidates(slot, ["atom_a"])
        score_no_prior = ranked_no_prior[0][1]

        # With high prior (0.9)
        ranked_with_prior = ledger.rank_candidates(
            slot, ["atom_a"], benchmark_priors={"atom_a": 0.9}
        )
        score_with_prior = ranked_with_prior[0][1]

        # High prior should boost the score
        assert score_with_prior > score_no_prior

    def test_prior_washes_out(self):
        ledger = AtomLedger()
        slot = _slot()
        # Record many observations
        for i in range(100):
            ledger.record(slot, "atom_a", 50.0, i)

        ranked_no_prior = ledger.rank_candidates(slot, ["atom_a"])
        ranked_with_prior = ledger.rank_candidates(
            slot, ["atom_a"], benchmark_priors={"atom_a": 0.9}, prior_strength=2
        )

        # With 100 observations and prior_strength=2, prior effect is minimal
        diff = abs(ranked_with_prior[0][1] - ranked_no_prior[0][1])
        assert diff < 0.1


class TestSupersessionPenalty:
    def test_superseded_untried_penalized(self):
        ledger = AtomLedger()
        slot = _slot()
        statuses = {"atom_a": "superseded"}
        priors = {"atom_a": 0.9, "atom_b": 0.9}
        ranked = ledger.rank_candidates(
            slot, ["atom_a", "atom_b"],
            benchmark_priors=priors,
            atom_statuses=statuses,
        )
        scores = {name: score for name, score in ranked}
        assert scores["atom_b"] > scores["atom_a"]

    def test_superseded_tried_penalized(self):
        ledger = AtomLedger()
        slot = _slot()
        for i in range(5):
            ledger.record(slot, "atom_a", 50.0, i)
            ledger.record(slot, "atom_b", 50.0, i)

        statuses = {"atom_a": "superseded"}
        ranked = ledger.rank_candidates(
            slot, ["atom_a", "atom_b"], atom_statuses=statuses
        )
        scores = {name: score for name, score in ranked}
        assert scores["atom_b"] > scores["atom_a"]

    def test_approved_not_penalized(self):
        ledger = AtomLedger()
        slot = _slot()
        for i in range(5):
            ledger.record(slot, "atom_a", 50.0, i)

        ranked_no_status = ledger.rank_candidates(slot, ["atom_a"])
        ranked_approved = ledger.rank_candidates(
            slot, ["atom_a"], atom_statuses={"atom_a": "approved"}
        )
        assert ranked_no_status[0][1] == ranked_approved[0][1]


class TestBackwardCompatibility:
    def test_no_priors_no_statuses(self):
        """Original behavior is preserved when no priors/statuses are passed."""
        ledger = AtomLedger()
        slot = _slot()
        ranked = ledger.rank_candidates(slot, ["a", "b", "c"])
        assert len(ranked) == 3
        assert all(math.isinf(s) for _, s in ranked)

    def test_tried_atoms_without_priors(self):
        ledger = AtomLedger()
        slot = _slot()
        ledger.record(slot, "a", 10.0, 0)
        ledger.record(slot, "b", 90.0, 0)
        ranked = ledger.rank_candidates(slot, ["a", "b", "c"])
        scores = {name: score for name, score in ranked}
        # c untried → inf, a low gradient → high reward, b high gradient → low reward
        assert math.isinf(scores["c"])
        assert scores["a"] > scores["b"]
