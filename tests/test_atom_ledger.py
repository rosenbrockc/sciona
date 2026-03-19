"""Tests for the Atom Performance Ledger and UCB1 bandit ranking."""

from __future__ import annotations

import math

from ageom.architect.models import AlgorithmicNode, ConceptType, IOSpec, NodeStatus
from ageom.principal.atom_ledger import (
    AtomLedger,
    SlotSignature,
    compute_slot_signature,
)


def _make_node(
    node_id: str = "n1",
    name: str = "filter_signal",
    concept_type: ConceptType = ConceptType.SIGNAL_FILTER,
    parent_id: str | None = None,
    inputs: list[IOSpec] | None = None,
    outputs: list[IOSpec] | None = None,
    matched_primitive: str | None = "filter_signal_for_detection",
) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id,
        name=name,
        description=name,
        concept_type=concept_type,
        status=NodeStatus.ATOMIC,
        parent_id=parent_id,
        inputs=inputs or [IOSpec(name="signal", type_desc="np.ndarray")],
        outputs=outputs or [IOSpec(name="filtered", type_desc="np.ndarray")],
        matched_primitive=matched_primitive,
    )


def _slot() -> SlotSignature:
    return SlotSignature(
        parent_name="process_signal",
        concept_type="signal_filter",
        input_types=("np.ndarray",),
        output_types=("np.ndarray",),
    )


def test_slot_signature_deterministic() -> None:
    s1 = _slot()
    s2 = SlotSignature(
        parent_name="process_signal",
        concept_type="signal_filter",
        input_types=("np.ndarray",),
        output_types=("np.ndarray",),
    )
    assert s1.key == s2.key

    s3 = SlotSignature(
        parent_name="other_parent",
        concept_type="signal_filter",
        input_types=("np.ndarray",),
        output_types=("np.ndarray",),
    )
    assert s1.key != s3.key


def test_slot_signature_ignores_node_id() -> None:
    parent = _make_node(node_id="p1", name="Process Signal", concept_type=ConceptType.CUSTOM)
    node_a = _make_node(node_id="a1", parent_id="p1")
    node_b = _make_node(node_id="b2", parent_id="p1")

    sig_a = compute_slot_signature(node_a, parent)
    sig_b = compute_slot_signature(node_b, parent)
    assert sig_a.key == sig_b.key


def test_ledger_record_and_count() -> None:
    ledger = AtomLedger()
    slot = _slot()

    ledger.record(slot, "atom_a", 30.0, trial=1)
    ledger.record(slot, "atom_a", 25.0, trial=2)
    ledger.record(slot, "atom_b", 80.0, trial=1)

    assert ledger.observation_count(slot, "atom_a") == 2
    assert ledger.observation_count(slot, "atom_b") == 1
    assert ledger.total_observations_for_slot(slot) == 3


def test_ucb1_untried_atoms_get_inf() -> None:
    ledger = AtomLedger()
    slot = _slot()

    ledger.record(slot, "atom_a", 50.0, trial=1)

    ranked = ledger.rank_candidates(slot, ["atom_a", "atom_b"])
    assert ranked[0][0] == "atom_b"
    assert ranked[0][1] == float("inf")


def test_ucb1_prefers_lower_gradient() -> None:
    ledger = AtomLedger()
    slot = _slot()

    # Record multiple observations to dilute exploration bonus
    for t in range(10):
        ledger.record(slot, "atom_bad", 80.0, trial=t)
        ledger.record(slot, "atom_good", 20.0, trial=t)

    ranked = ledger.rank_candidates(slot, ["atom_bad", "atom_good"])
    assert ranked[0][0] == "atom_good"
    assert ranked[1][0] == "atom_bad"


def test_ucb1_exploration_bonus() -> None:
    ledger = AtomLedger()
    slot = _slot()

    # atom_a tried many times, atom_b tried once — both similar reward
    for t in range(20):
        ledger.record(slot, "atom_a", 40.0, trial=t)
    ledger.record(slot, "atom_b", 40.0, trial=0)

    ranked = ledger.rank_candidates(slot, ["atom_a", "atom_b"])
    # atom_b should have higher UCB score due to exploration bonus
    scores = {name: score for name, score in ranked}
    assert scores["atom_b"] > scores["atom_a"]


def test_rank_candidates_empty_ledger() -> None:
    ledger = AtomLedger()
    slot = _slot()

    ranked = ledger.rank_candidates(slot, ["atom_a", "atom_b", "atom_c"])
    assert len(ranked) == 3
    assert all(score == float("inf") for _, score in ranked)


def test_rank_candidates_single_candidate() -> None:
    ledger = AtomLedger()
    slot = _slot()

    ranked = ledger.rank_candidates(slot, ["atom_a"])
    assert len(ranked) == 1
    assert ranked[0][0] == "atom_a"


def test_reward_clamping() -> None:
    ledger = AtomLedger()
    slot = _slot()

    # Gradient > 100 should be clamped to reward 0.0
    ledger.record(slot, "atom_over", 150.0, trial=1)
    # Gradient < 0 should be clamped to reward 1.0
    ledger.record(slot, "atom_under", -50.0, trial=1)

    ranked = ledger.rank_candidates(slot, ["atom_over", "atom_under"])
    scores = {name: score for name, score in ranked}
    # atom_under (reward ~1.0) should rank higher than atom_over (reward ~0.0)
    assert scores["atom_under"] > scores["atom_over"]


def test_compute_slot_signature_no_parent() -> None:
    node = _make_node(node_id="n1", parent_id=None)
    sig = compute_slot_signature(node, parent=None)
    assert sig.parent_name == ""
    assert sig.key  # should still produce a valid key
