from __future__ import annotations

from unittest.mock import MagicMock

from ageom.architect.handoff import CDGExport
from ageom.architect.models import AlgorithmicNode, AlgorithmicPrimitive, ConceptType, IOSpec, NodeStatus
from ageom.principal.atom_ledger import AtomLedger, compute_slot_signature
from ageom.principal.variant_mutation import (
    LedgerVariantFamily,
    SignalEventRateVariantFamily,
    maybe_apply_bottleneck_variant,
)


def _atomic_node(
    node_id: str,
    name: str,
    *,
    matched_primitive: str | None,
) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id,
        name=name,
        description=name,
        concept_type=ConceptType.SIGNAL_FILTER,
        status=NodeStatus.ATOMIC,
        matched_primitive=matched_primitive,
    )


def test_signal_event_rate_family_matches_curated_scaffold() -> None:
    family = SignalEventRateVariantFamily()
    cdg = CDGExport(
        nodes=[
            _atomic_node(
                "n1",
                "Filter Signal For Detection",
                matched_primitive="filter_signal_for_detection",
            ),
            _atomic_node(
                "n2",
                "Detect Peaks In Signal",
                matched_primitive="detect_peaks_in_signal",
            ),
            _atomic_node(
                "n3",
                "Compute Event Rate",
                matched_primitive="compute_event_rate",
            ),
        ],
        edges=[],
    )

    assert family.matches(cdg) is True


def test_signal_event_rate_family_rejects_unrelated_scaffold() -> None:
    family = SignalEventRateVariantFamily()
    cdg = CDGExport(
        nodes=[
            _atomic_node(
                "n1",
                "Unrelated Stage",
                matched_primitive="some_other_primitive",
            ),
        ],
        edges=[],
    )

    assert family.matches(cdg) is False


def test_maybe_apply_bottleneck_variant_returns_family_metadata() -> None:
    cdg = CDGExport(
        nodes=[
            _atomic_node(
                "n1",
                "Filter Signal For Detection",
                matched_primitive="filter_signal_for_detection",
            ),
            _atomic_node(
                "n2",
                "Detect Peaks In Signal",
                matched_primitive="detect_peaks_in_signal",
            ),
            _atomic_node(
                "n3",
                "Compute Event Rate",
                matched_primitive="compute_event_rate",
            ),
        ],
        edges=[],
    )

    result = maybe_apply_bottleneck_variant(
        cdg,
        bottleneck_name="Compute Event Rate",
    )

    assert result.applied is True
    assert result.family == "signal_event_rate"
    assert result.variant_name == "compute_event_rate_smoothed"
    updated = {
        node.name: node.matched_primitive
        for node in result.cdg.nodes
        if node.status == NodeStatus.ATOMIC
    }
    assert updated["Compute Event Rate"] == "compute_event_rate_smoothed"
    assert updated["Detect Peaks In Signal"] == "detect_peaks_in_signal"


def test_maybe_apply_bottleneck_variant_is_noop_without_family_variant() -> None:
    cdg = CDGExport(
        nodes=[
            _atomic_node(
                "n1",
                "Filter Signal For Detection",
                matched_primitive="filter_signal_for_detection",
            ),
            _atomic_node(
                "n2",
                "Detect Peaks In Signal",
                matched_primitive="detect_peaks_in_signal",
            ),
        ],
        edges=[],
    )

    result = maybe_apply_bottleneck_variant(
        cdg,
        bottleneck_name="Detect Peaks In Signal",
    )

    assert result.applied is False
    assert result.family == "signal_event_rate"
    assert result.variant_name is None
    assert result.allow_redecompose is False
    assert result.cdg == cdg


# ---------------------------------------------------------------------------
# Ledger bandit family tests
# ---------------------------------------------------------------------------


def _generic_node(
    node_id: str,
    name: str,
    *,
    matched_primitive: str | None,
    parent_id: str | None = None,
    concept_type: ConceptType = ConceptType.CUSTOM,
) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id,
        name=name,
        description=name,
        concept_type=concept_type,
        status=NodeStatus.ATOMIC,
        parent_id=parent_id,
        inputs=[IOSpec(name="x", type_desc="np.ndarray")],
        outputs=[IOSpec(name="y", type_desc="np.ndarray")],
        matched_primitive=matched_primitive,
    )


def _mock_catalog_with(*primitive_names: str, concept_type: ConceptType = ConceptType.CUSTOM) -> MagicMock:
    catalog = MagicMock()
    prims = []
    for name in primitive_names:
        prim = MagicMock()
        prim.name = name
        prim.category = concept_type
        prims.append(prim)
    catalog.search_by_category.return_value = prims
    return catalog


def test_ledger_variant_family_swaps_atom() -> None:
    ledger = AtomLedger()
    catalog = _mock_catalog_with("atom_a", "atom_b")

    cdg = CDGExport(
        nodes=[_generic_node("n1", "Step", matched_primitive="atom_a")],
        edges=[],
    )

    # Record: atom_a is bad, atom_b is good
    node = cdg.nodes[0]
    slot = compute_slot_signature(node, None)
    for t in range(5):
        ledger.record(slot, "atom_a", 80.0, trial=t)
        ledger.record(slot, "atom_b", 10.0, trial=t)

    result = maybe_apply_bottleneck_variant(
        cdg,
        bottleneck_name="Step",
        atom_ledger=ledger,
        catalog=catalog,
    )

    assert result.applied is True
    assert result.family == "ledger_bandit"
    assert result.variant_name == "atom_b"
    assert result.cdg.nodes[0].matched_primitive == "atom_b"


def test_ledger_variant_family_noop_when_current_is_best() -> None:
    ledger = AtomLedger()
    catalog = _mock_catalog_with("atom_a", "atom_b")

    cdg = CDGExport(
        nodes=[_generic_node("n1", "Step", matched_primitive="atom_a")],
        edges=[],
    )

    node = cdg.nodes[0]
    slot = compute_slot_signature(node, None)
    for t in range(5):
        ledger.record(slot, "atom_a", 10.0, trial=t)
        ledger.record(slot, "atom_b", 80.0, trial=t)

    result = maybe_apply_bottleneck_variant(
        cdg,
        bottleneck_name="Step",
        atom_ledger=ledger,
        catalog=catalog,
    )

    assert result.applied is False
    assert result.family == "ledger_bandit"
    assert result.allow_redecompose is True


def test_curated_family_takes_priority_over_ledger() -> None:
    ledger = AtomLedger()
    catalog = _mock_catalog_with(
        "compute_event_rate", "compute_event_rate_smoothed",
        concept_type=ConceptType.SIGNAL_FILTER,
    )

    # Build a signal_event_rate scaffold — curated family should match first
    cdg = CDGExport(
        nodes=[
            _atomic_node("n1", "Filter Signal For Detection", matched_primitive="filter_signal_for_detection"),
            _atomic_node("n2", "Detect Peaks In Signal", matched_primitive="detect_peaks_in_signal"),
            _atomic_node("n3", "Compute Event Rate", matched_primitive="compute_event_rate"),
        ],
        edges=[],
    )

    result = maybe_apply_bottleneck_variant(
        cdg,
        bottleneck_name="Compute Event Rate",
        atom_ledger=ledger,
        catalog=catalog,
    )

    # Curated family fires, not ledger
    assert result.applied is True
    assert result.family == "signal_event_rate"


def test_ledger_fires_after_curated_family_exhausted() -> None:
    """When the curated family has no variant left, the ledger should still get a turn."""
    ledger = AtomLedger()
    # Catalog returns two alternatives in the same category
    catalog = _mock_catalog_with(
        "compute_event_rate_smoothed", "compute_event_rate_windowed",
        concept_type=ConceptType.SIGNAL_FILTER,
    )

    # Scaffold where the curated variant was already applied (smoothed).
    # The curated family has no further variant for smoothed, so it will
    # return applied=False, allow_redecompose=False.
    cdg = CDGExport(
        nodes=[
            _atomic_node("n1", "Filter Signal For Detection", matched_primitive="filter_signal_for_detection"),
            _atomic_node("n2", "Detect Peaks In Signal", matched_primitive="detect_peaks_in_signal"),
            _atomic_node("n3", "Compute Event Rate", matched_primitive="compute_event_rate_smoothed"),
        ],
        edges=[],
    )

    # Record ledger observations: smoothed is bad, windowed is good
    node = cdg.nodes[2]
    slot = compute_slot_signature(node, None)
    for t in range(5):
        ledger.record(slot, "compute_event_rate_smoothed", 75.0, trial=t)
        ledger.record(slot, "compute_event_rate_windowed", 15.0, trial=t)

    result = maybe_apply_bottleneck_variant(
        cdg,
        bottleneck_name="Compute Event Rate",
        atom_ledger=ledger,
        catalog=catalog,
    )

    # Ledger fires because curated family was exhausted
    assert result.applied is True
    assert result.family == "ledger_bandit"
    assert result.variant_name == "compute_event_rate_windowed"


def test_exhausted_curated_allows_redecompose_with_ledger() -> None:
    """When curated is exhausted and ledger has no better atom, allow_redecompose should be True."""
    ledger = AtomLedger()
    catalog = _mock_catalog_with(
        "compute_event_rate_smoothed",
        concept_type=ConceptType.SIGNAL_FILTER,
    )

    cdg = CDGExport(
        nodes=[
            _atomic_node("n1", "Filter Signal For Detection", matched_primitive="filter_signal_for_detection"),
            _atomic_node("n2", "Detect Peaks In Signal", matched_primitive="detect_peaks_in_signal"),
            _atomic_node("n3", "Compute Event Rate", matched_primitive="compute_event_rate_smoothed"),
        ],
        edges=[],
    )

    # Ledger says current atom is already best (only candidate)
    node = cdg.nodes[2]
    slot = compute_slot_signature(node, None)
    ledger.record(slot, "compute_event_rate_smoothed", 30.0, trial=1)

    result = maybe_apply_bottleneck_variant(
        cdg,
        bottleneck_name="Compute Event Rate",
        atom_ledger=ledger,
        catalog=catalog,
    )

    # Not applied, but the ledger's allow_redecompose=True overrides
    # the curated family's allow_redecompose=False
    assert result.applied is False
    assert result.family == "ledger_bandit"
    assert result.allow_redecompose is True


def test_backward_compatible_without_ledger() -> None:
    cdg = CDGExport(
        nodes=[_generic_node("n1", "Step", matched_primitive="some_atom")],
        edges=[],
    )

    # No ledger, no catalog — should not error, just return no-op
    result = maybe_apply_bottleneck_variant(
        cdg,
        bottleneck_name="Step",
    )

    assert result.applied is False
    assert result.family is None
