from __future__ import annotations

from ageom.architect.handoff import CDGExport
from ageom.architect.models import AlgorithmicNode, ConceptType, NodeStatus
from ageom.principal.variant_mutation import (
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
