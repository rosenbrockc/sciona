from __future__ import annotations

import pytest

from sciona.heuristics import (
    CanonicalHeuristic,
    HeuristicActionClass,
    HeuristicApplicabilityScope,
    HeuristicEvidenceType,
    HeuristicProducerKind,
    canonical_heuristic_from_metric,
    compatibility_hint_for_metric,
    known_heuristic_ids,
    known_heuristic_compatibility_hints,
)


def test_canonical_heuristic_accepts_cross_family_identifier() -> None:
    heuristic = CanonicalHeuristic(
        heuristic_id="interval_instability",
        display_name="Interval Instability",
        dejargonized_meaning="Observed spacing between extracted events is unstable.",
        evidence_type=HeuristicEvidenceType.SCALAR_SCORE,
        confidence=0.7,
        producer_kind=HeuristicProducerKind.RUNTIME_TRANSFORM,
        applicability_scope=HeuristicApplicabilityScope.CROSS_FAMILY,
        supported_action_classes=[
            HeuristicActionClass.INSERT_CORRECTION,
            HeuristicActionClass.SMOOTH_OR_AGGREGATE,
        ],
    )
    assert heuristic.heuristic_id == "interval_instability"


def test_cross_family_identifier_rejects_domain_jargon() -> None:
    with pytest.raises(ValueError):
        CanonicalHeuristic(
            heuristic_id="rr_irregularity",
            display_name="RR Irregularity",
            dejargonized_meaning="Domain-specific shorthand should not be canonical.",
            evidence_type=HeuristicEvidenceType.SCALAR_SCORE,
            applicability_scope=HeuristicApplicabilityScope.CROSS_FAMILY,
        )


def test_compatibility_hint_maps_signal_specific_metric_to_generic_heuristic() -> None:
    hint = compatibility_hint_for_metric(
        "signal_quality_variance", source_domain="signal_event_rate"
    )
    assert hint is not None
    assert hint.heuristic_id == "quality_instability"
    assert hint.source_domain == "signal_event_rate"
    assert HeuristicActionClass.GATE_OR_VALIDATE in hint.supported_action_classes


def test_compatibility_hint_maps_non_signal_metric_to_generic_heuristic() -> None:
    hint = compatibility_hint_for_metric(
        "innovation_whiteness_pvalue", source_domain="sequential_filter"
    )
    assert hint is not None
    assert hint.heuristic_id == "residual_structure_after_transform"
    assert HeuristicActionClass.REPLACE_STAGE in hint.supported_action_classes


def test_canonical_heuristic_from_metric_preserves_cross_family_scope() -> None:
    heuristic = canonical_heuristic_from_metric(
        "interval_outlier_fraction",
        source_domain="signal_event_rate",
        confidence=0.6,
    )
    assert heuristic is not None
    assert heuristic.heuristic_id == "interval_instability"
    assert heuristic.applicability_scope == HeuristicApplicabilityScope.CROSS_FAMILY
    assert heuristic.producer_kind == HeuristicProducerKind.COMPATIBILITY_MAPPING
    assert heuristic.family_notes == ["source_domain:signal_event_rate"]


def test_unknown_metric_returns_none() -> None:
    assert compatibility_hint_for_metric("totally_unknown_metric") is None
    assert canonical_heuristic_from_metric("totally_unknown_metric") is None


def test_known_hints_cover_multiple_action_classes() -> None:
    hints = known_heuristic_compatibility_hints()
    action_classes = {
        action_class
        for hint in hints
        for action_class in hint.supported_action_classes
    }
    assert HeuristicActionClass.PRECONDITION in action_classes
    assert HeuristicActionClass.GATE_OR_VALIDATE in action_classes
    assert HeuristicActionClass.BRANCH_AND_COMPARE in action_classes


def test_canonical_heuristic_round_trip() -> None:
    heuristic = canonical_heuristic_from_metric("condition_number")
    assert heuristic is not None
    restored = CanonicalHeuristic.model_validate(heuristic.model_dump(mode="json"))
    assert restored.heuristic_id == "numerical_condition_instability"


def test_known_heuristic_ids_include_external_ageo_atoms_registry_entries() -> None:
    ids = set(known_heuristic_ids())

    assert "split_balance_instability" in ids
    assert "recursion_depth_pressure" in ids
