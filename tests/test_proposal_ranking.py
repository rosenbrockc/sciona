from __future__ import annotations

from sciona.architect.proposal_models import (
    EnrichmentProposal,
    ProposalType,
    proposal_placeholder_skeleton,
)
from sciona.architect.proposal_ranking import rank_proposals


def test_primitive_beats_skeleton_when_gains_are_similar() -> None:
    primitive = EnrichmentProposal(
        proposal_type=ProposalType.PRIMITIVE,
        source_family="signal_filter",
        source_label="primitive",
        confidence=0.78,
        compatibility_score=0.80,
        matched_primitive="sciona.atoms.signal.filter_signal_basic",
    )
    skeleton = proposal_placeholder_skeleton(
        skeleton_name="signal_detect_measure",
        source_family="signal_filter",
        source_label="signal_detect_measure",
        confidence=0.82,
        compatibility_score=0.82,
        delta_nodes=3,
        delta_edges=2,
        delta_family_count=1,
        delta_concept_type_count=1,
    )

    ranked = rank_proposals([skeleton, primitive], preferred_family="signal_filter")
    assert ranked[0].proposal.proposal_type == ProposalType.PRIMITIVE


def test_template_beats_skeleton_when_materially_simpler() -> None:
    template = EnrichmentProposal(
        proposal_type=ProposalType.TEMPLATE,
        source_family="signal_filter",
        source_label="template",
        confidence=0.79,
        compatibility_score=0.81,
        delta_nodes=1,
        delta_edges=1,
        delta_family_count=1,
        delta_concept_type_count=1,
        template_fqn="sciona.atoms.templates.lightweight_cleanup",
    )
    skeleton = proposal_placeholder_skeleton(
        skeleton_name="signal_detect_measure",
        source_family="signal_filter",
        source_label="signal_detect_measure",
        confidence=0.83,
        compatibility_score=0.83,
        delta_nodes=3,
        delta_edges=2,
        delta_family_count=1,
        delta_concept_type_count=1,
    )

    ranked = rank_proposals([skeleton, template], preferred_family="signal_filter")
    assert ranked[0].proposal.proposal_type == ProposalType.TEMPLATE


def test_skeleton_can_win_when_gain_is_materially_better() -> None:
    primitive = EnrichmentProposal(
        proposal_type=ProposalType.PRIMITIVE,
        source_family="signal_filter",
        source_label="primitive",
        confidence=0.55,
        compatibility_score=0.60,
        matched_primitive="sciona.atoms.signal.filter_signal_basic",
    )
    skeleton = proposal_placeholder_skeleton(
        skeleton_name="signal_detect_measure",
        source_family="signal_filter",
        source_label="signal_detect_measure",
        confidence=1.0,
        compatibility_score=1.0,
        delta_nodes=3,
        delta_edges=2,
        delta_family_count=1,
        delta_concept_type_count=1,
        payload={"objective_gain": 3.0},
    )

    ranked = rank_proposals([primitive, skeleton], preferred_family="signal_filter")
    assert ranked[0].proposal.proposal_type == ProposalType.SKELETON


def test_same_family_prior_does_not_overcome_large_complexity_gap() -> None:
    simpler_cross_family = EnrichmentProposal(
        proposal_type=ProposalType.TEMPLATE,
        source_family="analysis",
        source_label="cross_family_template",
        confidence=0.88,
        compatibility_score=0.88,
        delta_nodes=1,
        delta_edges=1,
        delta_family_count=1,
        delta_concept_type_count=1,
        template_fqn="sciona.atoms.templates.cross_family",
    )
    heavier_same_family = proposal_placeholder_skeleton(
        skeleton_name="signal_detect_measure",
        source_family="signal_filter",
        source_label="signal_detect_measure",
        confidence=0.90,
        compatibility_score=0.90,
        delta_nodes=4,
        delta_edges=3,
        delta_family_count=1,
        delta_concept_type_count=2,
    )

    ranked = rank_proposals(
        [heavier_same_family, simpler_cross_family],
        preferred_family="signal_filter",
    )
    assert ranked[0].proposal.source_label == "cross_family_template"
