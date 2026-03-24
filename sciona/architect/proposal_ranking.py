"""Unified passive ranking for enrichment proposals.

Phase 3 adds a conservative scoring policy so primitive, template, and
skeleton proposals can be compared under one model without changing live
acceptance behavior yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from sciona.architect.proposal_models import EnrichmentProposal, ProposalType

_BASE_COMPLEXITY_PENALTY = {
    ProposalType.PRIMITIVE: 0.0,
    ProposalType.TEMPLATE: 0.15,
    ProposalType.SKELETON: 0.50,
}


@dataclass(frozen=True)
class ScoredProposal:
    """Proposal plus transparent score breakdown."""

    proposal: EnrichmentProposal
    objective_gain: float
    complexity_penalty: float
    risk_penalty: float
    prior_bonus: float
    score: float


def _complexity_penalty(proposal: EnrichmentProposal) -> float:
    base = _BASE_COMPLEXITY_PENALTY[proposal.proposal_type]
    return (
        base
        + 0.20 * proposal.delta_nodes
        + 0.10 * proposal.delta_edges
        + 0.35 * proposal.delta_family_count
        + 0.25 * proposal.delta_concept_type_count
    )


def _risk_penalty(proposal: EnrichmentProposal) -> float:
    compatibility = min(max(float(proposal.compatibility_score), 0.0), 1.0)
    confidence = min(max(float(proposal.confidence), 0.0), 1.0)
    compatibility_gap = 1.0 - compatibility
    confidence_gap = 1.0 - confidence
    if proposal.proposal_type == ProposalType.PRIMITIVE:
        return 0.20 * compatibility_gap
    if proposal.proposal_type == ProposalType.TEMPLATE:
        return 0.25 * compatibility_gap + 0.15 * confidence_gap
    return 0.30 * compatibility_gap + 0.10 * confidence_gap


def _prior_bonus(proposal: EnrichmentProposal, *, preferred_family: str = "") -> float:
    target = str(preferred_family or "").strip().lower()
    family = str(proposal.source_family or "").strip().lower()
    if target and family == target:
        return 0.10
    return 0.0


def score_proposal(
    proposal: EnrichmentProposal,
    *,
    preferred_family: str = "",
) -> ScoredProposal:
    """Score a proposal conservatively with an explicit complexity penalty."""
    payload_gain = proposal.payload.get("objective_gain")
    if isinstance(payload_gain, (int, float)):
        objective_gain = float(payload_gain)
    else:
        objective_gain = max(
            float(proposal.confidence),
            float(proposal.compatibility_score),
        )
    complexity_penalty = _complexity_penalty(proposal)
    risk_penalty = _risk_penalty(proposal)
    prior_bonus = _prior_bonus(proposal, preferred_family=preferred_family)
    score = objective_gain - complexity_penalty - risk_penalty + prior_bonus
    return ScoredProposal(
        proposal=proposal,
        objective_gain=objective_gain,
        complexity_penalty=complexity_penalty,
        risk_penalty=risk_penalty,
        prior_bonus=prior_bonus,
        score=score,
    )


def rank_proposals(
    proposals: list[EnrichmentProposal],
    *,
    preferred_family: str = "",
) -> list[ScoredProposal]:
    """Return proposals sorted best-first under the conservative ranking policy."""
    scored = [
        score_proposal(proposal, preferred_family=preferred_family)
        for proposal in proposals
    ]
    return sorted(
        scored,
        key=lambda row: (
            -row.score,
            row.proposal.proposal_type.value,
            row.proposal.source_label,
        ),
    )
