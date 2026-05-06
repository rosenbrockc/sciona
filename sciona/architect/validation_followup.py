"""Post-validation review reports for CDG enrichment work."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from sciona.architect.expansion_gap_mining import (
    ExpansionGapMiningReport,
    load_validation_results,
    mine_expansion_gaps,
)


@dataclass(frozen=True)
class TrickReviewTicket:
    """One validation case where the tricks catalog is relevant for review."""

    competition_id: str
    title: str
    assessment_without_tricks: str
    assessment_with_trick_availability: str
    family: str
    template: str
    missing_techniques: tuple[str, ...]
    candidate_trick_ids: tuple[str, ...]
    high_risk_trick_ids: tuple[str, ...]
    recommended_action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "competition_id": self.competition_id,
            "title": self.title,
            "assessment_without_tricks": self.assessment_without_tricks,
            "assessment_with_trick_availability": self.assessment_with_trick_availability,
            "family": self.family,
            "template": self.template,
            "missing_techniques": list(self.missing_techniques),
            "candidate_trick_ids": list(self.candidate_trick_ids),
            "high_risk_trick_ids": list(self.high_risk_trick_ids),
            "recommended_action": self.recommended_action,
        }


@dataclass(frozen=True)
class ValidationFollowupReport:
    """Actionable follow-up report after a validation pass."""

    total_results: int
    assessment_counts: dict[str, int]
    base_assessment_counts: dict[str, int]
    assessment_with_trick_availability_counts: dict[str, int]
    rescued_by_expansion_count: int
    trick_review_tickets: tuple[TrickReviewTicket, ...]
    divergent_family_counts: dict[str, int]
    divergent_gap_report: ExpansionGapMiningReport

    @property
    def trick_review_ticket_count(self) -> int:
        return len(self.trick_review_tickets)

    @property
    def divergent_count(self) -> int:
        return int(self.assessment_counts.get("divergent", 0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_results": self.total_results,
            "assessment_counts": dict(self.assessment_counts),
            "base_assessment_counts": dict(self.base_assessment_counts),
            "assessment_with_trick_availability_counts": dict(
                self.assessment_with_trick_availability_counts
            ),
            "rescued_by_expansion_count": self.rescued_by_expansion_count,
            "trick_review_ticket_count": self.trick_review_ticket_count,
            "trick_review_tickets": [
                ticket.to_dict() for ticket in self.trick_review_tickets
            ],
            "divergent_count": self.divergent_count,
            "divergent_family_counts": dict(self.divergent_family_counts),
            "divergent_gap_report": self.divergent_gap_report.to_dict(),
        }


def build_validation_followup_report(
    validation_results: Iterable[dict[str, Any]],
    *,
    min_support: int = 2,
    similarity_threshold: float = 0.45,
    max_tickets: int = 100,
    max_clusters: int = 50,
) -> ValidationFollowupReport:
    """Build review queues from validation results without changing grading."""
    results = tuple(validation_results)
    assessment_counts = Counter(str(result.get("assessment", "")) for result in results)
    base_assessment_counts = Counter(
        str(result.get("base_assessment", "")) for result in results
    )
    trick_counts = Counter(
        str(result.get("assessment_with_trick_availability") or result.get("assessment", ""))
        for result in results
    )
    divergent_family_counts = Counter(
        _top_match(result).get("family", "")
        for result in results
        if result.get("assessment") == "divergent"
    )
    divergent_family_counts.pop("", None)
    tickets = tuple(
        sorted(
            (_build_trick_ticket(result) for result in results),
            key=_ticket_sort_key,
        )[:max_tickets]
    )
    tickets = tuple(ticket for ticket in tickets if ticket is not None)
    divergent_gap_report = mine_expansion_gaps(
        results,
        min_support=min_support,
        similarity_threshold=similarity_threshold,
        include_assessments=("divergent",),
        max_clusters=max_clusters,
    )
    return ValidationFollowupReport(
        total_results=len(results),
        assessment_counts=dict(assessment_counts),
        base_assessment_counts=dict(base_assessment_counts),
        assessment_with_trick_availability_counts=dict(trick_counts),
        rescued_by_expansion_count=sum(
            1 for result in results if result.get("rescued_by_expansion")
        ),
        trick_review_tickets=tickets,
        divergent_family_counts=dict(
            sorted(
                divergent_family_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        divergent_gap_report=divergent_gap_report,
    )


def build_validation_followup_report_from_paths(
    paths: Iterable[str],
    **kwargs: Any,
) -> ValidationFollowupReport:
    """Load validation files and build a follow-up report."""
    return build_validation_followup_report(load_validation_results(paths), **kwargs)


def _build_trick_ticket(result: dict[str, Any]) -> TrickReviewTicket | None:
    telemetry = result.get("trick_telemetry")
    if not isinstance(telemetry, dict):
        return None
    candidate_tricks = tuple(
        str(item.get("trick_id", ""))
        for item in telemetry.get("candidate_tricks", [])
        if isinstance(item, dict) and item.get("trick_id")
    )
    high_risk_tricks = tuple(
        str(item.get("trick_id", ""))
        for item in telemetry.get("suppressed_high_risk_tricks", [])
        if isinstance(item, dict) and item.get("trick_id")
    )
    if not candidate_tricks and not high_risk_tricks:
        return None
    top_match = _top_match(result)
    evaluation = _evaluation(result)
    return TrickReviewTicket(
        competition_id=str(result.get("competition_id", "")),
        title=str(result.get("title", "")),
        assessment_without_tricks=str(
            result.get("assessment_without_tricks") or result.get("assessment", "")
        ),
        assessment_with_trick_availability=str(
            result.get("assessment_with_trick_availability")
            or result.get("assessment", "")
        ),
        family=str(top_match.get("family", "")),
        template=str(top_match.get("template", "")),
        missing_techniques=tuple(
            str(item) for item in evaluation.get("missing_techniques", []) if item
        ),
        candidate_trick_ids=candidate_tricks,
        high_risk_trick_ids=high_risk_tricks,
        recommended_action=_trick_ticket_action(result, candidate_tricks, high_risk_tricks),
    )


def _trick_ticket_action(
    result: dict[str, Any],
    candidate_tricks: tuple[str, ...],
    high_risk_tricks: tuple[str, ...],
) -> str:
    assessment = str(result.get("assessment_without_tricks") or result.get("assessment", ""))
    if candidate_tricks and assessment == "divergent":
        return "review_for_expansion_refinement_or_metadata"
    if candidate_tricks:
        return "review_for_metadata_or_prompt_context"
    if high_risk_tricks:
        return "keep_suppressed_and_review_for_detection_only"
    return "no_action"


def _ticket_sort_key(ticket: TrickReviewTicket | None) -> tuple[int, str]:
    if ticket is None:
        return (99, "")
    action_rank = {
        "review_for_expansion_refinement_or_metadata": 0,
        "review_for_metadata_or_prompt_context": 1,
        "keep_suppressed_and_review_for_detection_only": 2,
    }.get(ticket.recommended_action, 9)
    return (action_rank, ticket.competition_id)


def _evaluation(result: dict[str, Any]) -> dict[str, Any]:
    evaluation = result.get("evaluation") or result.get("base_evaluation") or {}
    return evaluation if isinstance(evaluation, dict) else {}


def _top_match(result: dict[str, Any]) -> dict[str, Any]:
    matches = result.get("template_matches") or []
    if isinstance(matches, list) and matches and isinstance(matches[0], dict):
        return matches[0]
    return {}
