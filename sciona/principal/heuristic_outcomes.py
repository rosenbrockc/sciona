"""Heuristic outcome memory extracted from persisted search traces."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from pydantic import BaseModel, Field

from sciona.heuristics import HeuristicActionClass


class HeuristicOutcomeRecord(BaseModel):
    """One persisted heuristic/action outcome from a proposal-selection trial."""

    family: str = ""
    heuristic_ids: list[str] = Field(default_factory=list)
    proposal_label: str = ""
    candidate_action_classes: list[str] = Field(default_factory=list)
    selected: bool = False
    loss_delta: float = 0.0


def extract_heuristic_outcomes(search_trace: list[dict[str, Any]]) -> list[HeuristicOutcomeRecord]:
    """Extract heuristic/action/loss records from persisted proposal traces."""
    records: list[HeuristicOutcomeRecord] = []
    for entry in search_trace:
        if not isinstance(entry, dict):
            continue
        proposal = entry.get("proposal_selection", {})
        if not isinstance(proposal, dict):
            continue
        baseline_loss = proposal.get("baseline_loss")
        try:
            baseline_loss_value = float(baseline_loss)
        except (TypeError, ValueError):
            continue
        selected_label = str(proposal.get("selected", "") or "")
        for candidate in proposal.get("candidates", []) or []:
            if not isinstance(candidate, dict):
                continue
            try:
                candidate_loss = float(candidate.get("loss"))
            except (TypeError, ValueError):
                continue
            evidence = candidate.get("evidence", {})
            if not isinstance(evidence, dict):
                evidence = {}
            heuristic_ids = [
                str(item)
                for item in evidence.get("heuristic_ids", []) or []
                if str(item)
            ]
            action_classes = [
                str(item)
                for item in evidence.get("candidate_action_classes", []) or []
                if str(item)
            ]
            if not heuristic_ids or not action_classes:
                continue
            records.append(
                HeuristicOutcomeRecord(
                    family=str(candidate.get("family", "") or ""),
                    heuristic_ids=heuristic_ids,
                    proposal_label=str(candidate.get("label", "") or ""),
                    candidate_action_classes=action_classes,
                    selected=str(candidate.get("label", "") or "") == selected_label,
                    loss_delta=baseline_loss_value - candidate_loss,
                )
            )
    return records


def summarize_heuristic_outcomes(records: list[HeuristicOutcomeRecord]) -> dict[str, Any]:
    """Summarize persisted heuristic outcomes for benchmark/reporting surfaces."""
    positive = [record for record in records if record.loss_delta > 0.0]
    selected = [record for record in records if record.selected]
    action_counts: Counter[str] = Counter()
    for record in positive:
        action_counts.update(record.candidate_action_classes)
    return {
        "outcome_count": len(records),
        "positive_outcome_count": len(positive),
        "selected_outcome_count": len(selected),
        "mean_positive_loss_delta": (
            float(sum(record.loss_delta for record in positive) / len(positive))
            if positive
            else 0.0
        ),
        "mature_action_classes": sorted(
            action_class
            for action_class, count in action_counts.items()
            if count >= 2
        ),
    }


def heuristic_action_bonus(
    *,
    family: str,
    heuristic_ids: list[str],
    search_trace: list[dict[str, Any]] | None,
) -> Counter[HeuristicActionClass]:
    """Return a cautious same-run action prior from repeated positive outcomes."""
    bonuses: Counter[HeuristicActionClass] = Counter()
    if not family or not heuristic_ids or not isinstance(search_trace, list):
        return bonuses

    gains_by_action: dict[HeuristicActionClass, list[float]] = defaultdict(list)
    heuristic_set = set(heuristic_ids)
    for record in extract_heuristic_outcomes(search_trace):
        if record.family != family:
            continue
        if not heuristic_set.intersection(record.heuristic_ids):
            continue
        if record.loss_delta <= 0.0:
            continue
        for raw in record.candidate_action_classes:
            try:
                gains_by_action[HeuristicActionClass(raw)].append(record.loss_delta)
            except ValueError:
                continue

    for action_class, gains in gains_by_action.items():
        if len(gains) < 2:
            continue
        mean_gain = sum(gains) / len(gains)
        if mean_gain > 0.0:
            bonuses[action_class] += 1
    return bonuses
