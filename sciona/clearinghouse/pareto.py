"""Pareto front computation for multi-objective bounties."""

from __future__ import annotations

from typing import Sequence

from sciona.clearinghouse.models import ObjectiveSpec, WinningCDG


def dominates(
    a: dict[str, float],
    b: dict[str, float],
    objectives: Sequence[ObjectiveSpec],
) -> bool:
    """Return True if submission *a* Pareto-dominates submission *b*.

    A dominates B if A is at least as good on all objectives and strictly
    better on at least one.
    """
    at_least_as_good = True
    strictly_better = False

    for obj in objectives:
        va = a.get(obj.metric, float("inf"))
        vb = b.get(obj.metric, float("inf"))

        if obj.direction == "minimize":
            if va > vb:
                at_least_as_good = False
            if va < vb:
                strictly_better = True
        else:  # maximize
            if va < vb:
                at_least_as_good = False
            if va > vb:
                strictly_better = True

    return at_least_as_good and strictly_better


def compute_pareto_front(
    submissions: Sequence[WinningCDG],
    objectives: Sequence[ObjectiveSpec],
    max_winners: int = 3,
) -> list[WinningCDG]:
    """Return up to *max_winners* Pareto-optimal submissions.

    Parameters
    ----------
    submissions
        All verified submissions with their metric values.
    objectives
        The optimization objectives with direction and weight.
    max_winners
        Maximum number of Pareto-optimal submissions to return.

    Returns
    -------
    list[WinningCDG]
        Pareto-optimal submissions, up to *max_winners*.
    """
    if not submissions or not objectives:
        return []

    # Find non-dominated solutions
    pareto: list[WinningCDG] = []

    for candidate in submissions:
        is_dominated = False
        new_pareto: list[WinningCDG] = []

        for existing in pareto:
            if dominates(existing.metric_values, candidate.metric_values, objectives):
                is_dominated = True
                new_pareto.append(existing)
            elif dominates(candidate.metric_values, existing.metric_values, objectives):
                # candidate dominates existing — drop existing
                continue
            else:
                new_pareto.append(existing)

        if not is_dominated:
            new_pareto.append(candidate)

        pareto = new_pareto

    # Cap at max_winners, preferring by scalarized score
    if len(pareto) > max_winners:
        pareto.sort(key=lambda s: _scalarize(s.metric_values, objectives))
        pareto = pareto[:max_winners]

    return pareto


def _scalarize(
    metrics: dict[str, float],
    objectives: Sequence[ObjectiveSpec],
) -> float:
    """Compute weighted scalarization for tie-breaking.

    Lower is better (flip sign for maximize objectives).
    """
    score = 0.0
    for obj in objectives:
        v = metrics.get(obj.metric, 0.0)
        if obj.direction == "maximize":
            v = -v
        score += v * obj.weight
    return score


def split_architect_payout(
    winners: Sequence[WinningCDG],
    objectives: Sequence[ObjectiveSpec],
) -> dict[str, float]:
    """Split architect payout weights among Pareto winners.

    Returns a mapping of submission_id to normalized weight (sums to 1.0).
    Uses objective weights for scalarization-based ranking.
    """
    if not winners:
        return {}

    if len(winners) == 1:
        return {winners[0].submission_id: 1.0}

    # Use inverse scalarized score as weight (lower score = higher weight)
    scores = {
        w.submission_id: _scalarize(w.metric_values, objectives)
        for w in winners
    }

    # Shift so all scores are positive
    min_score = min(scores.values())
    shifted = {k: max(0.001, -(v - min_score - 1)) for k, v in scores.items()}

    total = sum(shifted.values())
    return {k: v / total for k, v in shifted.items()}
