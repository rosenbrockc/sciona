"""Verification engine — budget tracking, improvement threshold, flow orchestration."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

from sciona.clearinghouse.models import (
    BestScore,
    VerificationBudget,
    VerificationRun,
)


def should_trigger_verification(
    claimed_value: float,
    best_to_date: float,
    threshold_pct: float = 5.0,
    direction: str = "minimize",
) -> bool:
    """Determine whether a claimed metric value warrants verification.

    Parameters
    ----------
    claimed_value
        The metric value claimed by the submission.
    best_to_date
        The current best verified metric value.
    threshold_pct
        Minimum improvement percentage required (default 5%).
    direction
        ``"minimize"`` or ``"maximize"``.

    Returns
    -------
    bool
        True if the improvement exceeds the threshold.
    """
    if best_to_date == 0:
        return claimed_value != 0

    if direction == "minimize":
        improvement = (best_to_date - claimed_value) / abs(best_to_date)
    else:
        improvement = (claimed_value - best_to_date) / abs(best_to_date)

    return improvement >= threshold_pct / 100.0


def consume_verification_slot(budget: VerificationBudget) -> VerificationBudget:
    """Consume one verification slot, returning the updated budget.

    Raises
    ------
    ValueError
        If no slots remain.
    """
    if budget.used_slots >= budget.total_slots:
        raise ValueError(
            f"Verification budget exhausted: {budget.used_slots}/{budget.total_slots} slots used"
        )

    return VerificationBudget(
        bounty_id=budget.bounty_id,
        tier=budget.tier,
        total_slots=budget.total_slots,
        used_slots=budget.used_slots + 1,
        cost_per_extra=budget.cost_per_extra,
        overhead_deposit=budget.overhead_deposit,
        overhead_used=budget.overhead_used,
    )


def remaining_slots(budget: VerificationBudget) -> int:
    """Return the number of remaining verification slots."""
    return max(0, budget.total_slots - budget.used_slots)


def update_best_score(
    current: BestScore | None,
    new_value: float,
    submission_id: str,
    direction: str = "minimize",
) -> BestScore | None:
    """Return an updated BestScore if the new value is better, else None.

    Parameters
    ----------
    current
        The current best score (None if no baseline exists).
    new_value
        The newly verified metric value.
    submission_id
        The submission that produced this value.
    direction
        ``"minimize"`` or ``"maximize"``.
    """
    if current is None:
        return BestScore(
            bounty_id="",
            metric_name="",
            best_value=new_value,
            best_submission_id=submission_id,
        )

    is_better = (
        new_value < current.best_value
        if direction == "minimize"
        else new_value > current.best_value
    )

    if is_better:
        return BestScore(
            bounty_id=current.bounty_id,
            metric_name=current.metric_name,
            best_value=new_value,
            best_submission_id=submission_id,
        )

    return None


def compute_overhead_refund(budget: VerificationBudget) -> Decimal:
    """Compute the refundable overhead after bounty expiry.

    Returns the unused portion of the overhead deposit.
    """
    return budget.overhead_deposit - budget.overhead_used
