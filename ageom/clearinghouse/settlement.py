"""Bounty settlement engine — payout computation with Shapley integration."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import Sequence

from ageom.clearinghouse.models import (
    ObjectiveSpec,
    PayoutPlan,
    PayoutRecipient,
    WinningCDG,
)
from ageom.clearinghouse.pareto import compute_pareto_front, split_architect_payout
from ageom.provenance.shapley import compute_shapley_values


# Payout ratios — must sum to 1.
PLATFORM_SHARE = Fraction(5, 100)
ARCHITECT_SHARE = Fraction(65, 100)
ORIGINATOR_SHARE = Fraction(30, 100)

assert PLATFORM_SHARE + ARCHITECT_SHARE + ORIGINATOR_SHARE == Fraction(1)


def compute_settlement(
    escrow_amount: Decimal,
    winning_cdgs: Sequence[WinningCDG],
    atom_dags: dict[str, dict[str, set[str]]],
    *,
    platform_account_id: str = "platform",
    originator_accounts: dict[str, str] | None = None,
) -> PayoutPlan:
    """Compute the full settlement payout plan.

    Parameters
    ----------
    escrow_amount
        Total bounty escrow.
    winning_cdgs
        Verified winning CDG submissions.
    atom_dags
        Mapping of CDG hash to its atom dependency DAG
        (``{node: {deps}}``) for Shapley computation.
    platform_account_id
        Stripe account ID for the platform.
    originator_accounts
        Mapping of atom FQDN to Stripe account ID for originators.

    Returns
    -------
    PayoutPlan
        Complete payout breakdown with recipients and Shapley allocations.
    """
    if not winning_cdgs:
        return PayoutPlan(
            bounty_id="",
            escrow_amount=escrow_amount,
        )

    originator_accounts = originator_accounts or {}
    escrow_frac = Fraction(escrow_amount)

    # Platform share
    platform_amount = Decimal(str(float(escrow_frac * PLATFORM_SHARE)))
    recipients: list[PayoutRecipient] = [
        PayoutRecipient(
            recipient_id=platform_account_id,
            role="platform",
            amount=platform_amount,
            stripe_account_id=platform_account_id,
        )
    ]

    # Architect share — split among winners by weight
    architect_total_frac = escrow_frac * ARCHITECT_SHARE

    if len(winning_cdgs) == 1:
        winner = winning_cdgs[0]
        architect_amount = Decimal(str(float(architect_total_frac)))
        recipients.append(
            PayoutRecipient(
                recipient_id=winner.architect_id,
                role="architect",
                amount=architect_amount,
                cdg_hash=winner.cdg_hash,
            )
        )
    else:
        # Multi-winner: distribute by weight
        total_weight = sum(w.weight for w in winning_cdgs)
        for winner in winning_cdgs:
            share = Fraction(int(winner.weight * 1000), int(total_weight * 1000))
            amount = Decimal(str(float(architect_total_frac * share)))
            recipients.append(
                PayoutRecipient(
                    recipient_id=winner.architect_id,
                    role="architect",
                    amount=amount,
                    cdg_hash=winner.cdg_hash,
                )
            )

    # Originator share — computed via Shapley values per CDG
    originator_total_frac = escrow_frac * ORIGINATOR_SHARE
    shapley_allocations: dict[str, float] = {}

    for winner in winning_cdgs:
        dag = atom_dags.get(winner.cdg_hash, {})
        if not dag:
            continue

        shapley = compute_shapley_values(dag)

        # Weight by this winner's share of the architect pool
        if len(winning_cdgs) == 1:
            winner_weight = Fraction(1)
        else:
            total_w = sum(w.weight for w in winning_cdgs)
            winner_weight = Fraction(int(winner.weight * 1000), int(total_w * 1000))

        for atom_fqdn, atom_share in shapley.items():
            amount_frac = originator_total_frac * atom_share * winner_weight
            amount = Decimal(str(float(amount_frac)))

            shapley_allocations[atom_fqdn] = (
                shapley_allocations.get(atom_fqdn, 0.0) + float(atom_share)
            )

            recipients.append(
                PayoutRecipient(
                    recipient_id=originator_accounts.get(atom_fqdn, atom_fqdn),
                    role="originator",
                    amount=amount,
                    atom_fqdn=atom_fqdn,
                    cdg_hash=winner.cdg_hash,
                    stripe_account_id=originator_accounts.get(atom_fqdn, ""),
                )
            )

    # Residual assignment to platform to maintain conservation
    distributed = sum(r.amount for r in recipients)
    residual = escrow_amount - distributed
    if residual != Decimal("0"):
        recipients[0] = PayoutRecipient(
            recipient_id=recipients[0].recipient_id,
            role="platform",
            amount=recipients[0].amount + residual,
            stripe_account_id=recipients[0].stripe_account_id,
        )

    bounty_id = winning_cdgs[0].submission_id.split("-")[0] if winning_cdgs else ""

    return PayoutPlan(
        bounty_id=bounty_id,
        escrow_amount=escrow_amount,
        recipients=recipients,
        shapley_allocations=shapley_allocations,
        winners=list(winning_cdgs),
    )


def compute_expiry_refund(
    escrow_amount: Decimal,
    overhead_used: Decimal,
) -> Decimal:
    """Compute refund amount for an expired bounty.

    Refund = escrow - actual compute costs (overhead_used).
    """
    return max(Decimal("0"), escrow_amount - overhead_used)


def verify_payout_conservation(plan: PayoutPlan) -> bool:
    """Verify that the sum of all payouts equals the escrow amount.

    This is a critical invariant — violation indicates a bug.
    """
    total = sum(r.amount for r in plan.recipients)
    return total == plan.escrow_amount
