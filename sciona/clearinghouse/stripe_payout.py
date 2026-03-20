"""Stripe Connect payout execution."""

from __future__ import annotations

import logging
from decimal import Decimal

from sciona.clearinghouse.models import PayoutPlan, PayoutResult

logger = logging.getLogger(__name__)


class StripePayout:
    """Execute payouts via Stripe Connect transfers."""

    def __init__(self, stripe_secret_key: str) -> None:
        self._stripe_key = stripe_secret_key

    async def execute_payout(self, plan: PayoutPlan) -> PayoutResult:
        """Execute all transfers in the payout plan.

        Each recipient gets a Stripe Transfer to their Connected Account.
        All transfers belong to the same transfer_group for reconciliation.
        """
        try:
            import stripe  # type: ignore[import-untyped]
        except ImportError:
            return PayoutResult(
                bounty_id=plan.bounty_id,
                success=False,
                error="stripe not installed",
            )

        stripe.api_key = self._stripe_key
        transfer_ids: list[str] = []

        try:
            for recipient in plan.recipients:
                if not recipient.stripe_account_id:
                    logger.warning(
                        "Skipping payout for %s — no Stripe account",
                        recipient.recipient_id,
                    )
                    continue

                amount_cents = int(recipient.amount * Decimal("100"))
                if amount_cents <= 0:
                    continue

                transfer = stripe.Transfer.create(
                    amount=amount_cents,
                    currency="usd",
                    destination=recipient.stripe_account_id,
                    transfer_group=f"bounty_{plan.bounty_id}",
                    metadata={
                        "bounty_id": plan.bounty_id,
                        "role": recipient.role,
                        "cdg_hash": recipient.cdg_hash,
                        "atom_fqdn": recipient.atom_fqdn,
                    },
                )
                transfer_ids.append(transfer.id)

            return PayoutResult(
                bounty_id=plan.bounty_id,
                transfer_ids=transfer_ids,
                success=True,
            )
        except Exception as exc:
            logger.exception("Stripe payout failed for bounty %s", plan.bounty_id)
            return PayoutResult(
                bounty_id=plan.bounty_id,
                transfer_ids=transfer_ids,
                success=False,
                error=str(exc),
            )
