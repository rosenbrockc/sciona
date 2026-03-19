"""Tests for the bounty settlement engine."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

from ageom.clearinghouse.models import PayoutPlan, WinningCDG
from ageom.clearinghouse.settlement import (
    ARCHITECT_SHARE,
    ORIGINATOR_SHARE,
    PLATFORM_SHARE,
    compute_expiry_refund,
    compute_settlement,
    verify_payout_conservation,
)


class TestPayoutRatios:
    def test_shares_sum_to_one(self):
        assert PLATFORM_SHARE + ARCHITECT_SHARE + ORIGINATOR_SHARE == Fraction(1)

    def test_platform_5_percent(self):
        assert PLATFORM_SHARE == Fraction(5, 100)

    def test_architect_65_percent(self):
        assert ARCHITECT_SHARE == Fraction(65, 100)

    def test_originator_30_percent(self):
        assert ORIGINATOR_SHARE == Fraction(30, 100)


class TestSingleWinnerSettlement:
    def test_basic_settlement(self):
        winners = [
            WinningCDG(
                submission_id="sub1",
                architect_id="arch1",
                cdg_hash="cdg1",
                atom_versions={"pkg.filter": "hash1", "pkg.sort": "hash2"},
                metric_values={"loss": 0.1},
            )
        ]
        dag = {"cdg1": {"pkg.filter": {"pkg.sort"}, "pkg.sort": set()}}

        plan = compute_settlement(
            Decimal("100.00"),
            winners,
            dag,
        )

        assert len(plan.recipients) > 0
        assert plan.escrow_amount == Decimal("100.00")

    def test_conservation_invariant(self):
        winners = [
            WinningCDG(
                submission_id="sub1",
                architect_id="arch1",
                cdg_hash="cdg1",
                atom_versions={"a": "h1"},
            )
        ]
        dag = {"cdg1": {"a": set()}}

        plan = compute_settlement(Decimal("100.00"), winners, dag)
        assert verify_payout_conservation(plan)

    def test_conservation_odd_amount(self):
        winners = [
            WinningCDG(
                submission_id="sub1",
                architect_id="arch1",
                cdg_hash="cdg1",
                atom_versions={"a": "h1", "b": "h2"},
            )
        ]
        dag = {"cdg1": {"a": {"b"}, "b": set()}}

        plan = compute_settlement(Decimal("33.33"), winners, dag)
        assert verify_payout_conservation(plan)

    def test_has_all_roles(self):
        winners = [
            WinningCDG(
                submission_id="sub1",
                architect_id="arch1",
                cdg_hash="cdg1",
                atom_versions={"a": "h1"},
            )
        ]
        dag = {"cdg1": {"a": set()}}

        plan = compute_settlement(Decimal("1000.00"), winners, dag)
        roles = {r.role for r in plan.recipients}
        assert "platform" in roles
        assert "architect" in roles
        assert "originator" in roles

    def test_shapley_allocations(self):
        winners = [
            WinningCDG(
                submission_id="sub1",
                architect_id="arch1",
                cdg_hash="cdg1",
                atom_versions={"a": "h1", "b": "h2"},
            )
        ]
        dag = {"cdg1": {"a": {"b"}, "b": set()}}

        plan = compute_settlement(Decimal("100.00"), winners, dag)
        assert "a" in plan.shapley_allocations
        assert "b" in plan.shapley_allocations


class TestMultiWinnerSettlement:
    def test_two_winners(self):
        winners = [
            WinningCDG(
                submission_id="sub1",
                architect_id="arch1",
                cdg_hash="cdg1",
                atom_versions={"a": "h1"},
                weight=0.7,
            ),
            WinningCDG(
                submission_id="sub2",
                architect_id="arch2",
                cdg_hash="cdg2",
                atom_versions={"b": "h2"},
                weight=0.3,
            ),
        ]
        dag = {
            "cdg1": {"a": set()},
            "cdg2": {"b": set()},
        }

        plan = compute_settlement(Decimal("100.00"), winners, dag)
        assert verify_payout_conservation(plan)
        assert len(plan.winners) == 2

    def test_multi_winner_architect_split(self):
        winners = [
            WinningCDG(
                submission_id="sub1", architect_id="a1", cdg_hash="c1",
                atom_versions={"x": "h"}, weight=0.6,
            ),
            WinningCDG(
                submission_id="sub2", architect_id="a2", cdg_hash="c2",
                atom_versions={"y": "h"}, weight=0.4,
            ),
        ]
        dag = {"c1": {"x": set()}, "c2": {"y": set()}}

        plan = compute_settlement(Decimal("1000.00"), winners, dag)
        architect_payouts = [r for r in plan.recipients if r.role == "architect"]
        assert len(architect_payouts) == 2


class TestEmptySettlement:
    def test_no_winners(self):
        plan = compute_settlement(Decimal("100.00"), [], {})
        assert plan.recipients == []
        assert plan.escrow_amount == Decimal("100.00")


class TestExpiryRefund:
    def test_full_refund(self):
        assert compute_expiry_refund(Decimal("100"), Decimal("0")) == Decimal("100")

    def test_partial_refund(self):
        assert compute_expiry_refund(Decimal("100"), Decimal("30")) == Decimal("70")

    def test_all_used(self):
        assert compute_expiry_refund(Decimal("100"), Decimal("100")) == Decimal("0")

    def test_overspend_clamped(self):
        assert compute_expiry_refund(Decimal("100"), Decimal("150")) == Decimal("0")


class TestPayoutConservation:
    @pytest.mark.parametrize("amount", ["100.00", "33.33", "1000.00", "0.01", "999.99"])
    def test_conservation_parametrized(self, amount):
        winners = [
            WinningCDG(
                submission_id="sub1", architect_id="arch1", cdg_hash="cdg1",
                atom_versions={"a": "h1", "b": "h2", "c": "h3"},
            )
        ]
        dag = {"cdg1": {"a": {"b"}, "b": {"c"}, "c": set()}}

        plan = compute_settlement(Decimal(amount), winners, dag)
        assert verify_payout_conservation(plan), (
            f"Conservation violated for {amount}: "
            f"sum={sum(r.amount for r in plan.recipients)}"
        )
