"""Tests for the verification engine."""

from __future__ import annotations

from decimal import Decimal

import pytest

from sciona.clearinghouse.models import BestScore, VerificationBudget
from sciona.clearinghouse.verification import (
    compute_overhead_refund,
    consume_verification_slot,
    remaining_slots,
    should_trigger_verification,
    update_best_score,
)


class TestImprovementThreshold:
    def test_minimize_improvement(self):
        assert should_trigger_verification(0.90, 1.00, 5.0, "minimize")

    def test_minimize_no_improvement(self):
        assert not should_trigger_verification(0.99, 1.00, 5.0, "minimize")

    def test_maximize_improvement(self):
        assert should_trigger_verification(1.10, 1.00, 5.0, "maximize")

    def test_maximize_no_improvement(self):
        assert not should_trigger_verification(1.01, 1.00, 5.0, "maximize")

    def test_exact_threshold(self):
        # 5% improvement exactly
        assert should_trigger_verification(0.95, 1.00, 5.0, "minimize")

    def test_zero_best_to_date(self):
        assert should_trigger_verification(0.5, 0.0, 5.0, "minimize")
        assert not should_trigger_verification(0.0, 0.0, 5.0, "minimize")

    def test_large_improvement(self):
        assert should_trigger_verification(0.01, 1.00, 5.0, "minimize")

    def test_custom_threshold(self):
        assert should_trigger_verification(0.89, 1.00, 10.0, "minimize")
        assert not should_trigger_verification(0.92, 1.00, 10.0, "minimize")


class TestVerificationBudget:
    def test_consume_slot(self):
        budget = VerificationBudget(bounty_id="b1", total_slots=5, used_slots=0)
        updated = consume_verification_slot(budget)
        assert updated.used_slots == 1
        assert updated.total_slots == 5

    def test_exhaust_budget(self):
        budget = VerificationBudget(bounty_id="b1", total_slots=5, used_slots=5)
        with pytest.raises(ValueError, match="exhausted"):
            consume_verification_slot(budget)

    def test_remaining_slots(self):
        budget = VerificationBudget(bounty_id="b1", total_slots=5, used_slots=3)
        assert remaining_slots(budget) == 2

    def test_remaining_slots_zero(self):
        budget = VerificationBudget(bounty_id="b1", total_slots=5, used_slots=5)
        assert remaining_slots(budget) == 0

    def test_remaining_slots_overflow(self):
        budget = VerificationBudget(bounty_id="b1", total_slots=5, used_slots=6)
        assert remaining_slots(budget) == 0


class TestBestScore:
    def test_first_score(self):
        result = update_best_score(None, 0.5, "sub1", "minimize")
        assert result is not None
        assert result.best_value == 0.5

    def test_better_minimize(self):
        current = BestScore(bounty_id="b1", metric_name="loss", best_value=0.5, best_submission_id="s1")
        result = update_best_score(current, 0.3, "s2", "minimize")
        assert result is not None
        assert result.best_value == 0.3
        assert result.best_submission_id == "s2"

    def test_worse_minimize(self):
        current = BestScore(bounty_id="b1", metric_name="loss", best_value=0.5, best_submission_id="s1")
        result = update_best_score(current, 0.7, "s2", "minimize")
        assert result is None

    def test_better_maximize(self):
        current = BestScore(bounty_id="b1", metric_name="acc", best_value=0.8, best_submission_id="s1")
        result = update_best_score(current, 0.9, "s2", "maximize")
        assert result is not None
        assert result.best_value == 0.9

    def test_worse_maximize(self):
        current = BestScore(bounty_id="b1", metric_name="acc", best_value=0.8, best_submission_id="s1")
        result = update_best_score(current, 0.7, "s2", "maximize")
        assert result is None


class TestOverheadRefund:
    def test_full_refund(self):
        budget = VerificationBudget(
            bounty_id="b1",
            overhead_deposit=Decimal("100"),
            overhead_used=Decimal("0"),
        )
        assert compute_overhead_refund(budget) == Decimal("100")

    def test_partial_refund(self):
        budget = VerificationBudget(
            bounty_id="b1",
            overhead_deposit=Decimal("100"),
            overhead_used=Decimal("30"),
        )
        assert compute_overhead_refund(budget) == Decimal("70")

    def test_no_refund(self):
        budget = VerificationBudget(
            bounty_id="b1",
            overhead_deposit=Decimal("100"),
            overhead_used=Decimal("100"),
        )
        assert compute_overhead_refund(budget) == Decimal("0")
