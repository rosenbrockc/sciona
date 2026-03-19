"""Tests for the blind data splitting pipeline."""

from __future__ import annotations

import pytest

from ageom.clearinghouse.data_splitter import (
    assign_partition,
    compute_split_hash,
    split_dataset,
    validate_dataset,
)
from ageom.clearinghouse.models import SplitAssignment


class TestPartitionAssignment:
    def test_deterministic(self):
        """Same key + bounty always produces the same partition."""
        p1 = assign_partition("user_42", "bounty_abc")
        p2 = assign_partition("user_42", "bounty_abc")
        assert p1 == p2

    def test_different_bounty_different_split(self):
        """Different bounty IDs produce different assignments (with high probability)."""
        results_a = [assign_partition(f"u{i}", "bounty_A") for i in range(100)]
        results_b = [assign_partition(f"u{i}", "bounty_B") for i in range(100)]
        assert results_a != results_b

    def test_approximately_20_80(self):
        """Over many keys, ~20% should be public."""
        n = 1000
        partitions = [assign_partition(f"key_{i}", "test_bounty") for i in range(n)]
        public_pct = sum(1 for p in partitions if p == "public") / n
        assert 0.15 <= public_pct <= 0.25

    def test_valid_partition_values(self):
        p = assign_partition("any_key", "any_bounty")
        assert p in ("public", "blind")


class TestSplitHash:
    def test_deterministic(self):
        assignments = [
            SplitAssignment(unit_key="a", partition="public"),
            SplitAssignment(unit_key="b", partition="blind"),
        ]
        h1 = compute_split_hash(assignments, "salt1")
        h2 = compute_split_hash(assignments, "salt1")
        assert h1 == h2

    def test_different_salt_different_hash(self):
        assignments = [SplitAssignment(unit_key="a", partition="public")]
        h1 = compute_split_hash(assignments, "salt1")
        h2 = compute_split_hash(assignments, "salt2")
        assert h1 != h2

    def test_order_independent(self):
        a1 = [
            SplitAssignment(unit_key="b", partition="blind"),
            SplitAssignment(unit_key="a", partition="public"),
        ]
        a2 = [
            SplitAssignment(unit_key="a", partition="public"),
            SplitAssignment(unit_key="b", partition="blind"),
        ]
        assert compute_split_hash(a1, "s") == compute_split_hash(a2, "s")


class TestSplitDataset:
    def test_basic_split(self):
        keys = [f"user_{i}" for i in range(200)]
        result = split_dataset(keys, "bounty_1")
        assert result.public_count + result.blind_count == 200
        assert result.split_hash != ""
        assert len(result.assignments) == 200

    def test_too_few_samples(self):
        keys = [f"u{i}" for i in range(10)]
        with pytest.raises(ValueError, match="minimum is 50"):
            split_dataset(keys, "bounty_1")

    def test_custom_minimum(self):
        keys = [f"u{i}" for i in range(20)]
        result = split_dataset(keys, "bounty_1", min_samples=10, min_public_samples=1)
        assert result.public_count + result.blind_count == 20

    def test_deterministic_split(self):
        keys = [f"k{i}" for i in range(100)]
        r1 = split_dataset(keys, "bounty_X", min_samples=50)
        r2 = split_dataset(keys, "bounty_X", min_samples=50)
        assert r1.split_hash == r2.split_hash
        assert r1.public_count == r2.public_count

    def test_different_bounty_different_result(self):
        keys = [f"k{i}" for i in range(100)]
        r1 = split_dataset(keys, "bounty_A", min_samples=50)
        r2 = split_dataset(keys, "bounty_B", min_samples=50)
        assert r1.split_hash != r2.split_hash


class TestValidateDataset:
    def test_valid_dataset(self):
        reasons = validate_dataset({"total_samples": 200})
        assert reasons == []

    def test_too_few_samples(self):
        reasons = validate_dataset({"total_samples": 10})
        assert any("Too few" in r for r in reasons)

    def test_constant_columns(self):
        reasons = validate_dataset(
            {"total_samples": 200, "constant_columns": ["col_a"]}
        )
        assert any("constant" in r.lower() for r in reasons)
