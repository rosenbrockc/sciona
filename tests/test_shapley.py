"""Tests for the Shapley value engine."""

from __future__ import annotations

from fractions import Fraction

import pytest

from sciona.provenance.shapley import compute_shapley_values


class TestSingleNode:
    def test_single_node(self):
        dag = {"A": set()}
        result = compute_shapley_values(dag)
        assert result == {"A": Fraction(1)}


class TestLinearChain:
    def test_linear_chain(self):
        # A -> B -> C
        dag = {"A": {"B"}, "B": {"C"}, "C": set()}
        result = compute_shapley_values(dag)
        assert result["A"] == Fraction(1, 3)
        assert result["B"] == Fraction(1, 3)
        assert result["C"] == Fraction(1, 3)


class TestDiamond:
    def test_diamond(self):
        # A -> {B, C} -> D
        dag = {"A": {"B", "C"}, "B": {"D"}, "C": {"D"}, "D": set()}
        result = compute_shapley_values(dag)
        assert result["A"] == Fraction(1, 4)
        assert result["B"] == Fraction(1, 4)
        assert result["C"] == Fraction(1, 4)
        assert result["D"] == Fraction(1, 4)


class TestWideTree:
    def test_wide_tree(self):
        # root + 10 leaves
        leaves = {f"L{i}" for i in range(10)}
        dag: dict[str, set[str]] = {"root": leaves}
        for leaf in leaves:
            dag[leaf] = set()
        result = compute_shapley_values(dag)
        expected = Fraction(1, 11)
        for node, val in result.items():
            assert val == expected


class TestEdgeCases:
    def test_empty_graph_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_shapley_values({})

    def test_cycle_raises(self):
        dag = {"A": {"B"}, "B": {"A"}}
        with pytest.raises(ValueError, match="[Cc]ycle"):
            compute_shapley_values(dag)

    def test_self_loop_raises(self):
        dag = {"A": {"A"}}
        with pytest.raises(ValueError, match="[Cc]ycle"):
            compute_shapley_values(dag)


class TestSumEqualsOne:
    @pytest.mark.parametrize(
        "dag",
        [
            {"A": set()},
            {"A": {"B"}, "B": set()},
            {"A": {"B"}, "B": {"C"}, "C": set()},
            {"A": {"B", "C"}, "B": {"D"}, "C": {"D"}, "D": set()},
            {"R": {f"L{i}" for i in range(5)}, **{f"L{i}": set() for i in range(5)}},
        ],
        ids=["single", "pair", "chain3", "diamond", "wide5"],
    )
    def test_sum_equals_one(self, dag: dict[str, set[str]]):
        result = compute_shapley_values(dag)
        assert sum(result.values()) == Fraction(1)


class TestReturnsFractions:
    def test_returns_fractions(self):
        dag = {"A": {"B"}, "B": set()}
        result = compute_shapley_values(dag)
        for val in result.values():
            assert isinstance(val, Fraction)
