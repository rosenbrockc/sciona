"""Tests for dashboard helpers — impact factor, BibTeX, compute preserved."""

from __future__ import annotations

from sciona.ecosystem.dashboard import (
    compute_h_index,
    compute_impact_factor,
    estimate_compute_preserved,
    generate_bibtex,
)


class TestHIndex:
    def test_basic(self):
        # 3 bounties worth [5, 3, 1] → h=2 (2 bounties each ≥ 2)
        assert compute_h_index([5, 3, 1]) == 2

    def test_all_high(self):
        assert compute_h_index([10, 10, 10]) == 3

    def test_all_low(self):
        assert compute_h_index([0.5, 0.5, 0.5]) == 0

    def test_single(self):
        assert compute_h_index([100]) == 1

    def test_empty(self):
        assert compute_h_index([]) == 0

    def test_exact_match(self):
        # [3, 3, 3] → h=3
        assert compute_h_index([3, 3, 3]) == 3

    def test_descending_order(self):
        assert compute_h_index([10, 5, 3, 2, 1]) == 3


class TestImpactFactor:
    def test_basic(self):
        impact = compute_impact_factor(
            [10, 5, 3],
            atom_count=5,
            originator_id="o1",
            github_username="alice",
        )
        assert impact.bounty_count == 3
        assert impact.total_bounty_value == 18.0
        assert impact.atom_count == 5
        assert impact.h_index == 3

    def test_empty(self):
        impact = compute_impact_factor([])
        assert impact.bounty_count == 0
        assert impact.h_index == 0


class TestComputePreserved:
    def test_basic(self):
        bounties = [
            {"escrow_amount": 100.0, "cdg_node_count": 10},
            {"escrow_amount": 200.0, "cdg_node_count": 20},
        ]
        result = estimate_compute_preserved(bounties)
        assert result.total_bounties_settled == 2
        assert result.total_escrow_value == 300.0
        assert result.estimated_tokens_saved > 0
        assert result.estimated_cost_saved_usd > 0

    def test_empty(self):
        result = estimate_compute_preserved([])
        assert result.total_bounties_settled == 0
        assert result.estimated_tokens_saved == 0

    def test_custom_params(self):
        bounties = [{"escrow_amount": 50.0, "cdg_node_count": 5}]
        result = estimate_compute_preserved(
            bounties, tokens_per_step=1000, avg_attempts=3
        )
        assert result.estimated_tokens_saved == 5 * 1000 * 3


class TestBibTeX:
    def test_basic(self):
        bib = generate_bibtex(
            "pkg.mod.filter",
            ["alice", "bob"],
            year=2025,
            description="A signal filter",
        )
        assert "@misc{pkg_mod_filter" in bib
        assert "alice and bob" in bib
        assert "2025" in bib
        assert "Algorithmic Commons Registry" in bib
        assert "fqdn:pkg.mod.filter" in bib

    def test_no_authors(self):
        bib = generate_bibtex("pkg.sort", [])
        assert "Unknown" in bib

    def test_single_author(self):
        bib = generate_bibtex("pkg.sort", ["alice"])
        assert "alice" in bib
        assert " and " not in bib

    def test_description_as_title(self):
        bib = generate_bibtex("pkg.fft", ["alice"], description="Fast Fourier Transform")
        assert "Fast Fourier Transform" in bib
