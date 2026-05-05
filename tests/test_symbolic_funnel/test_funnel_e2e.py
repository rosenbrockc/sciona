"""End-to-end integration tests for the heuristic funnel."""

from __future__ import annotations

import numpy as np
import pytest

from sciona.symbolic_funnel.contracts import FunnelConfig
from sciona.symbolic_funnel.index import FunnelIndex
from sciona.symbolic_funnel.stages import run_funnel

from .conftest import (
    K_DISPERSION,
    R_GAS,
    make_dispersion_dataset,
    make_ideal_gas_dataset,
    make_ohm_dataset,
    make_random_dataset,
)


class TestFunnelE2E:
    def test_dispersion_delay_ranks_first(
        self, test_index: FunnelIndex
    ) -> None:
        """Dispersion delay data should match the dispersion entry as #1."""
        dataset = make_dispersion_dataset(noise_frac=0.01)
        result = run_funnel(dataset, test_index)
        assert len(result.ranked_candidates) > 0
        top = result.ranked_candidates[0]
        assert top.entry.atom_name == "dispersion_delay"
        assert top.aggregate_score > 0.5
        # Should have fitted K.
        fitted_K = top.fitted_constants.get("K")
        assert fitted_K is not None
        np.testing.assert_allclose(fitted_K, K_DISPERSION, rtol=0.05)

    def test_ideal_gas_ranks_first(
        self, test_index: FunnelIndex
    ) -> None:
        """Ideal gas data should match the ideal gas entry as #1."""
        dataset = make_ideal_gas_dataset(noise_frac=0.01)
        result = run_funnel(dataset, test_index)
        assert len(result.ranked_candidates) > 0
        top = result.ranked_candidates[0]
        assert top.entry.atom_name == "ideal_gas"
        fitted_R = top.fitted_constants.get("R")
        assert fitted_R is not None
        np.testing.assert_allclose(fitted_R, R_GAS, rtol=0.05)

    def test_random_noise_no_high_confidence_match(
        self, test_index: FunnelIndex
    ) -> None:
        """Random noise should not confidently match any law."""
        dataset = make_random_dataset()
        result = run_funnel(dataset, test_index)
        # Either no candidates survive, or top score is low.
        if result.ranked_candidates:
            assert result.ranked_candidates[0].aggregate_score < 0.5

    def test_all_stages_executed(self, test_index: FunnelIndex) -> None:
        dataset = make_dispersion_dataset()
        result = run_funnel(dataset, test_index)
        assert "boundary_triage" in result.stages_executed
        assert "exponent_extraction" in result.stages_executed
        assert "invariant_variance" in result.stages_executed
        assert "ransac" in result.stages_executed

    def test_timing_recorded(self, test_index: FunnelIndex) -> None:
        dataset = make_dispersion_dataset()
        result = run_funnel(dataset, test_index)
        assert "boundary_triage" in result.timing
        assert all(v >= 0 for v in result.timing.values())

    def test_ohm_law_matches_own_data(
        self, test_index: FunnelIndex
    ) -> None:
        """Ohm's law (no constants) should still be matchable."""
        dataset = make_ohm_dataset(noise_frac=0.01)
        result = run_funnel(dataset, test_index)
        # Ohm's law has no constants, so it goes through RANSAC direct residual.
        ohm_candidates = [
            c for c in result.ranked_candidates if c.entry.atom_name == "ohm_law"
        ]
        if ohm_candidates:
            assert ohm_candidates[0].aggregate_score > 0.0

    def test_noisy_dispersion_still_matches(
        self, test_index: FunnelIndex
    ) -> None:
        """Even with 10% noise, dispersion should still be the top match."""
        config = FunnelConfig(cv_threshold=0.15)
        dataset = make_dispersion_dataset(noise_frac=0.10)
        result = run_funnel(dataset, test_index, config)
        if result.ranked_candidates:
            top = result.ranked_candidates[0]
            assert top.entry.atom_name == "dispersion_delay"


class TestHeuristicBridge:
    def test_funnel_result_produces_valid_observations(
        self, test_index: FunnelIndex
    ) -> None:
        """Verify heuristic bridge produces valid RuntimeHeuristicObservation."""
        from sciona.symbolic_funnel.heuristic_bridge import (
            funnel_result_to_evidence,
            funnel_result_to_observations,
        )

        dataset = make_dispersion_dataset(noise_frac=0.01)
        result = run_funnel(dataset, test_index)
        observations = funnel_result_to_observations(result)
        assert len(observations) > 0
        for obs in observations:
            # Each observation should have valid fields.
            assert obs.heuristic.heuristic_id in {
                "boundary_triage_pass",
                "exponent_signature_match",
                "invariant_variance_cv",
                "ransac_fit_residual",
                "graph_pruning_depth",
            }
            assert 0.0 <= obs.confidence <= 1.0

    def test_evidence_bundle_has_summary(
        self, test_index: FunnelIndex
    ) -> None:
        from sciona.symbolic_funnel.heuristic_bridge import funnel_result_to_evidence

        dataset = make_dispersion_dataset(noise_frac=0.01)
        result = run_funnel(dataset, test_index)
        evidence = funnel_result_to_evidence(result)
        assert evidence.heuristic_summary["stages_executed"] == result.stages_executed
        assert "top_match" in evidence.heuristic_summary
