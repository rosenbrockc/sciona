"""Tests for Stage 3: Invariant Variance."""

from __future__ import annotations

import numpy as np
import pytest

from sciona.symbolic_funnel.contracts import FunnelCandidate, FunnelConfig
from sciona.symbolic_funnel.index import FunnelAtomEntry
from sciona.symbolic_funnel.stages import stage_invariant_variance

from .conftest import (
    K_DISPERSION,
    R_GAS,
    make_dispersion_dataset,
    make_ideal_gas_dataset,
    make_random_dataset,
)


class TestInvariantVariance:
    def test_clean_data_passes_with_high_confidence(
        self, dispersion_entry: FunnelAtomEntry, default_config: FunnelConfig
    ) -> None:
        dataset = make_dispersion_dataset(noise_frac=0.0)
        candidates = [FunnelCandidate(entry=dispersion_entry)]
        result = stage_invariant_variance(dataset, candidates, default_config)
        assert len(result) == 1
        verdict = result[0].verdicts[-1]
        assert verdict.passed is True
        assert verdict.score is not None and verdict.score > 0.9
        # Fitted constant should be close to K_DISPERSION.
        fitted_K = result[0].fitted_constants.get("K")
        assert fitted_K is not None
        np.testing.assert_allclose(fitted_K, K_DISPERSION, rtol=1e-6)

    @pytest.mark.parametrize("noise_frac", [0.01, 0.05])
    def test_noisy_data_still_passes(
        self, dispersion_entry: FunnelAtomEntry, noise_frac: float
    ) -> None:
        config = FunnelConfig(cv_threshold=0.10)
        dataset = make_dispersion_dataset(noise_frac=noise_frac)
        candidates = [FunnelCandidate(entry=dispersion_entry)]
        result = stage_invariant_variance(dataset, candidates, config)
        assert len(result) == 1
        assert result[0].verdicts[-1].passed is True

    def test_cv_scales_with_noise(
        self, dispersion_entry: FunnelAtomEntry
    ) -> None:
        config = FunnelConfig(cv_threshold=1.0)  # Very permissive
        cvs = []
        for noise_frac in [0.0, 0.01, 0.05, 0.10]:
            dataset = make_dispersion_dataset(noise_frac=noise_frac)
            candidates = [FunnelCandidate(entry=dispersion_entry)]
            result = stage_invariant_variance(dataset, candidates, config)
            cv = result[0].verdicts[-1].evidence.get("cv", 0.0)
            cvs.append(cv)
        # CV should be monotonically non-decreasing with noise.
        for i in range(len(cvs) - 1):
            assert cvs[i] <= cvs[i + 1] + 1e-10

    def test_ideal_gas_clean(
        self, ideal_gas_entry: FunnelAtomEntry, default_config: FunnelConfig
    ) -> None:
        dataset = make_ideal_gas_dataset(noise_frac=0.0)
        candidates = [FunnelCandidate(entry=ideal_gas_entry)]
        result = stage_invariant_variance(dataset, candidates, default_config)
        assert len(result) == 1
        verdict = result[0].verdicts[-1]
        assert verdict.passed is True
        fitted_R = result[0].fitted_constants.get("R")
        assert fitted_R is not None
        np.testing.assert_allclose(fitted_R, R_GAS, rtol=1e-6)

    def test_wrong_law_rejects(
        self, ideal_gas_entry: FunnelAtomEntry, default_config: FunnelConfig
    ) -> None:
        """Applying ideal gas invariant to dispersion delay data should fail."""
        # Create a dataset with columns named P, V, n, T but generated randomly.
        from sciona.symbolic_funnel.dataset import ColumnMetadata, EmpiricalDataset

        rng = np.random.default_rng(42)
        dataset = EmpiricalDataset(
            data=rng.uniform(0.1, 100, (1000, 4)),
            columns=[
                ColumnMetadata(name="P"),
                ColumnMetadata(name="V"),
                ColumnMetadata(name="n"),
                ColumnMetadata(name="T"),
            ],
        )
        candidates = [FunnelCandidate(entry=ideal_gas_entry)]
        result = stage_invariant_variance(dataset, candidates, default_config)
        # Should either be rejected (length 0) or have a failing verdict.
        if result:
            verdict = result[0].verdicts[-1]
            assert verdict.passed is False or verdict.score is not None and verdict.score < 0.5

    def test_no_invariant_forms_passes_through(
        self, ohm_entry: FunnelAtomEntry, default_config: FunnelConfig
    ) -> None:
        """Entries without invariant forms should pass through to RANSAC."""
        from .conftest import make_ohm_dataset

        dataset = make_ohm_dataset()
        candidates = [FunnelCandidate(entry=ohm_entry)]
        result = stage_invariant_variance(dataset, candidates, default_config)
        # Should pass through (no verdict added by this stage).
        assert len(result) == 1
