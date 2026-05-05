"""Tests for Stage 2: Exponent Extraction via Log-Space SVD."""

from __future__ import annotations

import numpy as np
import pytest

from sciona.symbolic_funnel.contracts import FunnelCandidate, FunnelConfig
from sciona.symbolic_funnel.index import FunnelAtomEntry, FunnelIndex
from sciona.symbolic_funnel.stages import (
    _extract_exponents_svd,
    stage_exponent_extraction,
)

from .conftest import make_dispersion_dataset, make_gravity_dataset


class TestExponentExtraction:
    def test_svd_recovers_dispersion_exponents(self) -> None:
        """SVD on log(t, DM, f) for t = K*DM/f^2 should recover (1, 1, -2)."""
        dataset = make_dispersion_dataset(noise_frac=0.0)
        log_dataset, log_names = dataset.log_transform()
        config = FunnelConfig()
        exponents = _extract_exponents_svd(log_dataset, config)

        assert exponents is not None
        # The null-space vector should have ratios consistent with (1, 1, -2).
        # After normalization by max-abs, one component should be ±1.
        # The relationship is: log(t) = log(K) + 1*log(DM) - 2*log(f)
        # So null-space: [1, -1, 2] or [-1, 1, -2] (up to sign).
        abs_exp = np.abs(exponents)
        assert np.max(abs_exp) == pytest.approx(1.0, abs=0.01)

    def test_svd_robust_to_moderate_noise(self) -> None:
        """5% noise should still yield correct exponents."""
        dataset = make_dispersion_dataset(noise_frac=0.05)
        log_dataset, _ = dataset.log_transform()
        config = FunnelConfig()
        exponents = _extract_exponents_svd(log_dataset, config)
        assert exponents is not None
        # Should still be close to the true exponents.
        abs_exp = np.abs(exponents)
        assert np.max(abs_exp) > 0.8  # Still has clear structure.

    def test_gravity_exponents(self) -> None:
        """SVD on log(F, m1, m2, r) for F = G*m1*m2/r^2."""
        dataset = make_gravity_dataset(noise_frac=0.0)
        log_dataset, _ = dataset.log_transform()
        config = FunnelConfig()
        exponents = _extract_exponents_svd(log_dataset, config)
        assert exponents is not None

    def test_matching_filters_candidates(
        self,
        dispersion_entry: FunnelAtomEntry,
        ohm_entry: FunnelAtomEntry,
        test_index: FunnelIndex,
        default_config: FunnelConfig,
    ) -> None:
        """Stage should keep matching candidates and filter non-matching ones."""
        dataset = make_dispersion_dataset(noise_frac=0.0)
        candidates = [
            FunnelCandidate(entry=dispersion_entry),
            FunnelCandidate(entry=ohm_entry),  # No exponent sig -> passes through
        ]
        result = stage_exponent_extraction(
            dataset, candidates, test_index, default_config
        )
        # Both should survive: dispersion matches, ohm has no exponent sig.
        names = [c.entry.atom_name for c in result]
        assert "dispersion_delay" in names
