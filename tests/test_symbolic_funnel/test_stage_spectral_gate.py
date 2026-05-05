"""Tests for Stage 0: Spectral Dimensionality Gate."""

from __future__ import annotations

import numpy as np
import pytest

from sciona.symbolic_funnel.contracts import FunnelConfig
from sciona.symbolic_funnel.dataset import ColumnMetadata, EmpiricalDataset
from sciona.symbolic_funnel.stages import (
    _participation_ratio,
    spectral_dimensionality_gate,
)

from .conftest import (
    make_dispersion_dataset,
    make_gravity_dataset,
    make_ideal_gas_dataset,
    make_ohm_dataset,
    make_random_dataset,
)


class TestParticipationRatio:
    def test_blob_has_high_pr(self) -> None:
        """Uncorrelated random data should have PR close to D."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((1000, 5))
        pr = _participation_ratio(data)
        assert pr is not None
        # PR should be close to 5 (all dimensions independent).
        assert pr > 4.0

    def test_rank_deficient_data_has_low_pr(self) -> None:
        """Data lying on a 1-D line should have PR close to 1."""
        rng = np.random.default_rng(42)
        t = rng.uniform(0, 10, 1000)
        # y = 2*t, z = 3*t — perfectly correlated, rank-1.
        data = np.column_stack([t, 2 * t, 3 * t])
        pr = _participation_ratio(data)
        assert pr is not None
        assert pr < 1.5

    def test_two_relationships_in_four_dims(self) -> None:
        """Data with 2 independent + 2 derived columns: PR ≈ 2."""
        rng = np.random.default_rng(42)
        x = rng.uniform(1, 100, 1000)
        y = rng.uniform(1, 100, 1000)
        z = x * y          # derived from x, y
        w = x / y           # another relationship
        data = np.column_stack([x, y, z, w])
        pr = _participation_ratio(data)
        assert pr is not None
        # 4 columns but only 2 independent: PR should be well below 4.
        assert pr < 3.0

    def test_single_column_returns_none(self) -> None:
        data = np.array([[1], [2], [3]])
        assert _participation_ratio(data) is None


class TestSpectralDimensionalityGate:
    def test_dispersion_data_passes(self) -> None:
        """Dispersion delay data (t = K*DM/f^2) has clear structure."""
        dataset = make_dispersion_dataset(noise_frac=0.01)
        config = FunnelConfig()
        verdict = spectral_dimensionality_gate(dataset, config)
        assert verdict.passed is True
        assert verdict.evidence.get("log_participation_ratio") is not None

    def test_ideal_gas_passes(self) -> None:
        dataset = make_ideal_gas_dataset(noise_frac=0.01)
        config = FunnelConfig()
        verdict = spectral_dimensionality_gate(dataset, config)
        assert verdict.passed is True

    def test_gravity_passes(self) -> None:
        dataset = make_gravity_dataset(noise_frac=0.01)
        config = FunnelConfig()
        verdict = spectral_dimensionality_gate(dataset, config)
        assert verdict.passed is True

    def test_ohm_law_passes(self) -> None:
        dataset = make_ohm_dataset(noise_frac=0.01)
        config = FunnelConfig()
        verdict = spectral_dimensionality_gate(dataset, config)
        assert verdict.passed is True

    def test_random_blob_rejected(self) -> None:
        """Pure random noise should fail the gate."""
        dataset = make_random_dataset(n=1000, n_cols=4)
        config = FunnelConfig()
        verdict = spectral_dimensionality_gate(dataset, config)
        assert verdict.passed is False
        assert verdict.evidence["reason"] == "no_structure_detected"

    def test_high_dim_random_blob_rejected(self) -> None:
        """8-dimensional random blob should still be rejected."""
        dataset = make_random_dataset(n=1000, n_cols=8)
        config = FunnelConfig()
        verdict = spectral_dimensionality_gate(dataset, config)
        assert verdict.passed is False

    def test_gate_disabled_always_passes(self) -> None:
        """When gate is disabled, funnel should run even on blobs."""
        from sciona.symbolic_funnel.index import FunnelIndex
        from sciona.symbolic_funnel.stages import run_funnel

        dataset = make_random_dataset()
        index = FunnelIndex()  # Empty index.
        config = FunnelConfig(spectral_gate_enabled=False)
        result = run_funnel(dataset, index, config)
        assert not result.gated
        assert "spectral_gate" not in result.stages_executed

    def test_gated_result_has_no_candidates(self) -> None:
        """When gated, the result should have no candidates."""
        from sciona.symbolic_funnel.index import FunnelIndex
        from sciona.symbolic_funnel.stages import run_funnel

        dataset = make_random_dataset()
        index = FunnelIndex()
        config = FunnelConfig(spectral_gate_enabled=True)
        result = run_funnel(dataset, index, config)
        assert result.gated is True
        assert len(result.ranked_candidates) == 0
        assert "spectral_gate" in result.stages_executed

    def test_noisy_power_law_passes_via_log_space(self) -> None:
        """Power-law data that looks blobby in linear space should
        still pass because log-transform reveals the structure."""
        rng = np.random.default_rng(42)
        # F = G * m1 * m2 / r^2 — wide dynamic range makes linear blob
        m1 = rng.uniform(1e10, 1e30, 500)
        m2 = rng.uniform(1e10, 1e30, 500)
        r = rng.uniform(1e3, 1e12, 500)
        F = 6.674e-11 * m1 * m2 / r**2
        # Add 5% noise.
        F *= 1 + 0.05 * rng.standard_normal(500)
        dataset = EmpiricalDataset(
            data=np.column_stack([F, m1, m2, r]),
            columns=[
                ColumnMetadata(name="F"),
                ColumnMetadata(name="m1"),
                ColumnMetadata(name="m2"),
                ColumnMetadata(name="r"),
            ],
        )
        config = FunnelConfig()
        verdict = spectral_dimensionality_gate(dataset, config)
        assert verdict.passed is True
