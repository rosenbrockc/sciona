"""Tests for Stage 1: Boundary Triage."""

from __future__ import annotations

import numpy as np
import pytest

from sciona.symbolic_funnel.contracts import FunnelCandidate, FunnelConfig
from sciona.symbolic_funnel.dataset import ColumnMetadata, EmpiricalDataset
from sciona.symbolic_funnel.index import FunnelAtomEntry
from sciona.symbolic_funnel.stages import stage_boundary_triage

from .conftest import make_dispersion_dataset


class TestBoundaryTriage:
    def test_valid_data_passes(
        self, dispersion_entry: FunnelAtomEntry, default_config: FunnelConfig
    ) -> None:
        dataset = make_dispersion_dataset(noise_frac=0.0)
        candidates = [FunnelCandidate(entry=dispersion_entry)]
        result = stage_boundary_triage(dataset, candidates, default_config)
        assert len(result) == 1
        assert result[0].verdicts[-1].passed is True

    def test_missing_columns_rejects(
        self, dispersion_entry: FunnelAtomEntry, default_config: FunnelConfig
    ) -> None:
        # Dataset with wrong column names.
        rng = np.random.default_rng(42)
        dataset = EmpiricalDataset(
            data=rng.uniform(1, 100, (100, 3)),
            columns=[
                ColumnMetadata(name="x"),
                ColumnMetadata(name="y"),
                ColumnMetadata(name="z"),
            ],
        )
        candidates = [FunnelCandidate(entry=dispersion_entry)]
        result = stage_boundary_triage(dataset, candidates, default_config)
        assert len(result) == 0

    def test_out_of_bounds_rejects(
        self, dispersion_entry: FunnelAtomEntry, default_config: FunnelConfig
    ) -> None:
        # DM has validity bound >= 0, but we provide negative values.
        rng = np.random.default_rng(42)
        dataset = EmpiricalDataset(
            data=np.column_stack([
                rng.uniform(1, 10, 100),   # t
                rng.uniform(-100, -1, 100), # DM (all negative)
                rng.uniform(100, 2000, 100), # f
            ]),
            columns=[
                ColumnMetadata(name="t"),
                ColumnMetadata(name="DM"),
                ColumnMetadata(name="f"),
            ],
        )
        candidates = [FunnelCandidate(entry=dispersion_entry)]
        result = stage_boundary_triage(dataset, candidates, default_config)
        assert len(result) == 0

    def test_multiple_candidates_filters_correctly(
        self,
        dispersion_entry: FunnelAtomEntry,
        ohm_entry: FunnelAtomEntry,
        default_config: FunnelConfig,
    ) -> None:
        dataset = make_dispersion_dataset()
        candidates = [
            FunnelCandidate(entry=dispersion_entry),  # should pass (t, DM, f present)
            FunnelCandidate(entry=ohm_entry),  # should fail (V, I, R_val missing)
        ]
        result = stage_boundary_triage(dataset, candidates, default_config)
        assert len(result) == 1
        assert result[0].entry.atom_name == "dispersion_delay"
