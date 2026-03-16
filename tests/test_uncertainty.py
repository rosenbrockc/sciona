"""Tests for the uncertainty estimation backends."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ageom.synthesizer.uncertainty import (
    AnalyticBackend,
    AtomUncertaintyEstimate,
    CatalogBackend,
    ChainBackend,
    HeuristicBackend,
    _numerical_jacobian,
    load_uncertainty_json,
)


# ---------------------------------------------------------------------------
# HeuristicBackend
# ---------------------------------------------------------------------------


class TestHeuristicBackend:
    def setup_method(self):
        self.backend = HeuristicBackend()

    def test_exact_match_fft(self):
        est = self.backend.estimate("fft")
        assert est.mode == "heuristic"
        assert est.scalar_factor == 1.5
        assert est.confidence == 0.2

    def test_exact_match_lfilter(self):
        est = self.backend.estimate("lfilter")
        assert est.scalar_factor == 2.0

    def test_keyword_filter(self):
        est = self.backend.estimate("bandpass_filter")
        assert est.mode == "heuristic"
        assert est.scalar_factor == 1.2
        assert est.confidence == 0.1

    def test_keyword_detect(self):
        est = self.backend.estimate("peak_detect")
        assert est.scalar_factor == 1.35

    def test_keyword_rate(self):
        est = self.backend.estimate("heart_rate")
        assert est.scalar_factor == 1.15

    def test_unknown_atom(self):
        est = self.backend.estimate("totally_unknown_op_xyz")
        assert est.mode == "unknown"
        assert est.scalar_factor is None
        assert est.confidence == 0.0


# ---------------------------------------------------------------------------
# CatalogBackend
# ---------------------------------------------------------------------------


class TestCatalogBackend:
    def test_lookup_hit(self):
        catalog = {
            "my_atom": AtomUncertaintyEstimate(
                mode="empirical", scalar_factor=2.5, confidence=0.8
            )
        }
        backend = CatalogBackend(catalog)
        est = backend.estimate("my_atom")
        assert est.mode == "empirical"
        assert est.scalar_factor == 2.5

    def test_lookup_miss(self):
        backend = CatalogBackend({})
        est = backend.estimate("missing_atom")
        assert est.mode == "unknown"


# ---------------------------------------------------------------------------
# ChainBackend
# ---------------------------------------------------------------------------


class TestChainBackend:
    def test_first_non_unknown_wins(self):
        empty = CatalogBackend({})
        full = CatalogBackend(
            {"x": AtomUncertaintyEstimate(mode="empirical", scalar_factor=3.0)}
        )
        chain = ChainBackend(empty, full)
        est = chain.estimate("x")
        assert est.mode == "empirical"
        assert est.scalar_factor == 3.0

    def test_all_unknown(self):
        chain = ChainBackend(CatalogBackend({}), CatalogBackend({}))
        est = chain.estimate("x")
        assert est.mode == "unknown"

    def test_heuristic_fallback(self):
        empty_catalog = CatalogBackend({})
        heuristic = HeuristicBackend()
        chain = ChainBackend(empty_catalog, heuristic)
        est = chain.estimate("fft")
        assert est.mode == "heuristic"
        assert est.scalar_factor == 1.5

    def test_catalog_overrides_heuristic(self):
        catalog = CatalogBackend(
            {"fft": AtomUncertaintyEstimate(mode="empirical", scalar_factor=1.7, confidence=0.9)}
        )
        heuristic = HeuristicBackend()
        chain = ChainBackend(catalog, heuristic)
        est = chain.estimate("fft")
        assert est.mode == "empirical"
        assert est.scalar_factor == 1.7


# ---------------------------------------------------------------------------
# load_uncertainty_json
# ---------------------------------------------------------------------------


class TestLoadUncertaintyJson:
    def test_valid_file(self, tmp_path):
        data = {
            "atom": "fft",
            "estimates": [
                {"mode": "heuristic", "scalar_factor": 1.5, "confidence": 0.2},
                {"mode": "empirical", "scalar_factor": 1.6, "confidence": 0.85, "n_trials": 500},
            ],
        }
        p = tmp_path / "uncertainty.json"
        p.write_text(json.dumps(data))
        name, est = load_uncertainty_json(p)
        assert name == "fft"
        assert est.mode == "empirical"
        assert est.scalar_factor == 1.6
        assert est.confidence == 0.85
        assert est.n_trials == 500

    def test_empty_estimates(self, tmp_path):
        data = {"atom": "fft", "estimates": []}
        p = tmp_path / "uncertainty.json"
        p.write_text(json.dumps(data))
        name, est = load_uncertainty_json(p)
        assert name == "fft"
        assert est.mode == "unknown"

    def test_malformed_json(self, tmp_path):
        p = tmp_path / "uncertainty.json"
        p.write_text("{bad json")
        name, est = load_uncertainty_json(p)
        assert name == ""
        assert est.mode == "unknown"

    def test_missing_file(self):
        name, est = load_uncertainty_json("/nonexistent/path/uncertainty.json")
        assert est.mode == "unknown"

    def test_picks_highest_confidence(self, tmp_path):
        data = {
            "atom": "butter",
            "estimates": [
                {"mode": "heuristic", "scalar_factor": 1.1, "confidence": 0.2},
                {"mode": "analytic", "scalar_factor": 1.15, "confidence": 0.7},
                {"mode": "empirical", "scalar_factor": 1.12, "confidence": 0.9},
            ],
        }
        p = tmp_path / "uncertainty.json"
        p.write_text(json.dumps(data))
        name, est = load_uncertainty_json(p)
        assert est.mode == "empirical"
        assert est.confidence == 0.9


# ---------------------------------------------------------------------------
# AnalyticBackend + _numerical_jacobian
# ---------------------------------------------------------------------------


class TestNumericalJacobian:
    def test_identity_jacobian(self):
        def identity(x):
            return x

        x0 = np.array([1.0, 2.0, 3.0])
        jac = _numerical_jacobian(identity, x0)
        np.testing.assert_allclose(jac, np.eye(3), atol=1e-5)

    def test_scaling_jacobian(self):
        def scale(x):
            return 2.0 * x

        x0 = np.array([1.0, 2.0])
        jac = _numerical_jacobian(scale, x0)
        np.testing.assert_allclose(jac, 2.0 * np.eye(2), atol=1e-5)

    def test_linear_map_jacobian(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])

        def linear(x):
            return A @ x

        x0 = np.array([1.0, 1.0])
        jac = _numerical_jacobian(linear, x0)
        np.testing.assert_allclose(jac, A, atol=1e-5)


class TestAnalyticBackend:
    def test_identity_factor_near_one(self):
        def identity(x):
            return x

        backend = AnalyticBackend(
            atom_registry={"identity": identity},
            base_inputs={"identity": np.array([1.0, 2.0, 3.0])},
            stable_atoms={"identity"},
        )
        est = backend.estimate("identity")
        assert est.mode == "analytic"
        assert est.scalar_factor is not None
        assert abs(est.scalar_factor - 1.0) < 0.01

    def test_unstable_atom_returns_unknown(self):
        backend = AnalyticBackend(
            atom_registry={"fft": np.fft.fft},
            base_inputs={"fft": np.random.randn(64)},
            stable_atoms=set(),  # fft not marked stable
        )
        est = backend.estimate("fft")
        assert est.mode == "unknown"

    def test_missing_atom_returns_unknown(self):
        backend = AnalyticBackend(
            atom_registry={},
            base_inputs={},
            stable_atoms={"nope"},
        )
        est = backend.estimate("nope")
        assert est.mode == "unknown"

    def test_large_input_returns_unknown(self):
        def identity(x):
            return x

        backend = AnalyticBackend(
            atom_registry={"big": identity},
            base_inputs={"big": np.zeros(1000)},
            stable_atoms={"big"},
        )
        est = backend.estimate("big")
        assert est.mode == "unknown"
        assert "exceeds cap" in est.notes
