"""Tests for baseline analysis runtime atoms, registry, and catalog seeding."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from sciona.architect.catalog import PrimitiveCatalog, seed_builtin_primitives
from sciona.architect.models import ConceptType, ParamStatus
from sciona.expansion_atoms.baseline_analysis_registry import (
    BASELINE_ANALYSIS_DECLARATIONS,
)
from sciona.expansion_atoms.runtime_baseline_analysis import (
    check_onset_coverage,
    detect_padding_saturation,
    monitor_normalization_clipping,
    validate_component_balance,
)


class TestCheckOnsetCoverage:
    def test_empty_results_have_low_density(self):
        density, ok = check_onset_coverage([], 1000)
        assert density == 0.0
        assert not ok

    def test_many_onsets_is_sufficient(self):
        density, ok = check_onset_coverage(list(range(4)), 1000)
        assert density == pytest.approx(0.004)
        assert ok

    def test_zero_length_signal_fails(self):
        density, ok = check_onset_coverage([1, 2], 0)
        assert density == 0.0
        assert not ok


class TestDetectPaddingSaturation:
    def test_short_padding_not_saturated(self):
        fraction, saturated = detect_padding_saturation(np.ones(12), 10)
        assert fraction == pytest.approx(2 / 12)
        assert not saturated

    def test_long_padding_is_saturated(self):
        fraction, saturated = detect_padding_saturation(np.ones(20), 5)
        assert fraction == pytest.approx(15 / 20)
        assert saturated

    def test_empty_array_is_safe(self):
        fraction, saturated = detect_padding_saturation(np.array([]), 10)
        assert fraction == 0.0
        assert not saturated


class TestMonitorNormalizationClipping:
    def test_varied_values_not_clipped(self):
        fraction, clipped = monitor_normalization_clipping(np.array([0.1, 0.5, 0.9]))
        assert fraction == pytest.approx(0.0)
        assert not clipped

    def test_mostly_ones_is_clipped(self):
        fraction, clipped = monitor_normalization_clipping(
            np.array([1.0, 1.0, 0.8, 1.0, 1.0])
        )
        assert fraction == pytest.approx(0.8)
        assert clipped

    def test_empty_array_is_safe(self):
        fraction, clipped = monitor_normalization_clipping(np.array([]))
        assert fraction == 0.0
        assert not clipped


class TestValidateComponentBalance:
    def test_equal_energy_is_balanced(self):
        entropy, balanced = validate_component_balance(
            [np.array([1.0, 1.0]), np.array([1.0, 1.0])]
        )
        assert entropy == pytest.approx(1.0)
        assert balanced

    def test_dominant_component_is_unbalanced(self):
        entropy, balanced = validate_component_balance(
            [np.array([10.0, 0.0]), np.array([0.1, 0.0])]
        )
        assert entropy < 0.5
        assert not balanced

    def test_single_component_is_balanced(self):
        entropy, balanced = validate_component_balance([np.array([1.0, 2.0, 3.0])])
        assert entropy == pytest.approx(1.0)
        assert balanced

    def test_zero_energy_components_are_balanced(self):
        entropy, balanced = validate_component_balance(
            [np.array([0.0, 0.0]), np.array([0.0, 0.0])]
        )
        assert entropy == pytest.approx(1.0)
        assert balanced


class TestRegistry:
    def test_all_expected_declarations_present(self):
        assert set(BASELINE_ANALYSIS_DECLARATIONS) == {
            "check_onset_coverage",
            "detect_padding_saturation",
            "monitor_normalization_clipping",
            "validate_component_balance",
        }

    @pytest.mark.parametrize("name", sorted(BASELINE_ANALYSIS_DECLARATIONS))
    def test_declaration_fqdns_are_importable(self, name: str):
        fqdn, _sig, _desc = BASELINE_ANALYSIS_DECLARATIONS[name]
        module_name, attr_name = fqdn.rsplit(".", 1)
        module = importlib.import_module(module_name)
        assert hasattr(module, attr_name)


class TestBaselineFitStackCatalog:
    def test_builtin_seed_registers_primitive(self):
        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)

        primitive = catalog.get("baseline_fit_stack")
        assert primitive is not None
        assert primitive.category == ConceptType.BASELINE_ANALYSIS
        assert primitive.param_status == ParamStatus.APPROVED
        assert [param.name for param in primitive.tunable_params] == [
            "onset_threshold",
            "center_hold_samples",
            "offset_decay_rate",
            "min_event_gap",
        ]

    def test_builtin_seed_registers_aliases(self):
        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)

        assert catalog.get("fit stack") is not None
        assert catalog.get("baseline fit stack") is not None
