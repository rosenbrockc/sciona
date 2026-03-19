"""Tests for Phase 1 — PrimitiveParamSpec model, manifest loader, catalog integration."""

from __future__ import annotations

import json

import pytest

from ageom.architect.catalog import PrimitiveCatalog
from ageom.architect.hyperparams import (
    get_runtime_signal_event_rate_params,
    load_hyperparams_manifest,
)
from ageom.architect.models import (
    AlgorithmicPrimitive,
    ConceptType,
    ParamStatus,
    PrimitiveParamSpec,
)


class TestPrimitiveParamSpec:
    def test_valid_float_spec(self):
        spec = PrimitiveParamSpec(
            name="threshold",
            kind="float",
            default=1.5,
            min_value=0.5,
            max_value=5.0,
        )
        assert spec.name == "threshold"
        assert spec.kind == "float"

    def test_valid_int_spec(self):
        spec = PrimitiveParamSpec(
            name="order",
            kind="int",
            default=4,
            min_value=2,
            max_value=8,
            step=2,
        )
        assert spec.step == 2

    def test_valid_categorical_spec(self):
        spec = PrimitiveParamSpec(
            name="method",
            kind="categorical",
            default="butterworth",
            choices=["butterworth", "chebyshev", "bessel"],
        )
        assert spec.choices is not None
        assert len(spec.choices) == 3

    def test_rejects_min_greater_than_max(self):
        with pytest.raises(ValueError, match="min_value.*max_value"):
            PrimitiveParamSpec(
                name="bad",
                kind="float",
                default=1.0,
                min_value=10.0,
                max_value=5.0,
            )

    def test_none_bounds_allowed(self):
        spec = PrimitiveParamSpec(
            name="unbounded",
            kind="float",
            default=1.0,
        )
        assert spec.min_value is None
        assert spec.max_value is None


class TestLoadManifest:
    def test_filters_blocked_atoms(self, tmp_path):
        manifest = {
            "atoms": [
                {
                    "name": "blocked_atom",
                    "status": "blocked",
                    "tunable_params": [
                        {"name": "x", "kind": "float", "default": 1.0}
                    ],
                },
                {
                    "name": "approved_atom",
                    "status": "approved",
                    "tunable_params": [
                        {"name": "y", "kind": "float", "default": 2.0, "min_value": 0.0, "max_value": 10.0}
                    ],
                },
            ]
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))
        result = load_hyperparams_manifest(path)
        assert "blocked_atom" not in result
        assert "approved_atom" in result
        assert len(result["approved_atom"]) == 1

    def test_approved_params_correct(self, tmp_path):
        manifest = {
            "atoms": [
                {
                    "name": "my_atom",
                    "status": "approved",
                    "tunable_params": [
                        {
                            "name": "alpha",
                            "kind": "float",
                            "default": 0.5,
                            "min_value": 0.0,
                            "max_value": 1.0,
                            "semantic_role": "learning rate",
                        },
                        {
                            "name": "unsafe_param",
                            "kind": "float",
                            "default": 1.0,
                            "safe_to_optimize": False,
                        },
                    ],
                }
            ]
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))
        result = load_hyperparams_manifest(path)
        assert "my_atom" in result
        # unsafe param should be filtered out
        assert len(result["my_atom"]) == 1
        assert result["my_atom"][0].name == "alpha"

    def test_missing_manifest_returns_empty(self, tmp_path):
        result = load_hyperparams_manifest(tmp_path / "nonexistent.json")
        assert result == {}


class TestRuntimeSignalEventRateParams:
    def test_returns_all_seven_params(self):
        params = get_runtime_signal_event_rate_params()
        all_names = [p.name for specs in params.values() for p in specs]
        assert len(all_names) == 7
        expected = {
            "filter_order", "clipping_scale", "low_cutoff_hz", "high_cutoff_hz",
            "prominence_scale", "refractory_scale", "smoothing_window",
        }
        assert set(all_names) == expected

    def test_all_specs_valid(self):
        params = get_runtime_signal_event_rate_params()
        for func_name, specs in params.items():
            for spec in specs:
                assert spec.name
                assert spec.kind in ("int", "float", "categorical", "bool")
                if spec.min_value is not None and spec.max_value is not None:
                    assert spec.min_value <= spec.max_value


class TestCatalogAttachesTunables:
    def test_attach_tunables(self):
        catalog = PrimitiveCatalog()
        prim = AlgorithmicPrimitive(
            name="filter_signal_for_detection",
            source="runtime",
            category=ConceptType.SIGNAL_FILTER,
            description="Bandpass filter for event detection",
        )
        catalog.add(prim)

        tunables = get_runtime_signal_event_rate_params()
        count = catalog.attach_tunables(tunables)
        assert count >= 1

        loaded = catalog.get("filter_signal_for_detection")
        assert loaded is not None
        assert len(loaded.tunable_params) == 4
        assert loaded.param_status == ParamStatus.APPROVED

    def test_attach_skips_unknown_primitives(self):
        catalog = PrimitiveCatalog()
        tunables = {"nonexistent_func": [
            PrimitiveParamSpec(name="x", kind="float", default=1.0)
        ]}
        count = catalog.attach_tunables(tunables)
        assert count == 0
