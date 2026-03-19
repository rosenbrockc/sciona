"""Tests for Phase 1 — PrimitiveParamSpec model, manifest loader, catalog integration."""

from __future__ import annotations

import json
import sqlite3

import pytest

from ageom.architect.catalog import PrimitiveCatalog
from ageom.architect.hyperparams import (
    get_runtime_signal_event_rate_params,
    load_hyperparams_manifest,
    load_hyperparams_manifest_sqlite,
    load_manifest,
)
from ageom.architect.models import (
    AlgorithmicPrimitive,
    ConceptType,
    ParamStatus,
    PrimitiveParamSpec,
)


def _create_test_db(db_path, atoms=None, hyperparams=None):
    """Create a minimal manifest.sqlite with the real schema."""
    con = sqlite3.connect(str(db_path))
    con.executescript(
        """
        CREATE TABLE atoms (
            atom_id INTEGER PRIMARY KEY,
            fqdn TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'fixed'
        );
        CREATE TABLE hyperparams (
            hp_id INTEGER PRIMARY KEY,
            atom_id INTEGER NOT NULL REFERENCES atoms(atom_id),
            name TEXT NOT NULL,
            kind TEXT,
            default_value TEXT,
            min_value TEXT,
            max_value TEXT,
            step_value TEXT,
            log_scale INTEGER DEFAULT 0,
            choices_json TEXT,
            constraints_json TEXT,
            semantic_role TEXT,
            status TEXT NOT NULL DEFAULT 'approved'
        );
        """
    )
    for atom in atoms or []:
        con.execute(
            "INSERT INTO atoms (atom_id, fqdn, status) VALUES (?, ?, ?)",
            atom,
        )
    for hp in hyperparams or []:
        con.execute(
            "INSERT INTO hyperparams "
            "(atom_id, name, default_value, min_value, max_value, step_value, "
            "log_scale, choices_json, constraints_json, semantic_role, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            hp,
        )
    con.commit()
    con.close()


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


class TestLoadManifestJSON:
    """Tests for JSON manifest loader using real reviewed_atoms schema."""

    def test_filters_blocked_atoms(self, tmp_path):
        manifest = {
            "reviewed_atoms": [
                {
                    "atom": "blocked_atom",
                    "status": "blocked",
                    "tunable_params": [
                        {"name": "x", "kind": "float", "default": 1.0}
                    ],
                },
                {
                    "atom": "approved_atom",
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
            "reviewed_atoms": [
                {
                    "atom": "my_atom",
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

    def test_infers_kind_when_missing(self, tmp_path):
        manifest = {
            "reviewed_atoms": [
                {
                    "atom": "infer_atom",
                    "status": "approved",
                    "tunable_params": [
                        {"name": "rate", "default": 0.5, "min_value": 0.0, "max_value": 1.0},
                    ],
                }
            ]
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))
        result = load_hyperparams_manifest(path)
        assert result["infer_atom"][0].kind == "float"


class TestLoadManifestSQLite:
    """Tests for SQLite manifest loader."""

    def test_load_sqlite_approved_params(self, tmp_path):
        db = tmp_path / "manifest.sqlite"
        _create_test_db(
            db,
            atoms=[(1, "pkg.mod.my_filter", "approved")],
            hyperparams=[
                # atom_id, name, default, min, max, step, log_scale, choices, constraints, role, status
                (1, "cutoff", "0.5", "0.1", "1.0", "null", 0, "null", "null", "low-pass cutoff", "approved"),
            ],
        )
        result = load_hyperparams_manifest_sqlite(db)
        assert "pkg.mod.my_filter" in result
        specs = result["pkg.mod.my_filter"]
        assert len(specs) == 1
        assert specs[0].name == "cutoff"
        assert specs[0].kind == "float"
        assert specs[0].default == 0.5
        assert specs[0].min_value == 0.1
        assert specs[0].max_value == 1.0
        assert specs[0].safe_to_optimize is True

    def test_load_sqlite_filters_blocked(self, tmp_path):
        db = tmp_path / "manifest.sqlite"
        _create_test_db(
            db,
            atoms=[(1, "pkg.blocked_atom", "blocked")],
            hyperparams=[
                (1, "x", "1.0", "null", "null", "null", 0, "null", "null", "", "approved"),
            ],
        )
        result = load_hyperparams_manifest_sqlite(db)
        assert result == {}

    def test_load_sqlite_filters_unapproved_param(self, tmp_path):
        db = tmp_path / "manifest.sqlite"
        _create_test_db(
            db,
            atoms=[(1, "pkg.good_atom", "approved")],
            hyperparams=[
                (1, "ok_param", "1.0", "null", "null", "null", 0, "null", "null", "", "approved"),
                (1, "bad_param", "2.0", "null", "null", "null", 0, "null", "null", "", "blocked"),
            ],
        )
        result = load_hyperparams_manifest_sqlite(db)
        assert len(result["pkg.good_atom"]) == 1
        assert result["pkg.good_atom"][0].name == "ok_param"

    def test_load_sqlite_infers_kind(self, tmp_path):
        db = tmp_path / "manifest.sqlite"
        _create_test_db(
            db,
            atoms=[(1, "pkg.types", "approved")],
            hyperparams=[
                (1, "int_p", "4", "null", "null", "null", 0, "null", "null", "", "approved"),
                (1, "float_p", "0.001", "null", "null", "null", 0, "null", "null", "", "approved"),
                (1, "bool_p", "true", "null", "null", "null", 0, "null", "null", "", "approved"),
                (1, "cat_p", '"linear"', "null", "null", "null", 0, '["linear","cubic"]', "null", "", "approved"),
            ],
        )
        result = load_hyperparams_manifest_sqlite(db)
        specs = {s.name: s for s in result["pkg.types"]}
        assert specs["int_p"].kind == "int"
        assert specs["float_p"].kind == "float"
        assert specs["bool_p"].kind == "bool"
        assert specs["cat_p"].kind == "categorical"
        assert specs["cat_p"].choices == ["linear", "cubic"]

    def test_load_sqlite_json_decoding(self, tmp_path):
        db = tmp_path / "manifest.sqlite"
        _create_test_db(
            db,
            atoms=[(1, "pkg.decode", "approved")],
            hyperparams=[
                (1, "p", "0.001", "null", "0.1", "0.0005", 1, "null", "null", "rate", "approved"),
            ],
        )
        result = load_hyperparams_manifest_sqlite(db)
        spec = result["pkg.decode"][0]
        assert spec.default == 0.001
        assert spec.min_value is None
        assert spec.max_value == 0.1
        assert spec.step == 0.0005
        assert spec.log_scale is True
        assert spec.semantic_role == "rate"

    def test_load_sqlite_missing_db_returns_empty(self, tmp_path):
        result = load_hyperparams_manifest_sqlite(tmp_path / "missing.sqlite")
        assert result == {}


class TestLoadManifestConvenience:
    """Tests for load_manifest() that auto-selects SQLite vs JSON."""

    def test_prefers_sqlite(self, tmp_path):
        # Create both files; SQLite has an approved atom, JSON does not
        db = tmp_path / "manifest.sqlite"
        _create_test_db(
            db,
            atoms=[(1, "pkg.sqlite_atom", "approved")],
            hyperparams=[
                (1, "s", "1.0", "null", "null", "null", 0, "null", "null", "", "approved"),
            ],
        )
        json_path = tmp_path / "manifest.json"
        json_path.write_text(json.dumps({
            "reviewed_atoms": [{
                "atom": "json_atom",
                "status": "approved",
                "tunable_params": [{"name": "j", "kind": "float", "default": 1.0}],
            }]
        }))

        result = load_manifest(json_path)
        # Should have loaded from SQLite, not JSON
        assert "pkg.sqlite_atom" in result
        assert "json_atom" not in result

    def test_falls_back_to_json(self, tmp_path):
        json_path = tmp_path / "manifest.json"
        json_path.write_text(json.dumps({
            "reviewed_atoms": [{
                "atom": "json_atom",
                "status": "approved",
                "tunable_params": [{"name": "j", "kind": "float", "default": 1.0}],
            }]
        }))
        result = load_manifest(json_path)
        assert "json_atom" in result

    def test_returns_empty_when_nothing_exists(self, tmp_path):
        result = load_manifest(tmp_path / "manifest.json")
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
