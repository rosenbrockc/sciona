"""Tests for ingest cache envelope stability and compatibility."""

from __future__ import annotations

import json

from sciona.ingester.cache import (
    compute_ingest_cache_key,
    load_ingest_cache,
    save_ingest_cache,
)
from sciona.ingester.models import IngestionBundle


def _sample_bundle() -> IngestionBundle:
    return IngestionBundle.model_validate(
        {
            "cdg": {"nodes": [], "edges": []},
            "generated_atoms": "class Atom:\n    pass\n",
            "generated_state_models": "",
            "generated_witnesses": "def witness_atom():\n    return True\n",
            "mypy_passed": True,
            "ghost_sim_passed": False,
            "ghost_sim_report": {"reason": "fixture"},
        }
    )


def test_save_and_load_cache_round_trip_uses_versioned_envelope(tmp_path):
    key = "cache-key-fixture"
    bundle = _sample_bundle()

    path = save_ingest_cache(tmp_path, key, bundle)
    payload = json.loads(path.read_text())

    assert payload["schema"] == "sciona.ingester.cache-envelope"
    assert payload["schema_version"] == 1
    assert payload["runtime_mode"] == "canonical-first"
    assert payload["payload_kind"] == "ingestion_bundle"
    assert payload["cache_key"] == key
    assert payload["payload_summary"]["cdg_node_count"] == 0
    assert payload["payload_summary"]["match_result_count"] == 0

    restored = load_ingest_cache(tmp_path, key)
    assert restored is not None
    assert restored.generated_atoms == bundle.generated_atoms
    assert restored.generated_witnesses == bundle.generated_witnesses
    assert restored.mypy_passed is True
    assert restored.ghost_sim_passed is False
    assert restored.ghost_sim_report == {"reason": "fixture"}


def test_load_cache_accepts_legacy_bundle_payload_shape(tmp_path):
    key = "legacy-shape"
    (tmp_path / f"{key}.json").write_text(
        json.dumps(
            {
                "cdg": {"nodes": [], "edges": []},
                "generated_atoms": "class Legacy:\n    pass\n",
                "generated_witnesses": "def witness_legacy():\n    return None\n",
                "mypy_passed": False,
                "ghost_sim_passed": False,
            }
        )
    )

    restored = load_ingest_cache(tmp_path, key)
    assert restored is not None
    assert restored.generated_atoms.startswith("class Legacy")
    assert restored.generated_witnesses.startswith("def witness_legacy")
    assert restored.generated_state_models == ""
    assert restored.sub_graphs == {}


def test_load_cache_rejects_unknown_envelope_schema_version(tmp_path):
    key = "unknown-schema-version"
    (tmp_path / f"{key}.json").write_text(
        json.dumps(
            {
                "schema": "sciona.ingester.cache-envelope",
                "schema_version": 999,
                "payload_kind": "ingestion_bundle",
                "payload": {"cdg": {"nodes": [], "edges": []}},
            }
        )
    )

    assert load_ingest_cache(tmp_path, key) is None


def test_load_cache_tolerates_partial_legacy_payload_and_bad_report_type(tmp_path):
    key = "partial-legacy"
    (tmp_path / f"{key}.json").write_text(
        json.dumps(
            {
                "generated_atoms": "class Partial:\n    pass\n",
                "ghost_sim_report": "not-a-dict",
            }
        )
    )

    restored = load_ingest_cache(tmp_path, key)
    assert restored is not None
    assert restored.generated_atoms.startswith("class Partial")
    assert restored.cdg.nodes == []
    assert restored.cdg.edges == []
    assert restored.ghost_sim_report == {}


def test_compute_ingest_cache_key_is_deterministic_and_parameter_sensitive(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("class A:\n    pass\n")

    key_a = compute_ingest_cache_key(
        source_path=str(source),
        class_name="A",
        max_depth=1,
        line_threshold=30,
    )
    key_a_repeat = compute_ingest_cache_key(
        source_path=str(source),
        class_name="A",
        max_depth=1,
        line_threshold=30,
    )
    key_b = compute_ingest_cache_key(
        source_path=str(source),
        class_name="B",
        max_depth=1,
        line_threshold=30,
    )

    assert key_a == key_a_repeat
    assert key_a != key_b
