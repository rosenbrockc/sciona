from __future__ import annotations

import pytest

from sciona.physics_ingest.ids import (
    DeterministicIdError,
    attach_deterministic_candidate_ids,
    build_snapshot_id_bindings,
    plan_source_bundle_ids,
    source_candidate_id,
    source_snapshot_id,
    stable_payload_sha256,
)
from sciona.physics_ingest.orchestration import orchestrate_physics_publication


def test_snapshot_ids_are_uuidv5_stable_for_canonical_payload_hashes() -> None:
    snapshot_row = _snapshot_row(payload_sha256=stable_payload_sha256({"b": 2, "a": 1}))
    reordered_snapshot = _snapshot_row(
        payload_sha256=stable_payload_sha256({"a": 1, "b": 2})
    )

    snapshot_id = source_snapshot_id(snapshot_row)

    assert snapshot_id == source_snapshot_id(reordered_snapshot)
    assert snapshot_id == "88c7b9fe-23ee-595d-b260-b86acb5ac1de"
    assert source_snapshot_id(
        _snapshot_row(payload_sha256=stable_payload_sha256({"a": 1, "b": 3}))
    ) != snapshot_id


def test_build_snapshot_id_bindings_uses_orchestration_keys_without_db_ids() -> None:
    bundle = _source_bundle()

    bindings = build_snapshot_id_bindings([bundle])

    assert bindings == {
        "fixture-bundle": "26d209bf-8b31-5d14-86a4-cbdf85c76d67",
        "manual": "26d209bf-8b31-5d14-86a4-cbdf85c76d67",
        "fixture.adapter": "26d209bf-8b31-5d14-86a4-cbdf85c76d67",
        "memory://fixture": "26d209bf-8b31-5d14-86a4-cbdf85c76d67",
    }


def test_build_snapshot_id_bindings_rejects_ambiguous_binding_keys() -> None:
    first = _source_bundle(
        bundle_key="same-key",
        snapshot_row=_snapshot_row(payload_sha256="a" * 64),
    )
    second = _source_bundle(
        bundle_key="same-key",
        snapshot_row=_snapshot_row(payload_sha256="b" * 64),
    )

    with pytest.raises(DeterministicIdError, match="multiple snapshot IDs"):
        build_snapshot_id_bindings([first, second])


def test_attach_deterministic_candidate_ids_returns_copies_and_detects_collisions() -> None:
    rows = [
        _candidate_row(source_candidate_id="fixture:eq:force", source_label="first"),
        _candidate_row(source_candidate_id="fixture:eq:energy", source_label="second"),
    ]
    original = [dict(row) for row in rows]
    snapshot_id = "26d209bf-8b31-5d14-86a4-cbdf85c76d67"

    planned = attach_deterministic_candidate_ids(rows, snapshot_id)

    assert rows == original
    assert [row["candidate_id"] for row in planned] == [
        source_candidate_id(snapshot_id, "fixture:eq:force"),
        source_candidate_id(snapshot_id, "fixture:eq:energy"),
    ]
    assert "candidate_id" not in rows[0]

    with pytest.raises(DeterministicIdError, match="collision"):
        attach_deterministic_candidate_ids(
            [
                _candidate_row(source_candidate_id="duplicate", source_label="first"),
                _candidate_row(source_candidate_id="duplicate", source_label="second"),
            ],
            snapshot_id,
        )


def test_plan_source_bundle_ids_feeds_orchestration_without_mutating_inputs() -> None:
    bundle = _source_bundle()
    original_candidate = dict(bundle["candidate_rows"][0])

    snapshot_id_bindings, planned_bundles = plan_source_bundle_ids([bundle])
    result = orchestrate_physics_publication(
        source_bundles=planned_bundles,
        snapshot_id_bindings=snapshot_id_bindings,
    )

    rows = result.to_insert_rows()
    assert result.diagnostics == ()
    assert rows["physics_ingest_snapshots"][0]["snapshot_id"] == (
        "26d209bf-8b31-5d14-86a4-cbdf85c76d67"
    )
    assert rows["physics_equation_candidates"][0]["snapshot_id"] == (
        "26d209bf-8b31-5d14-86a4-cbdf85c76d67"
    )
    assert rows["physics_equation_candidates"][0]["candidate_id"] == (
        "a3fac62f-d45d-5256-9982-982dc31c2175"
    )
    assert bundle["candidate_rows"][0] == original_candidate
    assert "candidate_id" not in bundle["candidate_rows"][0]


def _source_bundle(
    *,
    bundle_key: str = "fixture-bundle",
    snapshot_row: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "bundle_key": bundle_key,
        "snapshot_row": snapshot_row or _snapshot_row(payload_sha256="a" * 64),
        "candidate_rows": [
            _candidate_row(
                source_candidate_id="fixture:eq:force",
                source_label="Newton second law",
            )
        ],
    }


def _snapshot_row(*, payload_sha256: str) -> dict[str, object]:
    return {
        "source_system": "manual",
        "source_version": "fixture-v1",
        "source_uri": "memory://fixture",
        "adapter_name": "fixture.adapter",
        "adapter_version": "1",
        "payload_sha256": payload_sha256,
        "payload": {"record_count": 1},
    }


def _candidate_row(
    *,
    source_candidate_id: str,
    source_label: str,
) -> dict[str, object]:
    return {
        "source_candidate_id": source_candidate_id,
        "source_label": source_label,
        "raw_formula": "F = m a",
        "raw_formula_format": "plain_text",
        "candidate_status": "raw_imported",
        "parse_confidence": 0.5,
        "source_payload": {"fixture": True},
    }
