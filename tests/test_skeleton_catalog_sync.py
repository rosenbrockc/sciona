from __future__ import annotations

from dataclasses import dataclass

import pytest

from sciona.architect.skeleton_assets import load_local_skeleton_assets
from sciona.services.skeleton_catalog_sync import (
    build_skeleton_artifact_bundle,
    load_skeleton_artifact_bundles,
    sync_bundle_to_supabase,
    sync_bundles_to_graph_store,
)


class _FakeTable:
    def __init__(self, client: "_FakeSupabase", name: str) -> None:
        self.client = client
        self.name = name
        self.action = ""
        self.payload = None
        self.filters: list[tuple[str, str, object]] = []

    def upsert(self, payload):
        self.action = "upsert"
        self.payload = payload
        return self

    def update(self, payload):
        self.action = "update"
        self.payload = payload
        return self

    def delete(self):
        self.action = "delete"
        self.payload = None
        return self

    def eq(self, field: str, value: object):
        self.filters.append(("eq", field, value))
        return self

    def execute(self):
        self.client.calls.append(
            {
                "table": self.name,
                "action": self.action,
                "payload": self.payload,
                "filters": list(self.filters),
            }
        )
        return self


class _FakeSupabase:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def table(self, name: str) -> _FakeTable:
        return _FakeTable(self, name)


@dataclass
class _FakeGraphStore:
    ensure_constraints_calls: int = 0
    projections: list[object] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.projections = []

    async def ensure_constraints(self) -> None:
        self.ensure_constraints_calls += 1

    async def upsert_published_cdg(self, projection):
        self.projections.append(projection)
        return {"artifacts": 1, "nodes": len(projection.nodes), "data_flow": len(projection.edges), "parent_of": 0}


def test_build_skeleton_artifact_bundle_signal_detect_measure() -> None:
    bundle = next(
        bundle
        for bundle in load_skeleton_artifact_bundles()
        if bundle.asset_id == "signal_detect_measure"
    )

    assert bundle.artifact["fqdn"] == "cdg.skeleton.signal_detect_measure"
    assert bundle.version["semver"] == "phase2.v1"
    assert bundle.artifact["top_level_input_arity"] == 2
    assert bundle.artifact["top_level_output_arity"] == 1
    assert bundle.artifact["topo_hash"]
    assert any(row["kind"] == "dejargonized" for row in bundle.descriptions)
    assert len(bundle.references_registry) == len(bundle.references) == 2
    assert len(bundle.cdg_nodes) == 3
    assert len(bundle.cdg_edges) == 2


def test_sync_bundle_to_supabase_uses_deterministic_child_table_reload() -> None:
    bundle = build_skeleton_artifact_bundle(load_local_skeleton_assets()[0])
    supabase = _FakeSupabase()

    sync_bundle_to_supabase(supabase, bundle)

    operations = [(call["table"], call["action"]) for call in supabase.calls]
    assert operations[:3] == [
        ("artifacts", "upsert"),
        ("artifact_versions", "update"),
        ("artifact_versions", "upsert"),
    ]
    assert ("artifact_descriptions", "delete") in operations
    assert ("artifact_descriptions", "upsert") in operations
    assert ("artifact_io_specs", "delete") in operations
    assert ("artifact_io_specs", "upsert") in operations
    assert ("references_registry", "upsert") in operations
    assert ("artifact_references", "upsert") in operations
    assert ("artifact_audit_rollups", "upsert") in operations
    assert ("artifact_cdg_nodes", "upsert") in operations
    assert ("artifact_cdg_edges", "upsert") in operations


@pytest.mark.asyncio
async def test_sync_bundles_to_graph_store_uses_sorted_projection_order() -> None:
    bundles = load_skeleton_artifact_bundles()
    graph_store = _FakeGraphStore()

    results = await sync_bundles_to_graph_store(graph_store, bundles)

    assert graph_store.ensure_constraints_calls == 1
    assert [projection.fqdn for projection in graph_store.projections] == sorted(
        projection.fqdn for projection in graph_store.projections
    )
    assert [asset_id for asset_id, _counts in results] == [bundle.asset_id for bundle in bundles]
