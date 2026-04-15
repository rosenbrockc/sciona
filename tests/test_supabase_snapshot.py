from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from sciona.api.routers.catalog import get_artifact_document, get_atom_document
from sciona.api.snapshot import (
    DEFAULT_FILTER_BATCH_SIZE,
    export_tiered_manifests,
    fetch_manifest_data,
    generate_manifest_sqlite,
)
from sciona.commands.catalog_cmds import _cmd_catalog_sync, _resolve_manifest_url
from sciona.ecosystem.benchmarks import load_benchmarks_sqlite


@pytest.mark.asyncio
async def test_fetch_manifest_data_normalizes_benchmark_names(monkeypatch):
    atoms = [
        {
            "atom_id": "a1",
            "fqdn": "pkg.filter",
            "status": "approved",
            "domain_tags": ["signal"],
            "description": "Filter signal",
            "visibility_tier": "general",
            "source_kind": "hand_written",
            "stateful_kind": "none",
            "is_stochastic": False,
            "is_ffi": False,
            "namespace_root": "sciona.atoms",
            "namespace_path": "",
            "source_repo_id": None,
            "source_package": "",
            "source_module_path": "",
            "source_symbol": "",
            "is_publishable": True,
        }
    ]
    hyperparams = [{"hp_id": "hp1", "atom_id": "a1", "name": "cutoff", "kind": "float"}]
    rollups = [{"atom_id": "a1", "overall_verdict": "trusted"}]
    descriptions = [
        {
            "description_id": "d1",
            "atom_id": "a1",
            "kind": "dejargonized",
            "content": "Filter signal",
            "language": "en",
        }
    ]
    io_specs = [
        {
            "io_spec_id": "ios1",
            "atom_id": "a1",
            "name": "signal",
            "direction": "input",
            "type_desc": "np.ndarray",
            "constraints": "1D waveform",
            "data_kind": "signal",
            "required": True,
            "default_value_repr": "",
            "ordinal": 0,
        }
    ]
    benchmarks = [
        {
            "atom_fqdn": "pkg.filter",
            "content_hash": "abc123",
            "benchmark_name": "signal_v1",
            "metric_name": "loss",
            "metric_value": 0.42,
            "dataset_tag": "v1",
            "measured_at": "2026-03-31T00:00:00Z",
        }
    ]

    async def fake_fetch_all_rows(base_url, token, table, **kwargs):
        if table == "atoms":
            return atoms
        if table == "hyperparams":
            return hyperparams
        if table == "atom_audit_rollups":
            return rollups
        if table == "atom_descriptions":
            return descriptions
        if table == "atom_io_specs":
            return io_specs
        if table == "atom_benchmarks":
            return []
        raise AssertionError(f"unexpected table {table!r}")

    async def fake_call_rpc(base_url, token, rpc_name, payload=None, **kwargs):
        assert rpc_name == "get_manifest_benchmarks"
        return benchmarks

    monkeypatch.setattr("sciona.api.snapshot._fetch_all_rows", fake_fetch_all_rows)
    monkeypatch.setattr("sciona.api.snapshot._call_rpc", fake_call_rpc)

    data = await fetch_manifest_data("https://example.supabase.co", "token")

    assert set(data) == {
        "atoms",
        "hyperparams",
        "benchmarks",
        "rollups",
        "descriptions",
        "io_specs",
    }
    assert data["benchmarks"][0]["benchmark_id"] == "signal_v1"
    assert data["benchmarks"][0]["benchmark_name"] == "signal_v1"
    assert data["io_specs"][0]["port_name"] == "signal"


@pytest.mark.asyncio
async def test_fetch_manifest_data_applies_visibility_tier_filter(monkeypatch):
    captured_filters: list[dict[str, str]] = []

    async def fake_fetch_all_rows(base_url, token, table, **kwargs):
        del base_url, token
        if table == "atoms":
            captured_filters.append(dict(kwargs.get("filters") or {}))
            return []
        return []

    monkeypatch.setattr("sciona.api.snapshot._fetch_all_rows", fake_fetch_all_rows)
    monkeypatch.setattr(
        "sciona.api.snapshot._call_rpc",
        lambda *args, **kwargs: [],  # pragma: no cover - not reached
    )

    data = await fetch_manifest_data(
        "https://example.supabase.co",
        "token",
        visibility_tiers=["general", "early_access"],
    )

    assert data["atoms"] == []
    assert captured_filters[0]["visibility_tier"] == 'in.("general","early_access")'


@pytest.mark.asyncio
async def test_fetch_manifest_data_falls_back_to_atom_benchmarks(monkeypatch):
    atoms = [
        {
            "atom_id": "a1",
            "fqdn": "pkg.filter",
            "status": "approved",
            "domain_tags": ["signal"],
            "description": "Filter signal",
            "visibility_tier": "general",
            "source_kind": "hand_written",
            "stateful_kind": "none",
            "is_stochastic": False,
            "is_ffi": False,
            "namespace_root": "sciona.atoms",
            "namespace_path": "",
            "source_repo_id": None,
            "source_package": "",
            "source_module_path": "",
            "source_symbol": "",
            "is_publishable": True,
        }
    ]
    version_rows = [{"version_id": "v1", "atom_id": "a1", "content_hash": "abc123"}]
    benchmark_rows = [
        {
            "benchmark_id": "bench-1",
            "version_id": "v1",
            "benchmark_name": "signal_v1",
            "metric_name": "loss",
            "metric_value": 0.42,
            "dataset_tag": "v1",
            "measured_at": "2026-03-31T00:00:00Z",
        }
    ]

    async def fake_fetch_all_rows(base_url, token, table, **kwargs):
        del base_url, token, kwargs
        if table == "atoms":
            return atoms
        if table == "hyperparams":
            return []
        if table == "atom_audit_rollups":
            return []
        if table == "atom_descriptions":
            return []
        if table == "atom_io_specs":
            return []
        if table == "atom_benchmarks":
            return benchmark_rows
        if table == "atom_versions":
            return version_rows
        raise AssertionError(f"unexpected table {table!r}")

    async def fake_call_rpc(*args, **kwargs):
        request = httpx.Request(
            "POST",
            "https://example.supabase.co/rest/v1/rpc/get_manifest_benchmarks",
        )
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError("missing", request=request, response=response)

    monkeypatch.setattr("sciona.api.snapshot._fetch_all_rows", fake_fetch_all_rows)
    monkeypatch.setattr("sciona.api.snapshot._call_rpc", fake_call_rpc)

    data = await fetch_manifest_data("https://example.supabase.co", "token")

    assert data["benchmarks"][0]["atom_fqdn"] == "pkg.filter"
    assert data["benchmarks"][0]["content_hash"] == "abc123"
    assert data["benchmarks"][0]["benchmark_id"] == "bench-1"
    assert data["benchmarks"][0]["benchmark_name"] == "signal_v1"


@pytest.mark.asyncio
async def test_fetch_manifest_data_filters_related_tables_client_side(monkeypatch):
    atoms = [
        {
            "atom_id": f"a{i}",
            "fqdn": f"pkg.atom_{i}",
            "status": "approved",
            "domain_tags": ["signal"],
            "description": f"Atom {i}",
            "visibility_tier": "general",
            "source_kind": "hand_written",
            "stateful_kind": "none",
            "is_stochastic": False,
            "is_ffi": False,
            "namespace_root": "sciona.atoms",
            "namespace_path": "",
            "source_repo_id": None,
            "source_package": "",
            "source_module_path": "",
            "source_symbol": "",
            "is_publishable": True,
        }
        for i in range(DEFAULT_FILTER_BATCH_SIZE + 5)
    ]
    related_filters: dict[str, dict[str, str]] = {}

    async def fake_fetch_all_rows(base_url, token, table, **kwargs):
        del base_url, token
        if table == "atoms":
            return atoms
        if table in {
            "hyperparams",
            "atom_audit_rollups",
            "atom_descriptions",
            "atom_io_specs",
        }:
            related_filters[table] = dict(kwargs.get("filters") or {})
            return []
        if table in {"atom_benchmarks", "atom_versions"}:
            return []
        raise AssertionError(f"unexpected table {table!r}")

    async def fake_call_rpc(*args, **kwargs):
        request = httpx.Request(
            "POST",
            "https://example.supabase.co/rest/v1/rpc/get_manifest_benchmarks",
        )
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError("missing", request=request, response=response)

    monkeypatch.setattr("sciona.api.snapshot._fetch_all_rows", fake_fetch_all_rows)
    monkeypatch.setattr("sciona.api.snapshot._call_rpc", fake_call_rpc)

    await fetch_manifest_data("https://example.supabase.co", "token")

    assert related_filters["hyperparams"] == {"status": "eq.approved"}
    assert related_filters["atom_audit_rollups"] == {}
    assert related_filters["atom_descriptions"] == {
        "kind": "eq.dejargonized",
        "language": "eq.en",
    }
    assert related_filters["atom_io_specs"] == {}


def test_generate_manifest_sqlite_preserves_benchmark_id(tmp_path: Path):
    data = {
        "atoms": [
            {
                "atom_id": "a1",
                "fqdn": "pkg.filter",
                "status": "approved",
                "domain_tags": ["signal", "audio"],
                "description": "Filter signal",
                "is_publishable": True,
            }
        ],
        "hyperparams": [
            {
                "hp_id": "hp1",
                "atom_id": "a1",
                "name": "cutoff",
                "kind": "float",
                "default_value": "0.5",
                "status": "approved",
            }
        ],
        "benchmarks": [
            {
                "atom_fqdn": "pkg.filter",
                "content_hash": "abc123",
                "benchmark_name": "signal_v1",
                "metric_name": "loss",
                "metric_value": 0.42,
                "dataset_tag": "v1",
                "measured_at": "2026-03-31T00:00:00Z",
            }
        ],
        "io_specs": [
            {
                "atom_id": "a1",
                "port_name": "signal",
                "direction": "input",
                "type_desc": "np.ndarray",
                "constraints": "1D waveform",
                "data_kind": "signal",
                "required": True,
                "default_value_repr": "",
                "ordinal": 0,
            }
        ],
        "rollups": [
            {
                "atom_id": "a1",
                "overall_verdict": "trusted",
                "risk_tier": "low",
            }
        ],
        "descriptions": [
            {
                "description_id": "d1",
                "atom_id": "a1",
                "kind": "dejargonized",
                "content": "Filter signal",
                "language": "en",
            }
        ],
    }
    output = tmp_path / "manifest.sqlite"

    con = generate_manifest_sqlite(data, output_path=output)
    con.close()

    rows = load_benchmarks_sqlite(output)["pkg.filter"]
    assert rows[0].benchmark_id == "signal_v1"
    assert rows[0].metric_name == "loss"

    verify = sqlite3.connect(str(output))
    try:
        tables = {
            row[0]
            for row in verify.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {
            "atoms",
            "hyperparams",
            "benchmarks",
            "audit_rollups",
            "descriptions",
            "io_specs",
            "manifest_metadata",
        }.issubset(tables)
        row = verify.execute(
            "SELECT benchmark_id, benchmark_name FROM benchmarks"
        ).fetchone()
        assert row == ("signal_v1", "signal_v1")
        metadata = dict(
            verify.execute("SELECT key, value FROM manifest_metadata").fetchall()
        )
        assert metadata["visibility_tier"] == "all"
    finally:
        verify.close()


@pytest.mark.asyncio
async def test_export_tiered_manifests_writes_one_file_per_tier(monkeypatch, tmp_path: Path):
    async def fake_fetch_manifest_data(base_url, token, **kwargs):
        del base_url, token
        visibility_tiers = tuple(kwargs.get("visibility_tiers") or ())
        suffix = visibility_tiers[-1] if visibility_tiers else "none"
        return {
            "atoms": [
                {
                    "atom_id": f"a-{suffix}",
                    "fqdn": f"pkg.{suffix}",
                    "status": "approved",
                }
            ],
            "hyperparams": [],
            "benchmarks": [],
            "rollups": [],
            "descriptions": [],
            "io_specs": [],
        }

    monkeypatch.setattr("sciona.api.snapshot.fetch_manifest_data", fake_fetch_manifest_data)

    outputs = await export_tiered_manifests(
        "https://example.supabase.co",
        "token",
        tmp_path,
    )

    assert set(outputs) == {"general", "early_access", "internal"}
    for tier, path in outputs.items():
        assert path.exists()
        con = sqlite3.connect(str(path))
        try:
            metadata = dict(con.execute("SELECT key, value FROM manifest_metadata").fetchall())
            assert metadata["visibility_tier"] == tier
        finally:
            con.close()


@pytest.mark.asyncio
async def test_catalog_sync_downloads_manifest_sqlite(monkeypatch, tmp_path: Path, capsys):
    output = tmp_path / "manifest.sqlite"
    source = tmp_path / "source.sqlite"
    con = generate_manifest_sqlite(
        {
            "atoms": [{"atom_id": "a1", "fqdn": "pkg.filter", "status": "approved"}],
            "hyperparams": [],
            "benchmarks": [
                {
                    "atom_fqdn": "pkg.filter",
                    "content_hash": "abc123",
                    "benchmark_name": "signal_v1",
                    "metric_name": "loss",
                    "metric_value": 0.42,
                    "dataset_tag": "v1",
                    "measured_at": "2026-03-31T00:00:00Z",
                }
            ],
            "rollups": [],
            "descriptions": [],
        },
        output_path=source,
    )
    con.close()

    captured_url: dict[str, str] = {}

    async def fake_download_manifest_bytes(manifest_url: str) -> bytes:
        captured_url["value"] = manifest_url
        return source.read_bytes()

    monkeypatch.setattr(
        "sciona.commands.catalog_cmds._download_manifest_bytes",
        fake_download_manifest_bytes,
    )

    await _cmd_catalog_sync(
        argparse.Namespace(
            output=str(output),
            api_url=None,
            manifest_url="https://bucket.example/manifests/manifest.sqlite",
            tier=None,
        )
    )

    assert captured_url["value"] == "https://bucket.example/manifests/manifest.sqlite"
    assert output.exists()
    assert load_benchmarks_sqlite(output)["pkg.filter"][0].benchmark_id == "signal_v1"

    captured = capsys.readouterr()
    assert "Manifest written to" in captured.out


def test_resolve_manifest_url_uses_tier(monkeypatch):
    monkeypatch.delenv("SCIONA_MANIFEST_URL", raising=False)
    monkeypatch.delenv("SCIONA_MANIFEST_KEY", raising=False)
    monkeypatch.setenv("SCIONA_CATALOG_BUCKET", "bucket.example")

    url = _resolve_manifest_url(
        argparse.Namespace(manifest_url=None, tier="internal", output=None, api_url=None)
    )

    assert url == "https://bucket.example.s3.amazonaws.com/manifests/manifest-internal.sqlite"


class _FakeSupabase:
    def __init__(self, payloads: dict[str, object]):
        self.payloads = payloads

    def rpc(self, name: str, payload: dict[str, str]):
        del payload
        return SimpleNamespace(execute=lambda: self._execute(name))

    async def _execute(self, name: str):
        return SimpleNamespace(data=self.payloads.get(name))


@pytest.mark.asyncio
async def test_get_atom_document_returns_rpc_payload():
    payload = {"atom": {"fqdn": "pkg.filter"}}
    result = await get_atom_document(
        "pkg.filter",
        supabase=_FakeSupabase({"get_atom_document": payload}),
    )
    assert result == payload


@pytest.mark.asyncio
async def test_get_atom_document_raises_for_missing_atom():
    with pytest.raises(HTTPException) as excinfo:
        await get_atom_document(
            "pkg.missing",
            supabase=_FakeSupabase({"get_atom_document": None}),
        )
    assert excinfo.value.status_code == 404


@pytest.mark.asyncio
async def test_get_artifact_document_falls_back_to_atom_document():
    payload = {"atom": {"fqdn": "pkg.filter"}}
    result = await get_artifact_document(
        "pkg.filter",
        supabase=_FakeSupabase(
            {"get_artifact_document": None, "get_atom_document": payload}
        ),
    )
    assert result == payload
