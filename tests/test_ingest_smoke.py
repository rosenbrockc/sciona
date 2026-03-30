"""Tests for ingest-time deterministic smoke validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sciona.commands.ingest_cmds import _cmd_ingest


class _FakeEnv:
    async def close(self) -> None:
        return None


class _FakeCDG:
    def __init__(self) -> None:
        self.nodes = [{"id": "root"}]
        self.edges = []

    def model_dump(self) -> dict[str, object]:
        return {"nodes": self.nodes, "edges": self.edges}


class _FakeBundle:
    def __init__(self, *, generated_atoms: str) -> None:
        self.generated_atoms = generated_atoms
        self.generated_state_models = ""
        self.generated_witnesses = ""
        self.cdg = _FakeCDG()
        self.match_results: list[object] = []
        self.mypy_passed = True
        self.ghost_sim_passed = True


class _FakeAgent:
    bundle: _FakeBundle | None = None

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    async def ingest_state(self, source_path: str, class_name: str) -> dict[str, object]:
        assert self.bundle is not None
        return {"bundle": self.bundle, "error": ""}


def _patch_ingest_runtime(monkeypatch, tmp_path: Path) -> None:
    config = SimpleNamespace(
        ingester_llm_provider="tests",
        llm_provider="tests",
        ingester_llm_model="fixture",
        llm_model="fixture",
        ingester_max_depth=2,
        index_dir=tmp_path / "missing_index",
        ingester_decompose_line_threshold=30,
        ingester_shared_context_budget_chars=256,
        ingester_parallelism=1,
        ingester_cache_enabled=False,
        ingester_cache_dir=tmp_path / "cache",
    )
    mode_settings = SimpleNamespace(
        semantic_index_backend_override=None,
        ingester_shared_context_enabled=False,
    )

    async def _noop_warm(*args, **kwargs):
        return None

    async def _fake_shared_context(*args, **kwargs):
        return None, None

    monkeypatch.setattr("sciona.config.AgeomConfig", lambda: config)
    monkeypatch.setattr(
        "sciona.config.resolve_execution_mode",
        lambda _config, _mode: mode_settings,
    )
    monkeypatch.setattr("sciona.ingester.IngesterAgent", _FakeAgent)
    monkeypatch.setattr(
        "sciona.commands.ingest_cmds._create_llm_router",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "sciona.commands.ingest_cmds._warm_llm_if_supported",
        _noop_warm,
    )
    monkeypatch.setattr(
        "sciona.commands.ingest_cmds._create_proof_env",
        lambda *args, **kwargs: _FakeEnv(),
    )
    monkeypatch.setattr(
        "sciona.commands.ingest_cmds._create_shared_context",
        _fake_shared_context,
    )
    monkeypatch.setattr(
        "sciona.commands.ingest_cmds._print_mode_summary",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "sciona.commands.ingest_cmds._print_prompt_routing_summary",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "sciona.commands.ingest_cmds._print_shared_context_metrics",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "sciona.commands.ingest_cmds._write_shared_context_metrics_file",
        lambda *args, **kwargs: None,
    )


async def _run_ingest(
    monkeypatch,
    tmp_path: Path,
    *,
    class_name: str,
    generated_atoms: str,
):
    _patch_ingest_runtime(monkeypatch, tmp_path)
    _FakeAgent.bundle = _FakeBundle(generated_atoms=generated_atoms)
    source_path = tmp_path / "source.py"
    source_path.write_text("def stub():\n    return None\n", encoding="utf-8")
    output_dir = tmp_path / "sklearn" / "images"
    args = argparse.Namespace(
        source=str(source_path),
        class_name=class_name,
        output=str(output_dir),
        output_scope="family",
        llm_provider=None,
        llm_model=None,
        procedural=False,
        trace=False,
        monitor=False,
        stale_seconds=120,
        mode=None,
    )
    await _cmd_ingest(args)
    return output_dir


@pytest.mark.asyncio
async def test_smoke_validation_not_applicable_allows_publication(monkeypatch, tmp_path: Path):
    output_dir = await _run_ingest(
        monkeypatch,
        tmp_path,
        class_name="ungrouped_atom",
        generated_atoms=(
            "def ungrouped_atom(x):\n"
            "    return x\n"
        ),
    )

    status = json.loads((output_dir / ".ingest_status.json").read_text(encoding="utf-8"))
    completed = json.loads((output_dir / "COMPLETED.json").read_text(encoding="utf-8"))

    assert status["smoke_validation"]["status"] == "not_applicable"
    assert completed["summary"]["smoke_validation"]["status"] == "not_applicable"
    assert (output_dir / "atoms.py").exists()


@pytest.mark.asyncio
async def test_smoke_validation_pass_allows_publication(monkeypatch, tmp_path: Path):
    output_dir = await _run_ingest(
        monkeypatch,
        tmp_path,
        class_name="safe_grouped_atom",
        generated_atoms=(
            "def safe_grouped_atom(x):\n"
            "    if not isinstance(x, int):\n"
            "        raise TypeError('x')\n"
            "    return x + 1\n"
        ),
    )

    status = json.loads((output_dir / ".ingest_status.json").read_text(encoding="utf-8"))
    completed = json.loads((output_dir / "COMPLETED.json").read_text(encoding="utf-8"))

    assert status["smoke_validation"]["status"] == "pass"
    assert completed["summary"]["smoke_validation"]["status"] == "pass"
    assert (output_dir / "atoms.py").exists()


@pytest.mark.asyncio
async def test_smoke_validation_fail_blocks_publication(monkeypatch, tmp_path: Path):
    _patch_ingest_runtime(monkeypatch, tmp_path)
    _FakeAgent.bundle = _FakeBundle(
        generated_atoms=(
            "def safe_grouped_atom(x):\n"
            "    return x\n"
        )
    )
    source_path = tmp_path / "source.py"
    source_path.write_text("def stub():\n    return None\n", encoding="utf-8")
    output_dir = tmp_path / "sklearn" / "images"

    with pytest.raises(SystemExit) as excinfo:
        await _cmd_ingest(
            argparse.Namespace(
                source=str(source_path),
                class_name="safe_grouped_atom",
                output=str(output_dir),
                output_scope="family",
                llm_provider=None,
                llm_model=None,
                procedural=False,
                trace=False,
                monitor=False,
                stale_seconds=120,
                mode=None,
            )
        )

    assert excinfo.value.code == 1
    status = json.loads((output_dir / ".ingest_status.json").read_text(encoding="utf-8"))
    failed = json.loads((output_dir / "FAILED.json").read_text(encoding="utf-8"))

    assert status["state"] == "failed"
    assert status["smoke_validation"]["status"] == "fail"
    assert failed["summary"]["smoke_validation"]["status"] == "fail"
    assert failed["summary"]["published_files"] == []
    assert not (output_dir / "atoms.py").exists()
