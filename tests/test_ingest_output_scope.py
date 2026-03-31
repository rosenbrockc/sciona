"""Tests for grouped ingest output scope metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import sciona.cli as cli_module
from sciona.commands.ingest_cmds import _cmd_ingest, _resolve_output_scope


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
    def __init__(self) -> None:
        self.generated_atoms = "def grouped_atom():\n    return 'ok'\n"
        self.generated_state_models = ""
        self.generated_witnesses = "def witness_grouped_atom():\n    return True\n"
        self.cdg = _FakeCDG()
        self.match_results: list[object] = []
        self.mypy_passed = True
        self.ghost_sim_passed = True


class _FakeAgent:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    async def ingest_state(self, source_path: str, class_name: str) -> dict[str, object]:
        return {"bundle": _FakeBundle(), "error": ""}


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
    monkeypatch.setattr("sciona.commands.ingest_cmds._create_llm_router", lambda *args, **kwargs: object())
    monkeypatch.setattr("sciona.commands.ingest_cmds._warm_llm_if_supported", _noop_warm)
    monkeypatch.setattr("sciona.commands.ingest_cmds._create_proof_env", lambda *args, **kwargs: _FakeEnv())
    monkeypatch.setattr("sciona.commands.ingest_cmds._create_shared_context", _fake_shared_context)
    monkeypatch.setattr("sciona.commands.ingest_cmds._print_mode_summary", lambda *args, **kwargs: None)
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


def test_resolve_output_scope_prefers_explicit_argument(tmp_path: Path):
    args = argparse.Namespace(
        class_name="PatchExtractor",
        output=str(tmp_path / "images"),
        output_scope="family",
    )

    scope, source = _resolve_output_scope(args, output_dir=tmp_path / "images")

    assert scope == "family"
    assert source == "argument"


def test_cli_ingest_parser_accepts_output_scope(monkeypatch):
    captured: list[object] = []

    monkeypatch.setattr(
        cli_module,
        "_run_async_command",
        lambda payload: captured.append(payload),
    )
    monkeypatch.setattr(cli_module, "_cmd_ingest", lambda args: args)
    monkeypatch.setattr(
        "sys.argv",
        [
            "sciona",
            "ingest",
            "source.py",
            "--class",
            "PatchExtractor",
            "--output-scope",
            "family",
        ],
    )

    cli_module.main()

    assert len(captured) == 1
    assert getattr(captured[0], "output_scope") == "family"


def test_cli_ingest_parser_accepts_allow_family_replace(monkeypatch):
    captured: list[object] = []

    monkeypatch.setattr(
        cli_module,
        "_run_async_command",
        lambda payload: captured.append(payload),
    )
    monkeypatch.setattr(cli_module, "_cmd_ingest", lambda args: args)
    monkeypatch.setattr(
        "sys.argv",
        [
            "sciona",
            "ingest",
            "source.py",
            "--class",
            "PatchExtractor",
            "--output-scope",
            "family",
            "--allow-family-replace",
        ],
    )

    cli_module.main()

    assert len(captured) == 1
    assert getattr(captured[0], "allow_family_replace") is True


@pytest.mark.asyncio
async def test_cmd_ingest_records_grouped_output_scope_for_non_symbol_dir(
    monkeypatch,
    tmp_path: Path,
):
    source_path = tmp_path / "image_source.py"
    source_path.write_text(
        "class PatchExtractor:\n"
        "    def fit(self, X, y=None):\n"
        "        return self\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "sklearn" / "images"
    _patch_ingest_runtime(monkeypatch, tmp_path)

    await _cmd_ingest(
        argparse.Namespace(
            source=str(source_path),
            class_name="PatchExtractor",
            output=str(output_dir),
            output_scope="family",
            allow_family_replace=False,
            llm_provider=None,
            llm_model=None,
            procedural=False,
            trace=False,
            monitor=False,
            stale_seconds=120,
            mode=None,
        )
    )

    status = json.loads((output_dir / ".ingest_status.json").read_text(encoding="utf-8"))
    completed = json.loads((output_dir / "COMPLETED.json").read_text(encoding="utf-8"))

    assert status["output_scope"] == "family"
    assert status["output_scope_source"] == "argument"
    assert status["publication"]["target_basename"] == "images"
    assert status["publication"]["existing_family_output"] is False
    assert status["publication"]["existing_family_artifacts"] == []
    assert status["publication"]["allow_family_replace"] is False
    assert status["publication"]["family_publication_mode"] == "fresh_family_publish"
    assert status["publication"]["published_files"] == [
        "atoms.py",
        "cdg.json",
        "witnesses.py",
    ]
    assert completed["output_scope"] == "family"
    assert completed["summary"]["output_dir"] == str(output_dir)
    assert completed["summary"]["allow_family_replace"] is False
    assert completed["summary"]["publication"]["family_publication_mode"] == "fresh_family_publish"
    assert completed["summary"]["publication"]["missing_artifacts"] == [
        "state_models.py",
        "matches.json",
    ]
    assert (output_dir / "atoms.py").exists()
    assert (output_dir / "witnesses.py").exists()
    assert (output_dir / "cdg.json").exists()


@pytest.mark.asyncio
async def test_cmd_ingest_grouped_family_replace_fails_by_default(
    monkeypatch,
    tmp_path: Path,
):
    source_path = tmp_path / "offset_source.py"
    source_path.write_text("def tt2tdb_offset(seconds):\n    return seconds\n", encoding="utf-8")
    output_dir = tmp_path / "tempo_jl" / "offsets"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "atoms.py").write_text("def old_grouped_atom():\n    return 'old'\n", encoding="utf-8")
    _patch_ingest_runtime(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        await _cmd_ingest(
            argparse.Namespace(
                source=str(source_path),
                class_name="tt2tdb_offset",
                output=str(output_dir),
                output_scope="family",
                allow_family_replace=False,
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
    assert status["phase"] == "phase0_family_replace_guard"
    assert status["publication"]["existing_family_output"] is True
    assert status["publication"]["existing_family_artifacts"] == ["atoms.py"]
    assert status["publication"]["allow_family_replace"] is False
    assert status["publication"]["family_publication_mode"] == "family_replace"
    assert "--allow-family-replace" in failed["error"]
    assert failed["summary"]["published_files"] == []
    assert failed["summary"]["publication"]["existing_family_artifacts"] == ["atoms.py"]
    assert (output_dir / "atoms.py").read_text(encoding="utf-8") == (
        "def old_grouped_atom():\n    return 'old'\n"
    )


@pytest.mark.asyncio
async def test_cmd_ingest_grouped_family_replace_succeeds_with_explicit_flag(
    monkeypatch,
    tmp_path: Path,
):
    source_path = tmp_path / "offset_source.py"
    source_path.write_text("def tt2tdb_offset(seconds):\n    return seconds\n", encoding="utf-8")
    output_dir = tmp_path / "tempo_jl" / "offsets"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "atoms.py").write_text("def old_grouped_atom():\n    return 'old'\n", encoding="utf-8")
    _patch_ingest_runtime(monkeypatch, tmp_path)

    await _cmd_ingest(
        argparse.Namespace(
            source=str(source_path),
            class_name="tt2tdb_offset",
            output=str(output_dir),
            output_scope="family",
            allow_family_replace=True,
            llm_provider=None,
            llm_model=None,
            procedural=False,
            trace=False,
            monitor=False,
            stale_seconds=120,
            mode=None,
        )
    )

    status = json.loads((output_dir / ".ingest_status.json").read_text(encoding="utf-8"))
    completed = json.loads((output_dir / "COMPLETED.json").read_text(encoding="utf-8"))

    assert status["state"] == "completed"
    assert status["publication"]["existing_family_output"] is True
    assert status["publication"]["existing_family_artifacts"] == ["atoms.py"]
    assert status["publication"]["allow_family_replace"] is True
    assert status["publication"]["family_publication_mode"] == "family_replace"
    assert completed["summary"]["allow_family_replace"] is True
    assert completed["summary"]["publication"]["existing_family_artifacts"] == ["atoms.py"]
    assert completed["summary"]["publication"]["family_publication_mode"] == "family_replace"
    assert (output_dir / "atoms.py").read_text(encoding="utf-8") == (
        "def grouped_atom():\n    return 'ok'\n"
    )
