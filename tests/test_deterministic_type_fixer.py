from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sciona.architect.handoff import CDGExport
from sciona.commands._helpers import _create_llm_router
from sciona.ingester.deterministic_type_fixer import (
    DeterministicTypeFixer,
    _extract_line_number,
    _parse_fix_type_prompt,
)
from sciona.ingester.graph import IngesterDeps, repair_types, route_after_type_check, verify_types
from sciona.ingester.models import IngestionBundle
from sciona.ingester.monitor import IngestMonitor
from sciona.llm_router import INGESTER_FIX_TYPE, LLMRouter


def _prompt(errors: str, source: str, *, filename: str = "atoms.py") -> str:
    return (
        "mypy errors:\n"
        f"{errors}\n\n"
        "Generated files:\n"
        f"<<FILE: {filename}>>\n"
        "```python\n"
        f"{source}\n"
        "```\n"
        "<<END FILE>>\n\n"
        "Return JSON array of fixes:\n[]"
    )


def test_parse_fix_type_prompt_extracts_sections():
    errors, files = _parse_fix_type_prompt(
        _prompt('_check.py:2: error: Name "np" is not defined', "def f():\n    return np.array([1])")
    )

    assert 'Name "np" is not defined' in errors
    assert "return np.array" in files["atoms.py"]


def test_extract_line_number_handles_standard_mypy_format():
    assert _extract_line_number('_check.py:42: error: Name "np" is not defined') == 42
    assert _extract_line_number("not a mypy line") is None


@pytest.mark.asyncio
async def test_deterministic_type_fixer_inserts_missing_import():
    fallback = AsyncMock()
    fixer = DeterministicTypeFixer(fallback)
    response = await fixer.complete(
        "sys",
        _prompt(
            '_check.py:2: error: Name "np" is not defined',
            "def f(xs):\n    return np.array(xs)",
        ),
    )

    patches = json.loads(response)
    assert patches == [
        {
            "file": "atoms.py",
            "line_start": 1,
            "line_end": 1,
            "replacement": "def f(xs):\nimport numpy as np",
        }
    ]
    fallback.complete.assert_not_called()


@pytest.mark.asyncio
async def test_deterministic_type_fixer_wraps_return_value():
    fallback = AsyncMock()
    fixer = DeterministicTypeFixer(fallback)
    response = await fixer.complete(
        "sys",
        _prompt(
            '_check.py:2: error: Incompatible return value type (got "str", expected "int")',
            'def f() -> int:\n    return "1"',
        ),
    )

    patches = json.loads(response)
    assert patches == [
        {
            "file": "atoms.py",
            "line_start": 2,
            "line_end": 2,
            'replacement': '    return int("1")',
        }
    ]
    fallback.complete.assert_not_called()


@pytest.mark.asyncio
async def test_deterministic_type_fixer_falls_back_for_unknown_assignment_rewrite():
    fallback = AsyncMock()
    fallback.complete.return_value = "fallback"
    fixer = DeterministicTypeFixer(fallback)
    response = await fixer.complete(
        "sys",
        _prompt(
            "_check.py:2: error: Incompatible types in assignment",
            "x = 1\nx = '1'",
        ),
    )

    assert response == "fallback"
    fallback.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_deterministic_type_fixer_targets_matching_bundle_file():
    fallback = AsyncMock()
    fixer = DeterministicTypeFixer(fallback)
    response = await fixer.complete(
        "sys",
        _prompt(
            'state_models.py:2: error: Name "Literal" is not defined',
            "from typing import Any\nx: Literal['a'] | None = None",
            filename="state_models.py",
        ),
    )

    patches = json.loads(response)
    assert patches == [
        {
            "file": "state_models.py",
            "line_start": 1,
            "line_end": 1,
            "replacement": "from typing import Any\nfrom typing import Literal",
        }
    ]
    fallback.complete.assert_not_called()


@pytest.mark.asyncio
async def test_repair_types_uses_deterministic_type_fixer_patch():
    fallback = AsyncMock()
    fallback.complete.side_effect = AssertionError("fallback should not be used")
    fixer = DeterministicTypeFixer(fallback)
    llm = LLMRouter(default=fallback, overrides={INGESTER_FIX_TYPE: fixer})
    bundle = IngestionBundle(
        cdg=CDGExport(nodes=[], edges=[]),
        generated_atoms='def f() -> int:\n    return "1"\n',
    )
    state = {
        "bundle": bundle,
        "mypy_errors": '_check.py:2: error: Incompatible return value type (got "str", expected "int")',
        "type_repair_count": 0,
    }
    config = {"configurable": {"deps": IngesterDeps(llm=llm)}}

    result = await repair_types(state, config)

    assert result["type_repair_count"] == 1
    assert 'return int("1")' in result["bundle"].generated_atoms


@pytest.mark.asyncio
async def test_repair_types_applies_state_model_patch():
    fallback = AsyncMock()
    fallback.complete.side_effect = AssertionError("fallback should not be used")
    fixer = DeterministicTypeFixer(fallback)
    llm = LLMRouter(default=fallback, overrides={INGESTER_FIX_TYPE: fixer})
    bundle = IngestionBundle(
        cdg=CDGExport(nodes=[], edges=[]),
        generated_atoms="def f() -> None:\n    return None\n",
        generated_state_models="from typing import Any\nx: Literal['a'] | None = None\n",
    )
    state = {
        "bundle": bundle,
        "mypy_errors": 'state_models.py:2: error: Name "Literal" is not defined',
        "type_repair_count": 0,
    }
    config = {"configurable": {"deps": IngesterDeps(llm=llm)}}

    result = await repair_types(state, config)

    assert result["type_repair_count"] == 1
    assert "from typing import Literal" in result["bundle"].generated_state_models
    assert result["bundle"].generated_atoms == bundle.generated_atoms


@pytest.mark.asyncio
async def test_verify_types_prefers_bundle_checker_when_available():
    class _ProofEnv:
        def __init__(self) -> None:
            self.calls: list[tuple[dict[str, str], dict[str, object]]] = []

        async def check_generated_files(self, bundle_files, **kwargs):
            self.calls.append((bundle_files, kwargs))
            return (False, "state_models.py:2: error: boom")

    proof_env = _ProofEnv()
    bundle = IngestionBundle(
        cdg=CDGExport(nodes=[], edges=[]),
        generated_atoms="def f() -> None:\n    return None\n",
        generated_state_models="class S:\n    pass\n",
    )
    state = {"bundle": bundle}
    config = {"configurable": {"deps": IngesterDeps(llm=AsyncMock(), proof_env=proof_env)}}

    result = await verify_types(state, config)

    assert result["mypy_passed"] is False
    assert result["mypy_errors"] == "state_models.py:2: error: boom"
    assert result["type_failure_classification"]["reason_code"] == "unknown_or_unclassified"
    assert result["type_failure_classification"]["repairable"] is False
    assert len(proof_env.calls) == 1
    bundle_files = proof_env.calls[0][0]
    assert bundle_files["atoms.py"] == bundle.generated_atoms
    assert bundle_files["state_models.py"] == bundle.generated_state_models


@pytest.mark.asyncio
async def test_verify_types_marks_bundle_on_success():
    class _ProofEnv:
        async def check_generated_files(self, bundle_files, **kwargs):
            return (True, "")

    proof_env = _ProofEnv()
    bundle = IngestionBundle(
        cdg=CDGExport(nodes=[], edges=[]),
        generated_atoms="def f() -> None:\n    return None\n",
    )
    state = {"bundle": bundle}
    config = {"configurable": {"deps": IngesterDeps(llm=AsyncMock(), proof_env=proof_env)}}

    result = await verify_types(state, config)

    assert result["mypy_passed"] is True
    assert result["bundle"].mypy_passed is True


@pytest.mark.asyncio
async def test_repair_types_fails_fast_for_semantic_signature_error():
    bundle = IngestionBundle(
        cdg=CDGExport(nodes=[], edges=[]),
        generated_atoms="def f() -> None:\n    return None\n",
    )
    state = {
        "bundle": bundle,
        "mypy_errors": 'atoms.py:1: error: Too many arguments for "f"',
        "type_repair_count": 0,
    }
    config = {"configurable": {"deps": IngesterDeps(llm=AsyncMock())}}

    result = await repair_types(state, config)

    assert result == {}


def test_route_after_type_check_ends_for_non_repairable_semantic_failure():
    bundle = IngestionBundle(
        cdg=CDGExport(nodes=[], edges=[]),
        generated_atoms="def f() -> None:\n    return None\n",
    )
    state = {
        "bundle": bundle,
        "mypy_passed": False,
        "mypy_errors": 'atoms.py:1: error: Too many arguments for "f"',
        "type_repair_count": 0,
    }

    assert route_after_type_check(state) == "end"


@pytest.mark.asyncio
async def test_verify_types_publishes_failure_classification_artifact(tmp_path):
    class _ProofEnv:
        async def check_generated_files(self, bundle_files, **kwargs):
            return (False, 'atoms.py:1: error: Too many arguments for "f"')

    proof_env = _ProofEnv()
    monitor = IngestMonitor(tmp_path)
    bundle = IngestionBundle(
        cdg=CDGExport(nodes=[], edges=[]),
        generated_atoms="def f() -> None:\n    return None\n",
    )
    state = {
        "bundle": bundle,
        "mypy_passed": False,
        "ghost_passed": False,
        "mypy_errors": "",
        "ghost_errors": "",
        "type_repair_count": 0,
        "ghost_repair_count": 0,
        "type_failure_classification": {},
        "ghost_failure_classification": {},
    }
    config = {
        "configurable": {
            "deps": IngesterDeps(llm=AsyncMock(), proof_env=proof_env, monitor=monitor)
        }
    }

    await verify_types(state, config)

    payload = json.loads((tmp_path / "verification_failure.json").read_text())
    assert payload["stage"] == "verify_types"
    assert payload["reason_code"] == "semantic_signature"
    assert payload["repairable"] is False


def test_create_llm_router_wraps_ingester_fix_type_deterministically(monkeypatch):
    created: list[tuple[str, str]] = []

    def _fake_create_llm_client(*, provider, model, **kwargs):
        created.append((provider, model))
        client = AsyncMock()
        client.provider = provider
        client.model = model
        return client

    monkeypatch.setattr("sciona.hunter.llm.create_llm_client", _fake_create_llm_client)

    args = SimpleNamespace(llm_provider=None, llm_model=None, llm_max_tokens=None)
    config = SimpleNamespace(
        execution_mode="verified",
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-5-20250929",
        llm_max_tokens=4096,
        anthropic_api_key="",
        openai_api_key="",
        openai_base_url="",
        llama_cpp_base_url="http://127.0.0.1:8080/v1",
        llama_cpp_api_key="local",
        use_agent_layer=False,
        ingester_llm_provider="",
        ingester_llm_model="",
        ingester_fix_type_llm_provider="llama_cpp",
        ingester_fix_type_llm_model="qwen2.5-coder:7b",
        allow_legacy_subprocess_providers=False,
    )

    router = _create_llm_router(args, config, "ingester", [INGESTER_FIX_TYPE])

    assert isinstance(router, LLMRouter)
    assert created == [("anthropic", "claude-sonnet-4-5-20250929")]
    assert isinstance(router.for_prompt(INGESTER_FIX_TYPE), DeterministicTypeFixer)
