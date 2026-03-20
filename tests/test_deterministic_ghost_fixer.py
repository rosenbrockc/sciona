from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sciona.architect.handoff import CDGExport
from sciona.commands._helpers import _create_llm_router
from sciona.ingester.deterministic_ghost_fixer import (
    DeterministicGhostFixer,
    _parse_fix_ghost_prompt,
)
from sciona.ingester.graph import IngesterDeps, repair_ghost
from sciona.ingester.models import IngestionBundle
from sciona.llm_router import INGESTER_FIX_GHOST, LLMRouter


def _prompt(error_message: str, witness_source: str, function_name: str = "witness_filter") -> str:
    return (
        "Ghost simulation error:\n"
        "  Node: Filter\n"
        f"  Function: {function_name}\n"
        f"  Error: {error_message}\n\n"
        "Generated witnesses:\n"
        "```python\n"
        f"{witness_source}\n"
        "```\n\n"
        "Return JSON array of fixes:\n[]"
    )


def test_parse_fix_ghost_prompt_extracts_fields():
    node, function_name, error, source = _parse_fix_ghost_prompt(
        _prompt("NoneType has no attribute shape", "def witness_filter(x: AbstractArray) -> AbstractArray:\n    return None")
    )

    assert node == "Filter"
    assert function_name == "witness_filter"
    assert "NoneType" in error
    assert "return None" in source


@pytest.mark.asyncio
async def test_deterministic_ghost_fixer_rewrites_none_returning_witness():
    fallback = AsyncMock()
    fixer = DeterministicGhostFixer(fallback)
    response = await fixer.complete(
        "sys",
        _prompt(
            "NoneType has no attribute shape",
            "def witness_filter(x: AbstractArray) -> AbstractArray:\n    return None",
        ),
    )

    fixes = json.loads(response)
    assert fixes[0]["witness_name"] == "witness_filter"
    assert "return x" in fixes[0]["replacement"]
    fallback.complete.assert_not_called()


@pytest.mark.asyncio
async def test_deterministic_ghost_fixer_rewrites_state_tuple_stub():
    fallback = AsyncMock()
    fixer = DeterministicGhostFixer(fallback)
    response = await fixer.complete(
        "sys",
        _prompt(
            "shape access on None result",
            "def witness_filter(signal: AbstractSignal, state: AbstractSignal) -> tuple[AbstractSignal, AbstractSignal]:\n    return None, state",
        ),
    )

    fixes = json.loads(response)
    assert "return signal, state" in fixes[0]["replacement"]
    fallback.complete.assert_not_called()


@pytest.mark.asyncio
async def test_deterministic_ghost_fixer_falls_back_for_non_stub_error():
    fallback = AsyncMock()
    fallback.complete.return_value = "fallback"
    fixer = DeterministicGhostFixer(fallback)
    response = await fixer.complete(
        "sys",
        _prompt(
            "domain mismatch between signal and coefficients",
            "def witness_filter(x: AbstractArray) -> AbstractArray:\n    return x",
        ),
    )

    assert response == "fallback"
    fallback.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_repair_ghost_uses_deterministic_fix():
    fallback = AsyncMock()
    fallback.complete.side_effect = AssertionError("fallback should not be used")
    llm = LLMRouter(
        default=fallback,
        overrides={INGESTER_FIX_GHOST: DeterministicGhostFixer(fallback)},
    )
    bundle = IngestionBundle(
        cdg=CDGExport(nodes=[], edges=[]),
        generated_witnesses="def witness_filter(x: AbstractArray) -> AbstractArray:\n    return None\n",
        ghost_sim_report={
            "error_node": "Filter",
            "error_function": "witness_filter",
            "error": "NoneType has no attribute shape",
        },
    )
    state = {
        "bundle": bundle,
        "ghost_repair_count": 0,
    }
    config = {"configurable": {"deps": IngesterDeps(llm=llm)}}

    result = await repair_ghost(state, config)

    assert result["ghost_repair_count"] == 1
    assert "return x" in result["bundle"].generated_witnesses


def test_create_llm_router_wraps_ingester_fix_ghost_deterministically(monkeypatch):
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
        ingester_fix_ghost_llm_provider="llama_cpp",
        ingester_fix_ghost_llm_model="qwen3:14b",
        allow_legacy_subprocess_providers=False,
    )

    router = _create_llm_router(args, config, "ingester", [INGESTER_FIX_GHOST])

    assert isinstance(router, LLMRouter)
    assert created == [("anthropic", "claude-sonnet-4-5-20250929")]
    assert isinstance(router.for_prompt(INGESTER_FIX_GHOST), DeterministicGhostFixer)
