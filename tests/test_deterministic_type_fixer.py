from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ageom.architect.handoff import CDGExport
from ageom.commands._helpers import _create_llm_router
from ageom.ingester.deterministic_type_fixer import (
    DeterministicTypeFixer,
    _extract_line_number,
    _parse_fix_type_prompt,
)
from ageom.ingester.graph import IngesterDeps, repair_types
from ageom.ingester.models import IngestionBundle
from ageom.llm_router import INGESTER_FIX_TYPE, LLMRouter


def _prompt(errors: str, source: str) -> str:
    return (
        "mypy errors:\n"
        f"{errors}\n\n"
        "Generated source:\n"
        "```python\n"
        f"{source}\n"
        "```\n\n"
        "Return JSON array of fixes:\n[]"
    )


def test_parse_fix_type_prompt_extracts_sections():
    errors, source = _parse_fix_type_prompt(
        _prompt('_check.py:2: error: Name "np" is not defined', "def f():\n    return np.array([1])")
    )

    assert 'Name "np" is not defined' in errors
    assert "return np.array" in source


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


def test_create_llm_router_wraps_ingester_fix_type_deterministically(monkeypatch):
    created: list[tuple[str, str]] = []

    def _fake_create_llm_client(*, provider, model, **kwargs):
        created.append((provider, model))
        client = AsyncMock()
        client.provider = provider
        client.model = model
        return client

    monkeypatch.setattr("ageom.hunter.llm.create_llm_client", _fake_create_llm_client)

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
