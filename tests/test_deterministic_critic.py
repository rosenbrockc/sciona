from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sciona.architect.deterministic_critic import (
    DeterministicCritic,
    _parse_critique_prompt,
)
from sciona.architect.prompts import CRITIQUE_SYSTEM, CRITIQUE_USER
from sciona.commands._helpers import _create_llm_router
from sciona.llm_router import ARCHITECT_CRITIQUE, LLMRouter


def _prompt(
    *,
    parent_name: str,
    parent_description: str,
    parent_inputs: str,
    parent_outputs: str,
    sub_nodes: str,
    edges: str = "  (no edges)",
) -> str:
    return CRITIQUE_USER.format(
        parent_name=parent_name,
        parent_description=parent_description,
        parent_inputs=parent_inputs,
        parent_outputs=parent_outputs,
        sub_nodes=sub_nodes,
        edges=edges,
        current_depth=1,
        max_depth=8,
        primitives="No relevant primitives found.",
    )


def test_parse_critique_prompt_extracts_sub_nodes():
    prompt = _prompt(
        parent_name="Merge Sort",
        parent_description="Recursively split the list and merge sorted halves.",
        parent_inputs="data: list[int]",
        parent_outputs="result: list[int]",
        sub_nodes=(
            "  - Split [divide_and_conquer] (inputs: data: list[int], outputs: left: list[int], right: list[int], "
            "status: pending, matched_primitive: (none))\n"
            "  - Merge [divide_and_conquer] (inputs: left_result: list[int], right_result: list[int], outputs: result: list[int], "
            "status: pending, matched_primitive: merge)"
        ),
    )

    parsed = _parse_critique_prompt(prompt)

    assert parsed.parent_name == "Merge Sort"
    assert len(parsed.sub_nodes) == 2
    assert parsed.sub_nodes[1].name == "Merge"
    assert parsed.sub_nodes[1].matched_primitive == "merge"


@pytest.mark.asyncio
async def test_deterministic_critic_approves_clean_decomposition():
    fallback = AsyncMock()
    critic = DeterministicCritic(fallback)
    prompt = _prompt(
        parent_name="Merge Sort",
        parent_description="Recursively split the list and merge sorted halves.",
        parent_inputs="data: list[int]",
        parent_outputs="result: list[int]",
        sub_nodes=(
            "  - Split [divide_and_conquer] (inputs: data: list[int], outputs: left: list[int], right: list[int], "
            "status: pending, matched_primitive: (none))\n"
            "  - Recurse Left [divide_and_conquer] (inputs: left: list[int], outputs: left_result: list[int], "
            "status: pending, matched_primitive: (none))\n"
            "  - Recurse Right [divide_and_conquer] (inputs: right: list[int], outputs: right_result: list[int], "
            "status: pending, matched_primitive: (none))\n"
            "  - Merge [divide_and_conquer] (inputs: left_result: list[int], right_result: list[int], outputs: result: list[int], "
            "status: atomic, matched_primitive: merge)"
        ),
    )

    response = await critic.complete(CRITIQUE_SYSTEM, prompt)

    payload = json.loads(response)
    assert payload["approved"] is True
    assert "Deterministic semantic critique passed" in payload["reason"]
    fallback.complete.assert_not_called()


@pytest.mark.asyncio
async def test_deterministic_critic_falls_back_when_output_coverage_is_missing():
    fallback = AsyncMock()
    fallback.complete.return_value = "fallback"
    critic = DeterministicCritic(fallback)
    prompt = _prompt(
        parent_name="Merge Sort",
        parent_description="Recursively split the list and merge sorted halves.",
        parent_inputs="data: list[int]",
        parent_outputs="result: list[int]",
        sub_nodes=(
            "  - Split [divide_and_conquer] (inputs: data: list[int], outputs: left: list[int], right: list[int], "
            "status: pending, matched_primitive: (none))\n"
            "  - Recurse Left [divide_and_conquer] (inputs: left: list[int], outputs: left_result: list[int], "
            "status: pending, matched_primitive: (none))"
        ),
    )

    response = await critic.complete(CRITIQUE_SYSTEM, prompt)

    assert response == "fallback"
    fallback.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_deterministic_critic_falls_back_for_trivial_child():
    fallback = AsyncMock()
    fallback.complete.return_value = "fallback"
    critic = DeterministicCritic(fallback)
    prompt = _prompt(
        parent_name="Merge Sort",
        parent_description="Recursively split the list and merge sorted halves.",
        parent_inputs="data: list[int]",
        parent_outputs="result: list[int]",
        sub_nodes=(
            "  - Merge Sort [divide_and_conquer] (inputs: data: list[int], outputs: result: list[int], "
            "status: pending, matched_primitive: (none))\n"
            "  - Merge [divide_and_conquer] (inputs: left_result: list[int], right_result: list[int], outputs: result: list[int], "
            "status: atomic, matched_primitive: merge)"
        ),
    )

    response = await critic.complete(CRITIQUE_SYSTEM, prompt)

    assert response == "fallback"
    fallback.complete.assert_awaited_once()


def test_create_llm_router_wraps_architect_critique_deterministically(monkeypatch):
    created: list[tuple[str, str]] = []

    def _fake_create_llm_client(*, provider, model, **kwargs):
        created.append((provider, model))
        client = AsyncMock()
        client.provider = provider
        client.model = model
        return client

    monkeypatch.setattr("sciona.hunter.llm.create_llm_client", _fake_create_llm_client)

    args = SimpleNamespace(llm_provider=None, llm_model=None, llm_max_tokens=None, mode="verified")
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
        architect_llm_provider="",
        architect_llm_model="",
        architect_strategy_llm_provider="",
        architect_strategy_llm_model="",
        architect_decompose_llm_provider="",
        architect_decompose_llm_model="",
        architect_critique_llm_provider="codex_shim",
        architect_critique_llm_model="gpt-5.4-codex",
        allow_legacy_subprocess_providers=False,
    )

    router = _create_llm_router(args, config, "architect", [ARCHITECT_CRITIQUE])

    assert isinstance(router, LLMRouter)
    assert created == [
        ("anthropic", "claude-sonnet-4-5-20250929"),
        ("codex_shim", "gpt-5.4-codex"),
    ]
    assert isinstance(router.for_prompt(ARCHITECT_CRITIQUE), DeterministicCritic)
