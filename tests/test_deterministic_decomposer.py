from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sciona.architect.deterministic_decompose import (
    DeterministicDecomposer,
    _parse_decompose_prompt,
)
from sciona.architect.prompts import DECOMPOSE_NODE_SYSTEM, DECOMPOSE_NODE_USER
from sciona.commands._helpers import _create_llm_router
from sciona.llm_router import ARCHITECT_DECOMPOSE, LLMRouter


def _prompt(
    *,
    node_name: str,
    description: str,
    concept_type: str,
    inputs: str,
    outputs: str,
    depth: int = 1,
    max_depth: int = 8,
) -> str:
    return DECOMPOSE_NODE_USER.format(
        node_name=node_name,
        node_description=description,
        concept_type=concept_type,
        inputs=inputs,
        outputs=outputs,
        depth=depth,
        max_depth=max_depth,
        primitives="No relevant primitives found.",
        example_decompositions="",
        retry_context="",
    )


def test_parse_decompose_prompt_extracts_structured_fields():
    prompt = _prompt(
        node_name="Merge Sort",
        description="Recursively split the list and merge sorted halves.",
        concept_type="divide_and_conquer",
        inputs="data: list[int], descending: bool (optional, default=False)",
        outputs="result: list[int]",
    )

    parsed = _parse_decompose_prompt(prompt)

    assert parsed.node_name == "Merge Sort"
    assert parsed.concept_type is not None
    assert parsed.concept_type.value == "divide_and_conquer"
    assert [port.name for port in parsed.inputs] == ["data", "descending"]
    assert parsed.inputs[1].required is False
    assert parsed.inputs[1].default_value_repr == "False"
    assert parsed.outputs[0].name == "result"


@pytest.mark.asyncio
async def test_deterministic_decomposer_emits_divide_and_conquer_skeleton():
    fallback = AsyncMock()
    decomposer = DeterministicDecomposer(fallback)

    response = await decomposer.complete(
        DECOMPOSE_NODE_SYSTEM,
        _prompt(
            node_name="Merge Sort",
            description="Recursively split the list and merge sorted halves.",
            concept_type="divide_and_conquer",
            inputs="data: list[int]",
            outputs="result: list[int]",
        ),
    )

    payload = json.loads(response)
    names = [node["name"] for node in payload["sub_nodes"]]
    assert names == ["Split", "Recurse Left", "Recurse Right", "Merge"]
    assert payload["sub_nodes"][0]["inputs"][0]["name"] == "data"
    assert payload["sub_nodes"][-1]["outputs"][0]["name"] == "result"
    fallback.complete.assert_not_called()


@pytest.mark.asyncio
async def test_deterministic_decomposer_emits_signal_filter_skeleton():
    fallback = AsyncMock()
    decomposer = DeterministicDecomposer(fallback)

    response = await decomposer.complete(
        DECOMPOSE_NODE_SYSTEM,
        _prompt(
            node_name="ECG Bandpass Filter",
            description="Design and apply a stable bandpass filter to ECG samples.",
            concept_type="signal_filter",
            inputs="signal: ndarray, fs: float",
            outputs="filtered: ndarray",
        ),
    )

    payload = json.loads(response)
    names = [node["name"] for node in payload["sub_nodes"]]
    assert names == [
        "Design Filter",
        "Validate Stability",
        "Apply Filter",
        "Frequency Response",
    ]
    assert payload["sub_nodes"][0]["inputs"][0]["name"] == "signal"
    assert payload["sub_nodes"][-1]["outputs"][0]["name"] == "filtered"
    fallback.complete.assert_not_called()


@pytest.mark.asyncio
async def test_deterministic_decomposer_falls_back_for_ambiguous_goal():
    fallback = AsyncMock()
    fallback.complete.return_value = "fallback"
    decomposer = DeterministicDecomposer(fallback)

    response = await decomposer.complete(
        DECOMPOSE_NODE_SYSTEM,
        _prompt(
            node_name="Process Data",
            description="Process some input data and produce an output.",
            concept_type="custom",
            inputs="x: Any",
            outputs="y: Any",
        ),
    )

    assert response == "fallback"
    fallback.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_deterministic_decomposer_does_not_reexpand_template_child_node():
    fallback = AsyncMock()
    fallback.complete.return_value = "fallback"
    decomposer = DeterministicDecomposer(fallback)

    response = await decomposer.complete(
        DECOMPOSE_NODE_SYSTEM,
        _prompt(
            node_name="Validate Stability",
            description="Check that candidate coefficients satisfy stability criteria.",
            concept_type="signal_filter",
            inputs="coefficients: vector[float]",
            outputs="valid_coefficients: vector[float]",
            depth=2,
        ),
    )

    assert response == "fallback"
    fallback.complete.assert_awaited_once()


def test_create_llm_router_wraps_architect_decompose_deterministically(monkeypatch):
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
        architect_decompose_llm_provider="codex_shim",
        architect_decompose_llm_model="gpt-5.4-codex",
        architect_critique_llm_provider="",
        architect_critique_llm_model="",
        allow_legacy_subprocess_providers=False,
    )

    router = _create_llm_router(args, config, "architect", [ARCHITECT_DECOMPOSE])

    assert isinstance(router, LLMRouter)
    assert created == [
        ("anthropic", "claude-sonnet-4-5-20250929"),
        ("codex_shim", "gpt-5.4-codex"),
    ]
    assert isinstance(router.for_prompt(ARCHITECT_DECOMPOSE), DeterministicDecomposer)
