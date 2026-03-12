from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ageom.commands._helpers import _create_llm_router
from ageom.ingester.ast_state_hoister import (
    ASTStateHoister,
    _hoist_from_attrs,
    _parse_hoist_prompt,
)
from ageom.ingester.chunker import ChunkerDeps, hoist_state
from ageom.architect.models import IOSpec
from ageom.ingester.models import MacroAtomSpec, ProposedMacroPlan, RawDataFlowGraph
from ageom.llm_router import INGESTER_HOIST_STATE, LLMRouter


def _user(attrs: list[str], macro_plan: list[dict[str, object]]) -> str:
    return (
        f"Cross-window attributes: {attrs}\n\n"
        "Macro-atom plan:\n"
        f"{json.dumps(macro_plan, indent=2)}\n\n"
        "Return JSON:\n{}\n"
    )


def test_parse_hoist_prompt_extracts_attrs_and_plan():
    attrs, plan = _parse_hoist_prompt(
        _user(
            ["buffer", "count"],
            [
                {
                    "name": "Sample Accumulator",
                    "inputs": [{"name": "value", "type_desc": "float"}],
                    "outputs": [{"name": "buffer", "type_desc": "list[float]"}],
                }
            ],
        )
    )

    assert attrs == ["buffer", "count"]
    assert plan[0]["name"] == "Sample Accumulator"


def test_hoist_from_attrs_infers_types_from_macro_plan():
    result = _hoist_from_attrs(
        ["buffer", "count"],
        [
            {
                "name": "Sample Accumulator",
                "inputs": [{"name": "value", "type_desc": "float"}],
                "outputs": [{"name": "buffer", "type_desc": "list[float]"}],
            },
            {
                "name": "Average Computer",
                "inputs": [{"name": "count", "type_desc": "int"}],
                "outputs": [{"name": "result", "type_desc": "float"}],
            },
        ],
    )

    assert result is not None
    state_model = result["state_models"][0]
    assert state_model["model_name"] == "PipelineState"
    assert state_model["fields"] == [["buffer", "list[float]"], ["count", "int"]]
    assert state_model["source_attrs"] == ["buffer", "count"]


@pytest.mark.asyncio
async def test_ast_state_hoister_falls_back_when_types_are_too_unknown():
    fallback = AsyncMock()
    fallback.complete.return_value = "fallback"
    hoister = ASTStateHoister(fallback)

    response = await hoister.complete(
        "sys",
        _user(
            ["buffer", "count", "history"],
            [{"name": "Average Computer", "inputs": [], "outputs": []}],
        ),
    )

    assert response == "fallback"
    fallback.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_hoist_state_uses_deterministic_hoister():
    fallback = AsyncMock()
    fallback.complete.side_effect = AssertionError("fallback should not be used")
    llm = LLMRouter(
        default=fallback,
        overrides={INGESTER_HOIST_STATE: ASTStateHoister(fallback)},
    )
    state = {
        "raw_dfg": RawDataFlowGraph(class_name="RollingAverager", cross_window_attrs=["buffer", "count"]),
        "proposed_plan": ProposedMacroPlan(
            macro_atoms=[
                MacroAtomSpec(
                    name="Sample Accumulator",
                    method_names=["add_sample"],
                    inputs=[],
                    outputs=[IOSpec(name="buffer", type_desc="list[float]")],
                ),
                MacroAtomSpec(
                    name="Average Computer",
                    method_names=["compute_average"],
                    inputs=[IOSpec(name="count", type_desc="int")],
                    outputs=[IOSpec(name="result", type_desc="float")],
                ),
            ]
        ),
    }
    config = {"configurable": {"deps": ChunkerDeps(llm=llm)}}

    result = await hoist_state(state, config)

    assert result["proposed_plan"].state_models
    model = result["proposed_plan"].state_models[0]
    assert model.model_name == "PipelineState"
    assert ("buffer", "list[float]") in model.fields
    assert ("count", "int") in model.fields


def test_create_llm_router_wraps_ingester_hoist_state_deterministically(monkeypatch):
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
        ingester_hoist_state_llm_provider="llama_cpp",
        ingester_hoist_state_llm_model="qwen3:14b",
        allow_legacy_subprocess_providers=False,
    )

    router = _create_llm_router(args, config, "ingester", [INGESTER_HOIST_STATE])

    assert isinstance(router, LLMRouter)
    assert created == [("anthropic", "claude-sonnet-4-5-20250929")]
    assert isinstance(router.for_prompt(INGESTER_HOIST_STATE), ASTStateHoister)
