from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ageom.architect.handoff import CDGExport
from ageom.architect.models import DependencyEdge
from ageom.commands._helpers import _create_llm_router
from ageom.ingester.deterministic_cycle_breaker import (
    DeterministicCycleBreaker,
    _break_cycle,
    _parse_cycle_prompt,
)
from ageom.ingester.graph import IngesterDeps, repair_message_cycle
from ageom.ingester.models import IngestionBundle
from ageom.llm_router import INGESTER_FIX_MESSAGE_CYCLE, LLMRouter


def _prompt(source: str, deadlocked_nodes: str = "variable_to_factor, memoization_state") -> str:
    return (
        f"Deadlocked nodes: {deadlocked_nodes}\n"
        "Cycle edges: variable_to_factor -> factor_to_variable\n"
        "factor_to_variable -> memoization_state\n"
        "memoization_state -> variable_to_factor\n"
        "Current witness source:\n"
        "```python\n"
        f"{source}\n"
        "```\n\n"
        "Fix the memoization witness to break the cycle (add damping, convergence "
        "epsilon, or iteration cap).\n"
        "Return JSON array of patches:\n[]"
    )


def test_parse_cycle_prompt_extracts_multiline_cycle_edges():
    deadlocked_nodes, cycle_edges, witness_source = _parse_cycle_prompt(
        _prompt("def witness():\n    return None")
    )

    assert deadlocked_nodes == ["variable_to_factor", "memoization_state"]
    assert cycle_edges == [
        "variable_to_factor -> factor_to_variable",
        "factor_to_variable -> memoization_state",
        "memoization_state -> variable_to_factor",
    ]
    assert "def witness()" in witness_source


def test_break_cycle_prefers_iteration_cap_for_unbounded_loop():
    source = (
        "def witness_cycle() -> dict[str, float]:\n"
        "    while True:\n"
        "        new_msg = old_msg\n"
        "        if ready:\n"
        "            return new_msg"
    )

    result = _break_cycle(["a", "b"], ["a -> b", "b -> a"], source)

    assert result is not None
    patches, strategy = result
    assert strategy == "iteration_cap"
    assert "max_iter = 16" in patches[0]["replacement"]
    assert "while iter_count < max_iter:" in patches[0]["replacement"]


@pytest.mark.asyncio
async def test_deterministic_cycle_breaker_rewrites_memo_convergence():
    fallback = AsyncMock()
    fixer = DeterministicCycleBreaker(fallback)
    source = (
        "def witness_memoization_state(\n"
        "    var_messages: dict[str, AbstractArray],\n"
        "    factor_messages: dict[str, AbstractArray],\n"
        ") -> tuple[dict[str, AbstractArray], bool]:\n"
        '    """Ghost witness for message-passing: Memoization State."""\n'
        "    memo_state = {k: v for k, v in var_messages.items()}\n"
        "    converged = False\n"
        "    return memo_state, converged"
    )

    response = await fixer.complete("sys", _prompt(source))

    patches = json.loads(response)
    assert patches == [
        {
            "line_start": 7,
            "line_end": 7,
            "replacement": (
                "    converged = bool(var_messages) and "
                "set(var_messages.keys()) == set(factor_messages.keys())"
            ),
        }
    ]
    fallback.complete.assert_not_called()


@pytest.mark.asyncio
async def test_deterministic_cycle_breaker_falls_back_for_complex_cycle():
    fallback = AsyncMock()
    fallback.complete.return_value = "fallback"
    fixer = DeterministicCycleBreaker(fallback)

    response = await fixer.complete(
        "sys",
        _prompt(
            "def witness_cycle():\n    return None",
            deadlocked_nodes="a, b, c, d, e, f",
        ),
    )

    assert response == "fallback"
    fallback.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_repair_message_cycle_uses_deterministic_cycle_breaker_patch():
    fallback = AsyncMock()
    fallback.complete.side_effect = AssertionError("fallback should not be used")
    llm = LLMRouter(
        default=fallback,
        overrides={INGESTER_FIX_MESSAGE_CYCLE: DeterministicCycleBreaker(fallback)},
    )
    source = (
        "def witness_memoization_state(var_messages, factor_messages):\n"
        '    """Ghost witness for message-passing: Memoization State."""\n'
        "    memo_state = {k: v for k, v in var_messages.items()}\n"
        "    converged = False\n"
        "    return memo_state, converged\n"
    )
    bundle = IngestionBundle(
        cdg=CDGExport(
            nodes=[],
            edges=[
                DependencyEdge(
                    source_id="memoization_state",
                    target_id="variable_to_factor",
                    output_name="memo_state",
                    input_name="memo_state",
                    source_type="dict[str, ndarray]",
                    target_type="dict[str, ndarray]",
                )
            ],
        ),
        generated_witnesses=source,
        ghost_sim_report={
            "cyclic_deadlock": True,
            "deadlock_nodes": ["memoization_state", "variable_to_factor"],
        },
    )
    state = {"bundle": bundle, "ghost_repair_count": 0}
    config = {"configurable": {"deps": IngesterDeps(llm=llm)}}

    result = await repair_message_cycle(state, config)

    assert result["ghost_repair_count"] == 1
    assert "bool(var_messages)" in result["bundle"].generated_witnesses


def test_create_llm_router_wraps_ingester_fix_message_cycle_deterministically(monkeypatch):
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
        ingester_fix_message_cycle_llm_provider="llama_cpp",
        ingester_fix_message_cycle_llm_model="qwen3:14b",
        allow_legacy_subprocess_providers=False,
    )

    router = _create_llm_router(args, config, "ingester", [INGESTER_FIX_MESSAGE_CYCLE])

    assert isinstance(router, LLMRouter)
    assert created == [("anthropic", "claude-sonnet-4-5-20250929")]
    assert isinstance(
        router.for_prompt(INGESTER_FIX_MESSAGE_CYCLE),
        DeterministicCycleBreaker,
    )
