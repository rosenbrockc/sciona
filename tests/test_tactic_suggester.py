from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sciona.commands._helpers import _create_llm_router
from sciona.judge.models import CompilerFeedback
from sciona.llm_router import LLMRouter, SYNTHESIZER_TACTIC
from sciona.synthesizer.prompts import (
    GENERATE_IMPLEMENTATION_SYSTEM_PYTHON,
    GENERATE_TACTIC_SYSTEM,
    GENERATE_TACTIC_USER,
)
from sciona.synthesizer.repair import CompileCheck, RepairDeps, RepairState, repair_graph
from sciona.synthesizer.tactic_suggester import DeterministicTacticSuggester
from sciona.synthesizer.models import SkeletonFile


def _make_user(goal_type: str, hypotheses: str = "", lemmas: str = "(use standard Mathlib tactics)") -> str:
    return GENERATE_TACTIC_USER.format(
        goal_type=goal_type,
        hypotheses=hypotheses,
        available_lemmas=lemmas,
    )


def _make_skeleton(source: str, prover: str = "lean4") -> SkeletonFile:
    sorry_count = source.lower().count("sorry") if prover == "lean4" else source.count("Admitted.")
    if prover == "python":
        sorry_count = source.count("NotImplementedError")
    return SkeletonFile(prover=prover, source_code=source, sorry_count=sorry_count)


def _make_mock_env(feedback_sequence: list[CompilerFeedback]) -> AsyncMock:
    env = AsyncMock()
    env.prover_name = "lean4"
    env._run = AsyncMock(side_effect=feedback_sequence)
    env.close = AsyncMock()
    return env


@pytest.mark.asyncio
async def test_tactic_suggester_returns_rfl_for_identical_equality():
    fallback = AsyncMock()
    suggester = DeterministicTacticSuggester(fallback)

    response = await suggester.complete(
        GENERATE_TACTIC_SYSTEM,
        _make_user("x = x"),
    )

    assert response == "rfl"
    fallback.complete.assert_not_called()


@pytest.mark.asyncio
async def test_tactic_suggester_returns_norm_num_for_numeric_goal():
    fallback = AsyncMock()
    suggester = DeterministicTacticSuggester(fallback)

    response = await suggester.complete(
        GENERATE_TACTIC_SYSTEM,
        _make_user("1 + 1 = 2"),
    )

    assert response == "norm_num"
    fallback.complete.assert_not_called()


@pytest.mark.asyncio
async def test_tactic_suggester_returns_python_bool_stub():
    fallback = AsyncMock()
    suggester = DeterministicTacticSuggester(fallback)
    user = _make_user(
        "def is_sorted(xs: list[int]) -> bool:",
        " >>    1 | def is_sorted(xs: list[int]) -> bool:\n >>    2 |     raise NotImplementedError(\"TODO\")",
    )

    response = await suggester.complete(
        GENERATE_IMPLEMENTATION_SYSTEM_PYTHON,
        user,
    )

    assert response == "    return True"
    fallback.complete.assert_not_called()


@pytest.mark.asyncio
async def test_tactic_suggester_falls_back_for_complex_goal():
    fallback = AsyncMock()
    fallback.complete.return_value = "fallback"
    suggester = DeterministicTacticSuggester(fallback)

    response = await suggester.complete(
        GENERATE_TACTIC_SYSTEM,
        _make_user("forall n : Nat, n = n"),
    )

    assert response == "fallback"
    fallback.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_sorry_elimination_uses_deterministic_tactic():
    source = "theorem foo : 1 + 1 = 2 := by\n  sorry\n"
    skeleton = _make_skeleton(source)
    env = _make_mock_env(
        [
            CompilerFeedback(
                raw_output="unsolved goals\n⊢ 1 + 1 = 2",
                errors=[],
                goals_remaining=["⊢ 1 + 1 = 2"],
            ),
            CompilerFeedback(raw_output="", errors=[], goals_remaining=[]),
        ]
    )
    fallback = AsyncMock()
    fallback.complete.side_effect = AssertionError("fallback should not be used")
    llm = LLMRouter(
        default=fallback,
        overrides={SYNTHESIZER_TACTIC: DeterministicTacticSuggester(fallback)},
    )

    state = RepairState(skeleton=skeleton, max_iterations=5, sorry_remaining=1)
    result = await repair_graph.run(
        CompileCheck(),
        state=state,
        deps=RepairDeps(env=env, llm=llm),
    )

    assert state.compiled_ok is True
    assert "norm_num" in result.output.source_code
    assert "sorry" not in result.output.source_code


def test_create_llm_router_wraps_synthesizer_tactic_deterministically(monkeypatch):
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
        synthesizer_llm_provider="",
        synthesizer_llm_model="",
        synthesizer_tactic_llm_provider="",
        synthesizer_tactic_llm_model="",
        allow_legacy_subprocess_providers=False,
    )

    router = _create_llm_router(args, config, "synthesizer", [SYNTHESIZER_TACTIC])

    assert isinstance(router, LLMRouter)
    assert created == [("anthropic", "claude-sonnet-4-5-20250929")]
    assert isinstance(router.for_prompt(SYNTHESIZER_TACTIC), DeterministicTacticSuggester)
