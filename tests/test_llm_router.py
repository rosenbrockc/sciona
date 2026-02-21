"""Tests for ageom.llm_router — LLMRouter, select_llm, and per-prompt config."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ageom.llm_router import (
    ARCHITECT_STRATEGY,
    INGESTER_CHUNK,
    SYNTHESIZER_TACTIC,
    LLMRouter,
    select_llm,
)


def _make_mock_llm(name: str = "default") -> AsyncMock:
    llm = AsyncMock()
    llm.name = name
    llm.complete = AsyncMock(return_value=f"{name}-response")
    llm.complete_with_grammar = AsyncMock(return_value=f"{name}-grammar-response")
    return llm


class TestLLMRouter:
    def test_for_prompt_returns_override_when_present(self):
        default = _make_mock_llm("default")
        override = _make_mock_llm("override")
        router = LLMRouter(default=default, overrides={ARCHITECT_STRATEGY: override})

        result = router.for_prompt(ARCHITECT_STRATEGY)
        assert result is override

    def test_for_prompt_returns_default_when_key_missing(self):
        default = _make_mock_llm("default")
        override = _make_mock_llm("override")
        router = LLMRouter(default=default, overrides={ARCHITECT_STRATEGY: override})

        result = router.for_prompt(INGESTER_CHUNK)
        assert result is default

    def test_for_prompt_returns_default_when_no_overrides(self):
        default = _make_mock_llm("default")
        router = LLMRouter(default=default)

        result = router.for_prompt(ARCHITECT_STRATEGY)
        assert result is default

    @pytest.mark.asyncio
    async def test_complete_delegates_to_default(self):
        default = _make_mock_llm("default")
        override = _make_mock_llm("override")
        router = LLMRouter(default=default, overrides={ARCHITECT_STRATEGY: override})

        result = await router.complete("sys", "usr")
        assert result == "default-response"
        default.complete.assert_awaited_once_with("sys", "usr")

    @pytest.mark.asyncio
    async def test_complete_with_grammar_delegates_to_default(self):
        default = _make_mock_llm("default")
        router = LLMRouter(default=default)

        result = await router.complete_with_grammar("sys", "usr", "grammar")
        assert result == "default-grammar-response"
        default.complete_with_grammar.assert_awaited_once_with("sys", "usr", "grammar")


class TestSelectLLM:
    def test_plain_llm_returned_unchanged(self):
        llm = _make_mock_llm("plain")
        result = select_llm(llm, ARCHITECT_STRATEGY)
        assert result is llm

    def test_router_delegates_to_for_prompt(self):
        default = _make_mock_llm("default")
        override = _make_mock_llm("override")
        router = LLMRouter(default=default, overrides={SYNTHESIZER_TACTIC: override})

        result = select_llm(router, SYNTHESIZER_TACTIC)
        assert result is override

    def test_router_falls_back_to_default(self):
        default = _make_mock_llm("default")
        router = LLMRouter(default=default, overrides={})

        result = select_llm(router, SYNTHESIZER_TACTIC)
        assert result is default


class TestConfigPerPromptFields:
    def test_config_picks_up_per_prompt_env_vars(self, monkeypatch):
        monkeypatch.setenv("AGEOM_SYNTHESIZER_TACTIC_LLM_PROVIDER", "llama_cpp")
        monkeypatch.setenv("AGEOM_SYNTHESIZER_TACTIC_LLM_MODEL", "qwen-2.5-coder-32b")
        monkeypatch.setenv("AGEOM_INGESTER_CHUNK_LLM_PROVIDER", "codex")

        from ageom.config import AgeomConfig

        config = AgeomConfig()
        assert config.synthesizer_tactic_llm_provider == "llama_cpp"
        assert config.synthesizer_tactic_llm_model == "qwen-2.5-coder-32b"
        assert config.ingester_chunk_llm_provider == "codex"
        # Non-set fields should be empty
        assert config.architect_strategy_llm_provider == ""
        assert config.architect_strategy_llm_model == ""

    def test_all_per_prompt_fields_default_to_empty(self):
        from ageom.config import AgeomConfig

        config = AgeomConfig()
        from ageom.llm_router import ALL_PROMPT_KEYS

        for key in ALL_PROMPT_KEYS:
            assert getattr(config, f"{key}_llm_provider") == ""
            assert getattr(config, f"{key}_llm_model") == ""
