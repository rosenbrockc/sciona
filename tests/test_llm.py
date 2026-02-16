"""Tests for ageom.hunter.llm provider clients and factory."""

from __future__ import annotations

import sys
import types

import pytest

from ageom.hunter.llm import ClaudeLLMClient, CodexLLMClient, create_llm_client


class _FakeAnthropicMessages:
    async def create(self, **kwargs):
        content = types.SimpleNamespace(text="anthropic-ok")
        return types.SimpleNamespace(content=[content])


class _FakeAsyncAnthropic:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.messages = _FakeAnthropicMessages()


class _FakeChatCompletions:
    async def create(self, **kwargs):
        msg = types.SimpleNamespace(content="codex-ok")
        choice = types.SimpleNamespace(message=msg)
        return types.SimpleNamespace(choices=[choice])


class _FakeAsyncOpenAI:
    def __init__(self, api_key: str, base_url: str | None = None):
        self.api_key = api_key
        self.base_url = base_url
        self.chat = types.SimpleNamespace(completions=_FakeChatCompletions())


class TestCreateLLMClient:
    def test_rejects_unsupported_provider(self):
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            create_llm_client(
                provider="unknown",
                model="x",
                max_tokens=16,
            )

    def test_anthropic_requires_api_key(self):
        with pytest.raises(ValueError, match="AGEOM_ANTHROPIC_API_KEY not set"):
            create_llm_client(
                provider="anthropic",
                model="claude-sonnet-4-5-20250929",
                max_tokens=32,
            )

    def test_codex_requires_api_key(self):
        with pytest.raises(ValueError, match="AGEOM_OPENAI_API_KEY not set"):
            create_llm_client(
                provider="codex",
                model="codex-mini-latest",
                max_tokens=32,
            )


class TestConcreteClients:
    @pytest.mark.asyncio
    async def test_claude_client_complete(self, monkeypatch):
        fake_mod = types.SimpleNamespace(AsyncAnthropic=_FakeAsyncAnthropic)
        monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

        client = create_llm_client(
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            max_tokens=64,
            anthropic_api_key="test-key",
        )
        assert isinstance(client, ClaudeLLMClient)
        text = await client.complete("system", "user")
        assert text == "anthropic-ok"

    @pytest.mark.asyncio
    async def test_codex_client_complete(self, monkeypatch):
        fake_mod = types.SimpleNamespace(AsyncOpenAI=_FakeAsyncOpenAI)
        monkeypatch.setitem(sys.modules, "openai", fake_mod)

        client = create_llm_client(
            provider="codex",
            model="codex-mini-latest",
            max_tokens=64,
            openai_api_key="test-key",
            openai_base_url="https://example.invalid/v1",
        )
        assert isinstance(client, CodexLLMClient)
        text = await client.complete("system", "user")
        assert text == "codex-ok"

