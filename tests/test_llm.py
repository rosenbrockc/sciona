"""Tests for ageom.hunter.llm provider clients and factory."""

from __future__ import annotations

import sys
import types
import re
import shutil

import pytest

from ageom.hunter.llm import (
    ClaudeLLMClient,
    CodexLLMClient,
    LlamaCppLLMClient,
    create_llm_client,
)
from ageom.hunter.gemini_shim import _stage_gemini_home


class _FakeAnthropicMessages:
    async def create(self, **kwargs):
        content = types.SimpleNamespace(text="anthropic-ok")
        return types.SimpleNamespace(content=[content])


class _FakeAsyncAnthropic:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.messages = _FakeAnthropicMessages()


class _FakeChatCompletions:
    def __init__(self):
        self.last_kwargs = {}

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        msg = types.SimpleNamespace(content="codex-ok")
        choice = types.SimpleNamespace(message=msg)
        return types.SimpleNamespace(choices=[choice])


class _FakeAsyncOpenAI:
    def __init__(self, api_key: str, base_url: str | None = None):
        self.api_key = api_key
        self.base_url = base_url
        self.completions = _FakeChatCompletions()
        self.chat = types.SimpleNamespace(completions=self.completions)


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
    def test_stage_gemini_home_copies_auth_state(self, tmp_path):
        source_home = tmp_path / "source-home"
        source_gemini = source_home / ".gemini"
        source_gemini.mkdir(parents=True)
        (source_gemini / "oauth_creds.json").write_text('{"token":"x"}', encoding="utf-8")
        source_gcloud = source_home / ".config" / "gcloud"
        source_gcloud.mkdir(parents=True)
        (source_gcloud / "application_default_credentials.json").write_text(
            '{"client":"y"}', encoding="utf-8"
        )

        runtime_dir = tmp_path / "runtime"
        staged_home = _stage_gemini_home(runtime_dir, source_home=source_home)

        assert staged_home == runtime_dir / "home"
        assert (staged_home / ".gemini" / "oauth_creds.json").read_text(encoding="utf-8") == '{"token":"x"}'
        assert (
            staged_home / ".config" / "gcloud" / "application_default_credentials.json"
        ).read_text(encoding="utf-8") == '{"client":"y"}'

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

    @pytest.mark.asyncio
    async def test_llama_cpp_client_grammar(self, monkeypatch):
        fake_mod = types.SimpleNamespace(AsyncOpenAI=_FakeAsyncOpenAI)
        monkeypatch.setitem(sys.modules, "openai", fake_mod)

        client = create_llm_client(
            provider="llama_cpp",
            model="llama-3.1-8b-instruct",
            max_tokens=128,
            llama_cpp_base_url="http://127.0.0.1:8080/v1",
            llama_cpp_api_key="local",
        )
        assert isinstance(client, LlamaCppLLMClient)
        text = await client.complete_with_grammar("sys", "user", 'root ::= "[]"')
        assert text == "codex-ok"
        assert client._client.completions.last_kwargs["extra_body"] == {
            "grammar": 'root ::= "[]"'
        }

    @pytest.mark.asyncio
    async def test_gemini_shim_reuses_live_worker(self, monkeypatch):
        if shutil.which("node") is None:
            pytest.skip("node is required for gemini_shim test")

        monkeypatch.setenv("AGEOM_GEMINI_DAEMON_FAKE", "1")
        monkeypatch.setenv("AGEOM_GEMINI_SHIM_POOL_SIZE", "1")

        client = create_llm_client(
            provider="gemini_shim",
            model="flash-lite",
            max_tokens=64,
        )
        try:
            first = await client.complete("sys", "one")
            second = await client.complete("sys", "two")
        finally:
            close = getattr(client, "close", None)
            if close is not None:
                await close()

        pid_pattern = re.compile(r"pid=(\d+)")
        count_pattern = re.compile(r"count=(\d+)")
        first_pid = pid_pattern.search(first)
        second_pid = pid_pattern.search(second)
        first_count = count_pattern.search(first)
        second_count = count_pattern.search(second)

        assert first_pid is not None
        assert second_pid is not None
        assert first_count is not None
        assert second_count is not None
        assert first_pid.group(1) == second_pid.group(1)
        assert first_count.group(1) == "1"
        assert second_count.group(1) == "2"

    @pytest.mark.asyncio
    async def test_claude_shim_reuses_live_worker(self, monkeypatch):
        monkeypatch.setenv("AGEOM_CLI_SHIM_DAEMON_FAKE", "1")
        monkeypatch.setenv("AGEOM_CLI_SHIM_POOL_SIZE", "1")

        client = create_llm_client(
            provider="claude_shim",
            model="sonnet",
            max_tokens=64,
        )
        try:
            first = await client.complete("sys", "one")
            second = await client.complete("sys", "two")
        finally:
            close = getattr(client, "close", None)
            if close is not None:
                await close()

        pid_pattern = re.compile(r"pid=(\d+)")
        count_pattern = re.compile(r"count=(\d+)")
        assert pid_pattern.search(first) is not None
        assert pid_pattern.search(second) is not None
        assert pid_pattern.search(first).group(1) == pid_pattern.search(second).group(1)
        assert count_pattern.search(first).group(1) == "1"
        assert count_pattern.search(second).group(1) == "2"

    @pytest.mark.asyncio
    async def test_codex_shim_reuses_live_worker(self, monkeypatch):
        monkeypatch.setenv("AGEOM_CLI_SHIM_DAEMON_FAKE", "1")
        monkeypatch.setenv("AGEOM_CLI_SHIM_POOL_SIZE", "1")

        client = create_llm_client(
            provider="codex_shim",
            model="gpt-5.3-codex",
            max_tokens=64,
        )
        try:
            first = await client.complete("sys", "one")
            second = await client.complete("sys", "two")
        finally:
            close = getattr(client, "close", None)
            if close is not None:
                await close()

        pid_pattern = re.compile(r"pid=(\d+)")
        count_pattern = re.compile(r"count=(\d+)")
        assert pid_pattern.search(first) is not None
        assert pid_pattern.search(second) is not None
        assert pid_pattern.search(first).group(1) == pid_pattern.search(second).group(1)
        assert count_pattern.search(first).group(1) == "1"
        assert count_pattern.search(second).group(1) == "2"
