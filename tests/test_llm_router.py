"""Tests for ageom.llm_router — LLMRouter, select_llm, and per-prompt config."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ageom.llm_router import (
    ARCHITECT_STRATEGY,
    INGESTER_CHUNK,
    PromptKeyLLMClient,
    SYNTHESIZER_TACTIC,
    LLMRouter,
    prompt_timeout_seconds,
    select_llm,
)
from ageom.telemetry import (
    get_event_log,
    reset_telemetry_runtime,
    start_run,
    telemetry_scope,
)
from ageom.architect.strategy_classifier import StrategyClassifier
from ageom.architect.deterministic_decompose import DeterministicDecomposer
from ageom.hunter.candidate_ranker import HeuristicCandidateRanker
from ageom.hunter.failure_analyzer import DeterministicFailureAnalyzer
from ageom.hunter.query_reformulator import HeuristicQueryReformulator


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

    @pytest.mark.asyncio
    async def test_warmup_deduplicates_underlying_clients(self):
        default = _make_mock_llm("default")
        override = _make_mock_llm("override")
        default.warmup = AsyncMock()
        override.warmup = AsyncMock()
        router = LLMRouter(
            default=default,
            overrides={
                ARCHITECT_STRATEGY: override,
                SYNTHESIZER_TACTIC: override,
            },
        )

        await router.warmup()

        default.warmup.assert_awaited_once()
        override.warmup.assert_awaited_once()


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

    def test_wraps_when_telemetry_run_scope_active(self):
        llm = _make_mock_llm("plain")
        with telemetry_scope(run_id="test-run"):
            result = select_llm(llm, ARCHITECT_STRATEGY)
        assert isinstance(result, PromptKeyLLMClient)

    @pytest.mark.asyncio
    async def test_prompt_wrapper_surfaces_client_error_metadata(self):
        reset_telemetry_runtime()
        get_event_log().clear()

        class _FailingClient:
            _telemetry_provider = "codex_shim"
            _telemetry_model = "gpt-5.3-codex"

            async def complete(self, system: str, user: str) -> str:
                raise RuntimeError("shim worker exited early")

            def get_last_error_metadata(self) -> dict[str, object]:
                return {
                    "provider_error_phase": "startup",
                    "provider_exit_code": 17,
                    "provider_stderr_excerpt": "auth failed",
                }

        run_id = start_run("algorithm_creation", run_id="test-run")
        with telemetry_scope(run_id=run_id, stage="hunter_round_1"):
            wrapped = PromptKeyLLMClient(_FailingClient(), "hunter_score")
            with pytest.raises(RuntimeError, match="shim worker exited early"):
                await wrapped.complete("sys", "user")

        events = [
            ev for ev in get_event_log().events if ev.event_type == "PROMPT_DISPATCH_ERROR"
        ]
        assert len(events) == 1
        payload = events[0].payload
        assert payload["error_type"] == "RuntimeError"
        assert payload["provider_error_phase"] == "startup"
        assert payload["provider_exit_code"] == 17
        assert payload["provider_stderr_excerpt"] == "auth failed"

    @pytest.mark.asyncio
    async def test_prompt_wrapper_enforces_router_timeout(self, monkeypatch):
        reset_telemetry_runtime()
        get_event_log().clear()

        class _SlowClient:
            _telemetry_provider = "gemini_shim"
            _telemetry_model = "flash-lite"

            async def complete(self, system: str, user: str) -> str:
                await asyncio.sleep(0.05)
                return "late"

        monkeypatch.setenv("AGEOM_HUNTER_SCORE_TIMEOUT_S", "0.01")
        run_id = start_run("algorithm_creation", run_id="test-timeout")
        with telemetry_scope(run_id=run_id, stage="hunter_round_1"):
            wrapped = PromptKeyLLMClient(_SlowClient(), "hunter_score")
            with pytest.raises(RuntimeError, match="hunter_score timed out after 0.0s"):
                await wrapped.complete("sys", "user")

        events = [
            ev for ev in get_event_log().events if ev.event_type == "PROMPT_DISPATCH_ERROR"
        ]
        assert len(events) == 1
        payload = events[0].payload
        assert payload["error_type"] == "TimeoutError"
        assert payload["provider_error_phase"] == "router_timeout"
        assert payload["prompt_timeout_s"] == pytest.approx(0.01)
        monkeypatch.delenv("AGEOM_HUNTER_SCORE_TIMEOUT_S")


class TestConfigPerPromptFields:
    def test_config_picks_up_per_prompt_env_vars(self, monkeypatch):
        monkeypatch.setenv("AGEOM_SYNTHESIZER_TACTIC_LLM_PROVIDER", "llama_cpp")
        monkeypatch.setenv("AGEOM_SYNTHESIZER_TACTIC_LLM_MODEL", "qwen-2.5-coder-32b")
        monkeypatch.setenv("AGEOM_INGESTER_CHUNK_LLM_PROVIDER", "codex")

        from ageom.config import AgeomConfig

        config = AgeomConfig(_env_file=None)
        assert config.synthesizer_tactic_llm_provider == "llama_cpp"
        assert config.synthesizer_tactic_llm_model == "qwen-2.5-coder-32b"
        assert config.ingester_chunk_llm_provider == "codex"
        # Non-overridden fields keep their code defaults (may be non-empty
        # now that some prompts route to local Ollama by default).
        assert isinstance(config.architect_strategy_llm_provider, str)
        assert isinstance(config.architect_strategy_llm_model, str)

    def test_all_per_prompt_fields_exist(self):
        from ageom.config import AgeomConfig

        config = AgeomConfig(_env_file=None)
        from ageom.llm_router import ALL_PROMPT_KEYS

        for key in ALL_PROMPT_KEYS:
            assert hasattr(config, f"{key}_llm_provider")
            assert hasattr(config, f"{key}_llm_model")

    def test_prompt_timeout_defaults_exist_for_hunter_score(self):
        assert prompt_timeout_seconds("hunter_score") == pytest.approx(20.0)

    def test_unbenchmarked_prompt_defaults_are_not_applied_implicitly(self):
        from ageom.config import AgeomConfig, prompt_override_matches_code_default, should_apply_prompt_override

        config = AgeomConfig(_env_file=None)
        assert AgeomConfig.model_fields["ingester_fix_type_llm_provider"].default == "llama_cpp"
        assert should_apply_prompt_override(config, "ingester_fix_type") is False
        assert prompt_override_matches_code_default(config, "ingester_fix_type") is True

    def test_explicit_override_for_unbenchmarked_prompt_is_still_applied(self):
        from ageom.config import AgeomConfig, should_apply_prompt_override

        config = AgeomConfig(
            _env_file=None,
            ingester_fix_type_llm_provider="codex_shim",
            ingester_fix_type_llm_model="gpt-5.3-codex",
        )
        assert should_apply_prompt_override(config, "ingester_fix_type") is True

    def test_structured_mode_suppresses_benchmarked_code_defaults(self):
        from ageom.config import AgeomConfig, should_apply_prompt_override

        config = AgeomConfig(_env_file=None, execution_mode="structured")
        assert should_apply_prompt_override(
            config, "hunter_score", execution_mode="structured"
        ) is False

    def test_verified_mode_suppresses_deterministic_code_defaults(self):
        from ageom.config import AgeomConfig, should_apply_prompt_override

        config = AgeomConfig(_env_file=None, execution_mode="verified")
        assert should_apply_prompt_override(
            config, "hunter_score", execution_mode="verified"
        ) is False
        assert should_apply_prompt_override(
            config, "architect_strategy", execution_mode="verified"
        ) is False


class TestCreateLLMRouter:
    def test_architect_strategy_and_critique_use_explicit_overrides(self, monkeypatch):
        from ageom.cli import _create_llm_router

        created: list[tuple[str, str]] = []

        def _fake_create_llm_client(*, provider, model, **kwargs):
            created.append((provider, model))
            client = MagicMock()
            client.provider = provider
            client.model = model
            return client

        monkeypatch.setattr("ageom.hunter.llm.create_llm_client", _fake_create_llm_client)

        args = SimpleNamespace(llm_provider=None, llm_model=None, llm_max_tokens=None)
        config = SimpleNamespace(
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
            architect_strategy_llm_provider="codex_shim",
            architect_strategy_llm_model="gpt-5.4-codex",
            architect_decompose_llm_provider="",
            architect_decompose_llm_model="",
            architect_critique_llm_provider="codex_shim",
            architect_critique_llm_model="gpt-5.4-codex",
        )

        router = _create_llm_router(
            args,
            config,
            "architect",
            ["architect_strategy", "architect_decompose", "architect_critique"],
        )

        assert isinstance(router, LLMRouter)
        assert created == [
            ("anthropic", "claude-sonnet-4-5-20250929"),
            ("codex_shim", "gpt-5.4-codex"),
        ]
        assert isinstance(router.for_prompt("architect_strategy"), StrategyClassifier)
        assert router.for_prompt("architect_critique") is not router.for_prompt("architect_strategy")
        assert isinstance(router.for_prompt("architect_decompose"), DeterministicDecomposer)

    def test_hunter_prompts_use_explicit_overrides(self, monkeypatch):
        from ageom.cli import _create_llm_router

        created: list[tuple[str, str]] = []

        def _fake_create_llm_client(*, provider, model, **kwargs):
            created.append((provider, model))
            client = MagicMock()
            client.provider = provider
            client.model = model
            return client

        monkeypatch.setattr("ageom.hunter.llm.create_llm_client", _fake_create_llm_client)

        args = SimpleNamespace(llm_provider=None, llm_model=None, llm_max_tokens=None)
        config = SimpleNamespace(
            llm_provider="anthropic",
            llm_model="claude-sonnet-4-5-20250929",
            llm_max_tokens=4096,
            anthropic_api_key="",
            openai_api_key="",
            openai_base_url="",
            llama_cpp_base_url="http://127.0.0.1:8080/v1",
            llama_cpp_api_key="local",
            use_agent_layer=False,
            hunter_llm_provider="",
            hunter_llm_model="",
            hunter_score_llm_provider="codex_shim",
            hunter_score_llm_model="gpt-5.4-codex",
            hunter_reformulate_llm_provider="gemini_shim",
            hunter_reformulate_llm_model="flash-exp",
            hunter_analyze_failure_llm_provider="gemini_shim",
            hunter_analyze_failure_llm_model="flash-pro",
        )

        router = _create_llm_router(
            args,
            config,
            "hunter",
            ["hunter_score", "hunter_reformulate", "hunter_analyze_failure"],
        )

        assert isinstance(router, LLMRouter)
        assert created == [
            ("anthropic", "claude-sonnet-4-5-20250929"),
            ("codex_shim", "gpt-5.4-codex"),
            ("gemini_shim", "flash-exp"),
            ("gemini_shim", "flash-pro"),
        ]
        assert isinstance(router.for_prompt("hunter_score"), HeuristicCandidateRanker)
        assert isinstance(
            router.for_prompt("hunter_reformulate"), HeuristicQueryReformulator
        )
        assert isinstance(
            router.for_prompt("hunter_analyze_failure"), DeterministicFailureAnalyzer
        )
        assert router.for_prompt("hunter_reformulate") is not router.for_prompt("hunter_score")
        assert router.for_prompt("hunter_analyze_failure") is not router.for_prompt("hunter_score")

    def test_architect_code_defaults_do_not_spawn_extra_override_clients(self, monkeypatch):
        from ageom.cli import _create_llm_router

        created: list[tuple[str, str]] = []

        def _fake_create_llm_client(*, provider, model, **kwargs):
            created.append((provider, model))
            client = MagicMock()
            client.provider = provider
            client.model = model
            return client

        monkeypatch.setattr("ageom.hunter.llm.create_llm_client", _fake_create_llm_client)

        args = SimpleNamespace(llm_provider=None, llm_model=None, llm_max_tokens=None)
        config = SimpleNamespace(
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
            architect_strategy_llm_provider="codex_shim",
            architect_strategy_llm_model="gpt-5.3-codex",
            architect_decompose_llm_provider="",
            architect_decompose_llm_model="",
            architect_critique_llm_provider="codex_shim",
            architect_critique_llm_model="gpt-5.3-codex",
        )

        router = _create_llm_router(
            args,
            config,
            "architect",
            ["architect_strategy", "architect_decompose", "architect_critique"],
        )

        assert created == [("anthropic", "claude-sonnet-4-5-20250929")]
        assert isinstance(router, LLMRouter)
        assert isinstance(router.for_prompt("architect_strategy"), StrategyClassifier)
        assert isinstance(router.for_prompt("architect_decompose"), DeterministicDecomposer)
        assert router.for_prompt("architect_critique") is not router.for_prompt("architect_strategy")

    def test_unbenchmarked_default_override_is_filtered_out(self, monkeypatch):
        from ageom.cli import _create_llm_router

        created: list[tuple[str, str]] = []

        def _fake_create_llm_client(*, provider, model, **kwargs):
            created.append((provider, model))
            client = MagicMock()
            client.provider = provider
            client.model = model
            return client

        monkeypatch.setattr("ageom.hunter.llm.create_llm_client", _fake_create_llm_client)

        args = SimpleNamespace(llm_provider=None, llm_model=None, llm_max_tokens=None)
        config = SimpleNamespace(
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
        )

        router = _create_llm_router(args, config, "ingester", ["ingester_fix_type"])

        assert router is not None
        assert created == [("anthropic", "claude-sonnet-4-5-20250929")]

    def test_rapid_mode_suppresses_benchmark_default_overrides(self, monkeypatch):
        from ageom.cli import _create_llm_router

        created: list[tuple[str, str]] = []

        def _fake_create_llm_client(*, provider, model, **kwargs):
            created.append((provider, model))
            client = MagicMock()
            client.provider = provider
            client.model = model
            return client

        monkeypatch.setattr("ageom.hunter.llm.create_llm_client", _fake_create_llm_client)

        args = SimpleNamespace(llm_provider=None, llm_model=None, llm_max_tokens=None, mode="rapid")
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
            hunter_llm_provider="",
            hunter_llm_model="",
            hunter_score_llm_provider="codex_shim",
            hunter_score_llm_model="gpt-5.3-codex",
        )

        router = _create_llm_router(args, config, "hunter", ["hunter_score"])

        assert router is not None
        assert created == [("anthropic", "claude-sonnet-4-5-20250929")]

    def test_structured_mode_suppresses_benchmark_default_overrides(self, monkeypatch):
        from ageom.cli import _create_llm_router

        created: list[tuple[str, str]] = []

        def _fake_create_llm_client(*, provider, model, **kwargs):
            created.append((provider, model))
            client = MagicMock()
            client.provider = provider
            client.model = model
            return client

        monkeypatch.setattr("ageom.hunter.llm.create_llm_client", _fake_create_llm_client)

        args = SimpleNamespace(
            llm_provider=None,
            llm_model=None,
            llm_max_tokens=None,
            mode="structured",
        )
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
            hunter_llm_provider="",
            hunter_llm_model="",
            hunter_score_llm_provider="codex_shim",
            hunter_score_llm_model="gpt-5.3-codex",
        )

        router = _create_llm_router(args, config, "hunter", ["hunter_score"])

        assert router is not None
        assert created == [("anthropic", "claude-sonnet-4-5-20250929")]


class TestPromptRoutingSummary:
    def test_summary_reports_active_and_suppressed_overrides(self, capsys):
        from ageom.cli import _print_prompt_routing_summary, _summarize_prompt_routing
        from ageom.config import AgeomConfig

        config = AgeomConfig(_env_file=None)
        summary = _summarize_prompt_routing(
            config,
            "ingester",
            ["ingester_fix_type", "ingester_chunk"],
        )

        assert summary["default_provider"] == "anthropic"
        assert summary["suppressed_default_overrides"] == ["ingester_fix_type"]
        assert summary["active_overrides"] == []

        _print_prompt_routing_summary(
            config,
            "ingester",
            ["ingester_fix_type", "ingester_chunk"],
        )
        out = capsys.readouterr().out
        assert "LLM routing (ingester)" in out
        assert "suppressed_defaults=[ingester_fix_type]" in out

    def test_summary_reports_custom_nonbenchmark_override(self):
        from ageom.cli import _routing_metadata_summary, _summarize_prompt_routing
        from ageom.config import AgeomConfig

        config = AgeomConfig(
            _env_file=None,
            ingester_fix_type_llm_provider="codex_shim",
            ingester_fix_type_llm_model="gpt-5.3-codex",
        )
        summary = _summarize_prompt_routing(
            config,
            "ingester",
            ["ingester_fix_type"],
        )

        assert summary["custom_nonbenchmark_overrides"] == ["ingester_fix_type"]
        assert summary["active_overrides"] == [
            {
                "prompt_key": "ingester_fix_type",
                "provider": "codex_shim",
                "model": "gpt-5.3-codex",
            }
        ]
        metadata = _routing_metadata_summary(summary)
        assert metadata["custom_nonbenchmark_overrides"] == ["ingester_fix_type"]
        assert metadata["active_overrides"][0]["provider"] == "codex_shim"

    def test_rapid_mode_summary_suppresses_benchmark_defaults(self):
        from ageom.cli import _summarize_prompt_routing
        from ageom.config import AgeomConfig

        config = AgeomConfig(_env_file=None)
        summary = _summarize_prompt_routing(
            config,
            "hunter",
            ["hunter_score", "hunter_reformulate", "hunter_analyze_failure"],
            "rapid",
        )

        assert summary["mode"] == "rapid"
        assert summary["default_provider"] == "anthropic"
        assert summary["default_model"] == "claude-sonnet-4-5-20250929"
        assert summary["active_overrides"] == []
        assert summary["suppressed_default_overrides"] == [
            "hunter_score",
            "hunter_reformulate",
            "hunter_analyze_failure",
        ]

    def test_structured_mode_summary_suppresses_benchmark_defaults(self):
        from ageom.cli import _summarize_prompt_routing
        from ageom.config import AgeomConfig

        config = AgeomConfig(_env_file=None)
        summary = _summarize_prompt_routing(
            config,
            "hunter",
            ["hunter_score", "hunter_reformulate", "hunter_analyze_failure"],
            "structured",
        )

        assert summary["mode"] == "structured"
        assert summary["default_provider"] == "anthropic"
        assert summary["default_model"] == "claude-sonnet-4-5-20250929"
        assert summary["active_overrides"] == []
        assert summary["suppressed_default_overrides"] == [
            "hunter_score",
            "hunter_reformulate",
            "hunter_analyze_failure",
        ]


# ---------------------------------------------------------------------------
# SubprocessCLIClient tests
# ---------------------------------------------------------------------------

from ageom.hunter.llm import SubprocessCLIClient, create_llm_client


def _fake_proc(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
    """Return an AsyncMock that behaves like asyncio.subprocess.Process."""
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    return proc


class TestSubprocessCLIClient:
    # -- command construction ------------------------------------------------

    def test_claude_cmd_basic(self):
        c = SubprocessCLIClient(cli="claude", model="sonnet", max_tokens=2048)
        cmd = c._build_cmd("You are helpful.")
        assert cmd == [
            "claude", "-p", "--output-format", "text",
            "--system-prompt", "You are helpful.",
            "--model", "sonnet",
        ]
        assert "--max-tokens" not in cmd

    def test_claude_cmd_no_system(self):
        c = SubprocessCLIClient(cli="claude", model="sonnet", max_tokens=1024)
        cmd = c._build_cmd("")
        assert "--system-prompt" not in cmd

    def test_codex_cmd_basic(self):
        c = SubprocessCLIClient(cli="codex", model="o4-mini", max_tokens=4096)
        cmd = c._build_cmd("sys")
        assert cmd[:2] == ["codex", "exec"]
        assert "--json" in cmd
        assert "-o" not in cmd
        assert "-m" in cmd
        assert "o4-mini" in cmd

    def test_gemini_cmd_basic(self):
        c = SubprocessCLIClient(cli="gemini", model="gemini-2.5-pro", max_tokens=4096)
        cmd = c._build_cmd("sys")
        assert cmd[0] == "gemini"
        assert "-p" in cmd
        assert "--model" in cmd
        assert "gemini-2.5-pro" in cmd

    def test_agent_layer_prefix(self):
        c = SubprocessCLIClient(
            cli="claude", model="sonnet", max_tokens=1024, use_agent_layer=True,
        )
        cmd = c._build_cmd("sys")
        assert cmd[0] == "al"
        assert cmd[1] == "claude"

    def test_agent_layer_codex(self):
        c = SubprocessCLIClient(
            cli="codex", model="o4-mini", max_tokens=1024, use_agent_layer=True,
        )
        cmd = c._build_cmd("")
        assert cmd[0] == "al"
        assert cmd[1] == "codex"

    def test_agent_layer_gemini(self):
        c = SubprocessCLIClient(
            cli="gemini", model="pro", max_tokens=1024, use_agent_layer=True,
        )
        cmd = c._build_cmd("")
        assert cmd[0] == "al"
        assert cmd[1] == "gemini"

    # -- stdin content -------------------------------------------------------

    def test_claude_stdin_is_user_only(self):
        c = SubprocessCLIClient(cli="claude", model="m", max_tokens=1)
        assert c._build_stdin("system text", "user text") == "user text"

    def test_codex_stdin_prepends_system(self):
        c = SubprocessCLIClient(cli="codex", model="m", max_tokens=1)
        result = c._build_stdin("sys", "usr")
        assert result == "[System]\nsys\n\nusr"

    def test_gemini_stdin_prepends_system(self):
        c = SubprocessCLIClient(cli="gemini", model="m", max_tokens=1)
        result = c._build_stdin("sys", "usr")
        assert result == "[System]\nsys\n\nusr"

    def test_codex_stdin_no_system(self):
        c = SubprocessCLIClient(cli="codex", model="m", max_tokens=1)
        assert c._build_stdin("", "usr") == "usr"

    # -- output parsing ------------------------------------------------------

    def test_claude_parse_output_plain(self):
        c = SubprocessCLIClient(cli="claude", model="m", max_tokens=1)
        assert c._parse_output("  hello world  \n") == "hello world"

    def test_gemini_parse_output_plain(self):
        c = SubprocessCLIClient(cli="gemini", model="m", max_tokens=1)
        assert c._parse_output("  result\n") == "result"

    def test_codex_parse_output_jsonl(self):
        c = SubprocessCLIClient(cli="codex", model="m", max_tokens=1)
        events = "\n".join([
            json.dumps({"role": "user", "content": "hi"}),
            json.dumps({"role": "assistant", "content": "hello back"}),
        ])
        assert c._parse_output(events) == "hello back"

    def test_codex_parse_output_message_field(self):
        c = SubprocessCLIClient(cli="codex", model="m", max_tokens=1)
        events = json.dumps({"message": "done"})
        assert c._parse_output(events) == "done"

    def test_codex_parse_output_item_completed_agent_message(self):
        c = SubprocessCLIClient(cli="codex", model="m", max_tokens=1)
        events = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "abc"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "parsed text"},
                    }
                ),
            ]
        )
        assert c._parse_output(events) == "parsed text"

    def test_codex_parse_output_fallback_raw(self):
        c = SubprocessCLIClient(cli="codex", model="m", max_tokens=1)
        assert c._parse_output("plain text") == "plain text"

    # -- async complete ------------------------------------------------------

    @pytest.mark.asyncio
    async def test_complete_success_claude(self):
        c = SubprocessCLIClient(cli="claude", model="sonnet", max_tokens=1024)
        proc = _fake_proc(stdout=b"reply text\n", returncode=0)
        with patch("ageom.hunter.llm.asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            result = await c.complete("sys", "usr")
        assert result == "reply text"
        # Verify stdin was just the user prompt
        proc.communicate.assert_awaited_once()
        stdin_bytes = proc.communicate.call_args[0][0]
        assert stdin_bytes == b"usr"

    @pytest.mark.asyncio
    async def test_complete_success_codex_jsonl(self):
        c = SubprocessCLIClient(cli="codex", model="o4-mini", max_tokens=1024)
        jsonl = json.dumps({"role": "assistant", "content": "answer"}) + "\n"
        proc = _fake_proc(stdout=jsonl.encode(), returncode=0)
        with patch("ageom.hunter.llm.asyncio.create_subprocess_exec", return_value=proc):
            result = await c.complete("sys", "usr")
        assert result == "answer"

    @pytest.mark.asyncio
    async def test_complete_success_gemini(self):
        c = SubprocessCLIClient(cli="gemini", model="pro", max_tokens=1024)
        proc = _fake_proc(stdout=b"gemini reply", returncode=0)
        with patch("ageom.hunter.llm.asyncio.create_subprocess_exec", return_value=proc):
            result = await c.complete("sys", "usr")
        assert result == "gemini reply"

    @pytest.mark.asyncio
    async def test_complete_nonzero_exit_raises(self):
        c = SubprocessCLIClient(cli="claude", model="m", max_tokens=1)
        proc = _fake_proc(stderr=b"some error", returncode=1)
        with patch("ageom.hunter.llm.asyncio.create_subprocess_exec", return_value=proc):
            with pytest.raises(RuntimeError, match="claude CLI exited with code 1"):
                await c.complete("sys", "usr")

    @pytest.mark.asyncio
    async def test_complete_with_grammar_delegates(self):
        c = SubprocessCLIClient(cli="claude", model="m", max_tokens=1)
        proc = _fake_proc(stdout=b"ok", returncode=0)
        with patch("ageom.hunter.llm.asyncio.create_subprocess_exec", return_value=proc):
            result = await c.complete_with_grammar("sys", "usr", "grammar")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_complete_retries_after_timeout(self):
        c = SubprocessCLIClient(cli="claude", model="sonnet", max_tokens=1024)
        c._max_retries = 1
        c._retry_backoff_s = 0.0

        proc1 = _fake_proc(returncode=None)
        proc1.kill = MagicMock()
        proc1.wait = AsyncMock(return_value=None)
        proc2 = _fake_proc(stdout=b"recovered", returncode=0)
        calls = {"count": 0}

        async def fake_wait_for(awaitable, timeout):
            calls["count"] += 1
            if calls["count"] == 1:
                close = getattr(awaitable, "close", None)
                if callable(close):
                    close()
                raise asyncio.TimeoutError()
            return await awaitable

        with patch(
            "ageom.hunter.llm.asyncio.create_subprocess_exec",
            side_effect=[proc1, proc2],
        ), patch(
            "ageom.hunter.llm.asyncio.wait_for",
            new=fake_wait_for,
        ), patch(
            "ageom.hunter.llm.asyncio.sleep",
            new=AsyncMock(return_value=None),
        ):
            result = await c.complete("sys", "usr")

        assert result == "recovered"
        assert proc1.kill.called


class TestCreateLLMClientCLIProviders:
    def test_cli_providers_rejected_by_default(self):
        with pytest.raises(ValueError, match="disabled by default because it uses the legacy one-shot subprocess path"):
            create_llm_client(
                provider="claude_cli", model="sonnet", max_tokens=1024,
            )

    @pytest.mark.filterwarnings("ignore:Provider '.*_cli' uses the legacy one-shot subprocess path:DeprecationWarning")
    def test_claude_cli_creates_subprocess_client(self):
        client = create_llm_client(
            provider="claude_cli", model="sonnet", max_tokens=1024,
            allow_legacy_subprocess=True,
        )
        assert isinstance(client, SubprocessCLIClient)
        assert client._cli == "claude"

    @pytest.mark.filterwarnings("ignore:Provider '.*_cli' uses the legacy one-shot subprocess path:DeprecationWarning")
    def test_codex_cli_creates_subprocess_client(self):
        client = create_llm_client(
            provider="codex_cli", model="o4-mini", max_tokens=1024,
            allow_legacy_subprocess=True,
        )
        assert isinstance(client, SubprocessCLIClient)
        assert client._cli == "codex"
        assert client._model == "o4-mini"

    @pytest.mark.filterwarnings("ignore:Provider '.*_cli' uses the legacy one-shot subprocess path:DeprecationWarning")
    def test_gemini_cli_creates_subprocess_client(self):
        client = create_llm_client(
            provider="gemini_cli", model="pro", max_tokens=1024,
            allow_legacy_subprocess=True,
        )
        assert isinstance(client, SubprocessCLIClient)
        assert client._cli == "gemini"
        assert client._model == "pro"

    @pytest.mark.filterwarnings("ignore:Provider '.*_cli' uses the legacy one-shot subprocess path:DeprecationWarning")
    def test_codex_cli_falls_back_from_claude_model(self):
        client = create_llm_client(
            provider="codex_cli", model="claude-sonnet-4-5-20250929", max_tokens=1024,
            allow_legacy_subprocess=True,
        )
        assert isinstance(client, SubprocessCLIClient)
        assert client._cli == "codex"
        assert client._model == "gpt-5.3-codex"

    @pytest.mark.filterwarnings("ignore:Provider '.*_cli' uses the legacy one-shot subprocess path:DeprecationWarning")
    def test_claude_cli_falls_back_from_codex_model(self):
        client = create_llm_client(
            provider="claude_cli", model="o4-mini", max_tokens=1024,
            allow_legacy_subprocess=True,
        )
        assert isinstance(client, SubprocessCLIClient)
        assert client._cli == "claude"
        assert client._model == "sonnet"

    @pytest.mark.filterwarnings("ignore:Provider '.*_cli' uses the legacy one-shot subprocess path:DeprecationWarning")
    def test_gemini_cli_falls_back_from_codex_model(self):
        client = create_llm_client(
            provider="gemini_cli", model="o4-mini", max_tokens=1024,
            allow_legacy_subprocess=True,
        )
        assert isinstance(client, SubprocessCLIClient)
        assert client._cli == "gemini"
        assert client._model == "gemini-2.5-pro"

    @pytest.mark.filterwarnings("ignore:Provider '.*_cli' uses the legacy one-shot subprocess path:DeprecationWarning")
    def test_codex_cli_invalid_model_raises(self):
        with pytest.raises(ValueError, match="not codex-compatible"):
            create_llm_client(
                provider="codex_cli", model="not-a-real-model", max_tokens=1024,
                allow_legacy_subprocess=True,
            )

    @pytest.mark.filterwarnings("ignore:Provider '.*_cli' uses the legacy one-shot subprocess path:DeprecationWarning")
    def test_agent_layer_forwarded(self):
        client = create_llm_client(
            provider="claude_cli", model="m", max_tokens=1, use_agent_layer=True,
            allow_legacy_subprocess=True,
        )
        assert isinstance(client, SubprocessCLIClient)
        assert client._use_agent_layer is True

    def test_unknown_cli_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown CLI variant"):
            SubprocessCLIClient(cli="unknown", model="m", max_tokens=1)
