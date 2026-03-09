from __future__ import annotations

from ageom.cli import _mode_feature_summary
from ageom.config import AgeomConfig, effective_round_provider_model, resolve_execution_mode


def test_verified_mode_preserves_existing_feature_flags(monkeypatch):
    monkeypatch.setenv("AGEOM_EXECUTION_MODE", "verified")
    monkeypatch.setenv("AGEOM_GRAPH_RETRIEVAL_ENABLED", "true")
    monkeypatch.setenv("AGEOM_ARCHITECT_SHARED_CONTEXT_ENABLED", "true")
    monkeypatch.setenv("AGEOM_HUNTER_SHARED_CONTEXT_ENABLED", "true")
    monkeypatch.setenv("AGEOM_HUNTER_MODE", "speculative_local")
    monkeypatch.setenv("AGEOM_HUNTER_USE_GBNF", "true")

    config = AgeomConfig()
    mode = resolve_execution_mode(config)

    assert mode.mode == "verified"
    assert mode.skill_index_enabled is True
    assert mode.graph_retrieval_enabled is True
    assert mode.architect_shared_context_enabled is True
    assert mode.hunter_shared_context_enabled is True
    assert mode.hunter_mode == "speculative_local"
    assert mode.hunter_use_gbnf is True
    assert mode.semantic_index_backend_override is None


def test_structured_mode_disables_heavier_optional_features(monkeypatch):
    monkeypatch.setenv("AGEOM_EXECUTION_MODE", "structured")
    monkeypatch.setenv("AGEOM_GRAPH_RETRIEVAL_ENABLED", "true")
    monkeypatch.setenv("AGEOM_ARCHITECT_SHARED_CONTEXT_ENABLED", "true")
    monkeypatch.setenv("AGEOM_HUNTER_SHARED_CONTEXT_ENABLED", "true")
    monkeypatch.setenv("AGEOM_HUNTER_MODE", "speculative_local")
    monkeypatch.setenv("AGEOM_HUNTER_USE_GBNF", "true")

    config = AgeomConfig()
    mode = resolve_execution_mode(config)

    assert mode.mode == "structured"
    assert mode.skill_index_enabled is True
    assert mode.graph_retrieval_enabled is False
    assert mode.architect_shared_context_enabled is False
    assert mode.hunter_shared_context_enabled is False
    assert mode.synthesizer_shared_context_enabled is False
    assert mode.ingester_shared_context_enabled is False
    assert mode.hunter_mode == "standard"
    assert mode.hunter_use_gbnf is True
    assert mode.semantic_index_backend_override is None


def test_rapid_mode_uses_lexical_and_disables_shared_features(monkeypatch):
    monkeypatch.setenv("AGEOM_EXECUTION_MODE", "rapid")
    monkeypatch.setenv("AGEOM_HUNTER_USE_GBNF", "true")
    monkeypatch.setenv("AGEOM_HUNTER_MODE", "speculative_local")

    config = AgeomConfig()
    mode = resolve_execution_mode(config)

    assert mode.mode == "rapid"
    assert mode.skill_index_enabled is False
    assert mode.graph_retrieval_enabled is False
    assert mode.architect_shared_context_enabled is False
    assert mode.hunter_shared_context_enabled is False
    assert mode.synthesizer_shared_context_enabled is False
    assert mode.ingester_shared_context_enabled is False
    assert mode.hunter_mode == "standard"
    assert mode.hunter_use_gbnf is False
    assert mode.semantic_index_backend_override == "lexical"


def test_mode_feature_summary_renders_expected_flags(monkeypatch):
    monkeypatch.setenv("AGEOM_EXECUTION_MODE", "rapid")
    config = AgeomConfig()
    mode = resolve_execution_mode(config)

    summary = _mode_feature_summary(mode)

    assert summary["mode"] == "rapid"
    assert summary["skill_index"] == "off"
    assert summary["graph_retrieval"] == "off"
    assert summary["hunter_mode"] == "standard"
    assert summary["hunter_gbnf"] == "off"
    assert summary["semantic_backend"] == "lexical"


def test_rapid_mode_suppresses_round_defaults_but_keeps_explicit_round_override():
    config = AgeomConfig(
        _env_file=None,
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-5-20250929",
        hunter_llm_provider="llama_cpp",
        hunter_llm_model="qwen2.5-coder:7b",
    )

    provider, model = effective_round_provider_model(
        config,
        "hunter",
        execution_mode="rapid",
    )
    assert (provider, model) == ("anthropic", "claude-sonnet-4-5-20250929")

    explicit = AgeomConfig(
        _env_file=None,
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-5-20250929",
        hunter_llm_provider="codex_shim",
        hunter_llm_model="gpt-5.3-codex",
    )
    provider, model = effective_round_provider_model(
        explicit,
        "hunter",
        execution_mode="rapid",
    )
    assert (provider, model) == ("codex_shim", "gpt-5.3-codex")
