from __future__ import annotations

from ageom.config import AgeomConfig, resolve_execution_mode


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
