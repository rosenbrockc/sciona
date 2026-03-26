"""LLM client and router helpers for CLI command handlers."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sciona.config import AgeomConfig
    from sciona.hunter.llm import LLMClient
    from sciona.indexer.embedder import Embedder


def _create_llm(
    args: argparse.Namespace, config: "AgeomConfig", round_name: str
) -> "LLMClient":
    """Create an LLM client with per-round provider/model overrides."""
    from sciona.config import effective_round_provider_model
    from sciona.hunter.llm import create_llm_client

    max_tokens_attr = f"{round_name}_llm_max_tokens" if round_name == "hunter" else None
    execution_mode = str(
        getattr(args, "mode", None)
        or getattr(config, "execution_mode", "verified")
        or "verified"
    ).strip().lower()
    default_provider, default_model = effective_round_provider_model(
        config,
        round_name,
        execution_mode=execution_mode,
    )

    llm_provider = getattr(args, "llm_provider", None) or default_provider
    llm_model = getattr(args, "llm_model", None) or default_model
    llm_max_tokens = (
        getattr(args, "llm_max_tokens", None)
        or (getattr(config, max_tokens_attr, None) if max_tokens_attr else None)
        or config.llm_max_tokens
    )

    return create_llm_client(
        provider=llm_provider,
        model=llm_model,
        max_tokens=llm_max_tokens,
        anthropic_api_key=config.anthropic_api_key,
        openai_api_key=config.openai_api_key,
        openai_base_url=config.openai_base_url,
        llama_cpp_base_url=config.llama_cpp_base_url,
        llama_cpp_api_key=config.llama_cpp_api_key,
        use_agent_layer=config.use_agent_layer,
        allow_legacy_subprocess=getattr(
            config, "allow_legacy_subprocess_providers", False
        ),
    )


def _create_llm_router(
    args: argparse.Namespace,
    config: "AgeomConfig",
    round_name: str,
    prompt_keys: list[str],
    *,
    embedder: "Embedder | None" = None,
) -> "LLMClient":
    """Create an ``LLMRouter`` wrapping the default client with per-prompt overrides."""
    from sciona.architect.deterministic_critic import DeterministicCritic
    from sciona.architect.deterministic_decompose import DeterministicDecomposer
    from sciona.architect.strategy_classifier import StrategyClassifier
    from sciona.config import should_apply_prompt_override
    from sciona.hunter.candidate_ranker import HeuristicCandidateRanker
    from sciona.hunter.embedding_reranker import EmbeddingReranker
    from sciona.hunter.failure_analyzer import DeterministicFailureAnalyzer
    from sciona.hunter.llm import create_llm_client
    from sciona.hunter.query_reformulator import HeuristicQueryReformulator
    from sciona.ingester.ast_state_hoister import ASTStateHoister
    from sciona.ingester.deterministic_cycle_breaker import DeterministicCycleBreaker
    from sciona.ingester.deterministic_ghost_fixer import DeterministicGhostFixer
    from sciona.ingester.deterministic_type_fixer import DeterministicTypeFixer
    from sciona.ingester.template_abstractor import TemplateAbstractor
    from sciona.ingester.template_witness_generator import TemplateWitnessGenerator
    from sciona.llm_router import (
        ARCHITECT_CRITIQUE,
        ARCHITECT_DECOMPOSE,
        ARCHITECT_STRATEGY,
        HUNTER_ANALYZE_FAILURE,
        HUNTER_REFORMULATE,
        HUNTER_SCORE,
        INGESTER_ABSTRACT,
        INGESTER_FIX_GHOST,
        INGESTER_FIX_MESSAGE_CYCLE,
        INGESTER_FIX_TYPE,
        INGESTER_HOIST_STATE,
        INGESTER_OPAQUE_WITNESS,
        LLMRouter,
        SYNTHESIZER_TACTIC,
    )
    from sciona.synthesizer.tactic_suggester import DeterministicTacticSuggester

    default = _create_llm(args, config, round_name)
    overrides: dict[str, "LLMClient"] = {}
    client_cache: dict[tuple[str, str], "LLMClient"] = {}
    execution_mode = str(
        getattr(args, "mode", None)
        or getattr(config, "execution_mode", "verified")
        or "verified"
    ).strip().lower()

    for key in prompt_keys:
        provider = getattr(config, f"{key}_llm_provider", "")
        model = getattr(config, f"{key}_llm_model", "")
        if not provider:
            continue
        if not should_apply_prompt_override(config, key, execution_mode):
            continue
        if not model:
            model = config.llm_model

        cache_key = (provider, model)
        if cache_key not in client_cache:
            client_cache[cache_key] = create_llm_client(
                provider=provider,
                model=model,
                max_tokens=config.llm_max_tokens,
                anthropic_api_key=config.anthropic_api_key,
                openai_api_key=config.openai_api_key,
                openai_base_url=config.openai_base_url,
                llama_cpp_base_url=config.llama_cpp_base_url,
                llama_cpp_api_key=config.llama_cpp_api_key,
                use_agent_layer=config.use_agent_layer,
                allow_legacy_subprocess=getattr(
                    config, "allow_legacy_subprocess_providers", False
                ),
            )
        overrides[key] = client_cache[cache_key]

    if round_name == "architect" and ARCHITECT_STRATEGY in prompt_keys:
        strategy_fallback = overrides.get(ARCHITECT_STRATEGY, default)
        overrides[ARCHITECT_STRATEGY] = StrategyClassifier(strategy_fallback)
    if round_name == "architect" and ARCHITECT_DECOMPOSE in prompt_keys:
        decompose_fallback = overrides.get(ARCHITECT_DECOMPOSE, default)
        overrides[ARCHITECT_DECOMPOSE] = DeterministicDecomposer(decompose_fallback)
    if round_name == "architect" and ARCHITECT_CRITIQUE in prompt_keys:
        critique_fallback = overrides.get(ARCHITECT_CRITIQUE, default)
        overrides[ARCHITECT_CRITIQUE] = DeterministicCritic(critique_fallback)
    if round_name == "hunter" and HUNTER_SCORE in prompt_keys:
        score_fallback = overrides.get(HUNTER_SCORE, default)
        heuristic_ranker = HeuristicCandidateRanker(score_fallback)
        if embedder is not None:
            overrides[HUNTER_SCORE] = EmbeddingReranker(embedder, heuristic_ranker)
        else:
            overrides[HUNTER_SCORE] = heuristic_ranker
    if round_name == "hunter" and HUNTER_ANALYZE_FAILURE in prompt_keys:
        analyze_fallback = overrides.get(HUNTER_ANALYZE_FAILURE, default)
        overrides[HUNTER_ANALYZE_FAILURE] = DeterministicFailureAnalyzer(
            analyze_fallback
        )
    if round_name == "hunter" and HUNTER_REFORMULATE in prompt_keys:
        reformulate_fallback = overrides.get(HUNTER_REFORMULATE, default)
        query_expander = None
        if embedder is not None and config.hunter_embedding_query_expander:
            from sciona.hunter.embedding_query_expander import EmbeddingQueryExpander
            from sciona.indexer.faiss_store import FAISSStore

            try:
                store = FAISSStore.load(config.index_dir)
                decls = (
                    list(store._declarations.values())
                    if hasattr(store, "_declarations")
                    and isinstance(store._declarations, dict)
                    else []
                )
                if decls:
                    query_expander = EmbeddingQueryExpander(embedder, decls)
            except Exception:
                pass
        overrides[HUNTER_REFORMULATE] = HeuristicQueryReformulator(
            reformulate_fallback,
            query_expander=query_expander,
        )
    if round_name == "ingester" and INGESTER_FIX_TYPE in prompt_keys:
        fix_type_fallback = overrides.get(INGESTER_FIX_TYPE, default)
        overrides[INGESTER_FIX_TYPE] = DeterministicTypeFixer(fix_type_fallback)
    if round_name == "ingester" and INGESTER_FIX_GHOST in prompt_keys:
        ghost_fallback = overrides.get(INGESTER_FIX_GHOST, default)
        overrides[INGESTER_FIX_GHOST] = DeterministicGhostFixer(ghost_fallback)
    if round_name == "ingester" and INGESTER_FIX_MESSAGE_CYCLE in prompt_keys:
        cycle_fallback = overrides.get(INGESTER_FIX_MESSAGE_CYCLE, default)
        overrides[INGESTER_FIX_MESSAGE_CYCLE] = DeterministicCycleBreaker(
            cycle_fallback
        )
    if round_name == "ingester" and INGESTER_OPAQUE_WITNESS in prompt_keys:
        opaque_fallback = overrides.get(INGESTER_OPAQUE_WITNESS, default)
        overrides[INGESTER_OPAQUE_WITNESS] = TemplateWitnessGenerator(opaque_fallback)
    if round_name == "ingester" and INGESTER_ABSTRACT in prompt_keys:
        abstract_fallback = overrides.get(INGESTER_ABSTRACT, default)
        overrides[INGESTER_ABSTRACT] = TemplateAbstractor(abstract_fallback)
    if round_name == "ingester" and INGESTER_HOIST_STATE in prompt_keys:
        hoist_fallback = overrides.get(INGESTER_HOIST_STATE, default)
        overrides[INGESTER_HOIST_STATE] = ASTStateHoister(hoist_fallback)
    if round_name == "synthesizer" and SYNTHESIZER_TACTIC in prompt_keys:
        tactic_fallback = overrides.get(SYNTHESIZER_TACTIC, default)
        overrides[SYNTHESIZER_TACTIC] = DeterministicTacticSuggester(
            tactic_fallback
        )

    if not overrides:
        return default
    return LLMRouter(default=default, overrides=overrides)


async def _warm_llm_if_supported(llm: object, label: str) -> None:
    """Prewarm clients that expose a warmup hook so failures surface early."""
    warm = getattr(llm, "warmup", None)
    if not callable(warm):
        return
    print(f"Prewarming {label} LLM clients...")
    await warm()
