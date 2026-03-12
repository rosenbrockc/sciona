"""Shared utilities for CLI command handlers."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ageom.config import AgeomConfig, resolve_execution_mode
    from ageom.hunter.llm import LLMClient
    from ageom.indexer.embedder import Embedder
    from ageom.protocols import ProofEnvironment, SemanticIndex
    from ageom.shared_context import SharedContextMetrics, SharedContextStore
    from ageom.types import Prover


def _create_llm(
    args: argparse.Namespace, config: "AgeomConfig", round_name: str
) -> "LLMClient":
    """Create an LLM client with per-round provider/model overrides.

    Args:
        args: Parsed CLI arguments (may have llm_provider, llm_model, llm_max_tokens).
        config: The AgeomConfig instance.
        round_name: One of "architect", "hunter", "synthesizer" to select per-round overrides.
    """
    from ageom.hunter.llm import create_llm_client

    max_tokens_attr = f"{round_name}_llm_max_tokens" if round_name == "hunter" else None
    from ageom.config import effective_round_provider_model

    execution_mode = str(
        getattr(args, "mode", None) or getattr(config, "execution_mode", "verified") or "verified"
    ).strip().lower()
    default_provider, default_model = effective_round_provider_model(
        config,
        round_name,
        execution_mode=execution_mode,
    )

    llm_provider = (
        getattr(args, "llm_provider", None)
        or default_provider
    )
    llm_model = (
        getattr(args, "llm_model", None)
        or default_model
    )
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
    """Create an ``LLMRouter`` wrapping the default client with per-prompt overrides.

    For each *prompt_key* in *prompt_keys*, if the config has a non-empty
    ``{prompt_key}_llm_provider``, a dedicated ``LLMClient`` is created.
    Clients with matching (provider, model) pairs are deduplicated.
    """
    from ageom.config import should_apply_prompt_override
    from ageom.architect.strategy_classifier import StrategyClassifier
    from ageom.hunter.candidate_ranker import HeuristicCandidateRanker
    from ageom.hunter.embedding_reranker import EmbeddingReranker
    from ageom.hunter.failure_analyzer import DeterministicFailureAnalyzer
    from ageom.hunter.query_reformulator import HeuristicQueryReformulator
    from ageom.ingester.ast_state_hoister import ASTStateHoister
    from ageom.ingester.deterministic_ghost_fixer import DeterministicGhostFixer
    from ageom.ingester.deterministic_type_fixer import DeterministicTypeFixer
    from ageom.hunter.llm import create_llm_client
    from ageom.llm_router import (
        ARCHITECT_STRATEGY,
        HUNTER_ANALYZE_FAILURE,
        HUNTER_REFORMULATE,
        HUNTER_SCORE,
        INGESTER_FIX_GHOST,
        INGESTER_FIX_TYPE,
        INGESTER_HOIST_STATE,
        LLMRouter,
        SYNTHESIZER_TACTIC,
    )
    from ageom.synthesizer.tactic_suggester import DeterministicTacticSuggester

    default = _create_llm(args, config, round_name)
    overrides: dict[str, "LLMClient"] = {}
    # Cache by (provider, model) to avoid redundant connections
    client_cache: dict[tuple[str, str], "LLMClient"] = {}
    execution_mode = str(
        getattr(args, "mode", None) or getattr(config, "execution_mode", "verified") or "verified"
    ).strip().lower()

    for key in prompt_keys:
        provider = getattr(config, f"{key}_llm_provider", "")
        model = getattr(config, f"{key}_llm_model", "")
        if not provider:
            continue
        if not should_apply_prompt_override(config, key, execution_mode):
            continue
        # Fall back model to global if only provider is set
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
    if round_name == "hunter" and HUNTER_SCORE in prompt_keys:
        score_fallback = overrides.get(HUNTER_SCORE, default)
        heuristic_ranker = HeuristicCandidateRanker(score_fallback)
        if embedder is not None:
            overrides[HUNTER_SCORE] = EmbeddingReranker(embedder, heuristic_ranker)
        else:
            overrides[HUNTER_SCORE] = heuristic_ranker
    if round_name == "hunter" and HUNTER_ANALYZE_FAILURE in prompt_keys:
        analyze_fallback = overrides.get(HUNTER_ANALYZE_FAILURE, default)
        overrides[HUNTER_ANALYZE_FAILURE] = DeterministicFailureAnalyzer(analyze_fallback)
    if round_name == "hunter" and HUNTER_REFORMULATE in prompt_keys:
        reformulate_fallback = overrides.get(HUNTER_REFORMULATE, default)
        overrides[HUNTER_REFORMULATE] = HeuristicQueryReformulator(
            reformulate_fallback
        )
    if round_name == "ingester" and INGESTER_FIX_TYPE in prompt_keys:
        fix_type_fallback = overrides.get(INGESTER_FIX_TYPE, default)
        overrides[INGESTER_FIX_TYPE] = DeterministicTypeFixer(fix_type_fallback)
    if round_name == "ingester" and INGESTER_FIX_GHOST in prompt_keys:
        ghost_fallback = overrides.get(INGESTER_FIX_GHOST, default)
        overrides[INGESTER_FIX_GHOST] = DeterministicGhostFixer(ghost_fallback)
    if round_name == "ingester" and INGESTER_HOIST_STATE in prompt_keys:
        hoist_fallback = overrides.get(INGESTER_HOIST_STATE, default)
        overrides[INGESTER_HOIST_STATE] = ASTStateHoister(hoist_fallback)
    if round_name == "synthesizer" and SYNTHESIZER_TACTIC in prompt_keys:
        tactic_fallback = overrides.get(SYNTHESIZER_TACTIC, default)
        overrides[SYNTHESIZER_TACTIC] = DeterministicTacticSuggester(tactic_fallback)

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


def _summarize_prompt_routing(
    config: "AgeomConfig",
    round_name: str,
    prompt_keys: list[str],
    execution_mode: str | None = None,
) -> dict[str, Any]:
    """Summarize which prompt-key overrides are active vs suppressed."""
    from ageom.config import (
        effective_round_provider_model,
        BENCHMARK_JUSTIFIED_PROMPT_KEYS,
        prompt_override_matches_code_default,
        should_apply_prompt_override,
    )

    resolved_mode = str(
        execution_mode or getattr(config, "execution_mode", "verified") or "verified"
    ).strip().lower()
    default_provider, default_model = effective_round_provider_model(
        config,
        round_name,
        execution_mode=resolved_mode,
    )
    active_overrides: list[dict[str, str]] = []
    suppressed_defaults: list[str] = []
    custom_nonbenchmark: list[str] = []

    for key in prompt_keys:
        provider = getattr(config, f"{key}_llm_provider", "")
        model = getattr(config, f"{key}_llm_model", "")
        if not provider:
            continue
        if should_apply_prompt_override(config, key, resolved_mode):
            model = model or config.llm_model
            active_overrides.append(
                {"prompt_key": key, "provider": provider, "model": model}
            )
            if key not in BENCHMARK_JUSTIFIED_PROMPT_KEYS:
                custom_nonbenchmark.append(key)
            continue
        if prompt_override_matches_code_default(config, key):
            suppressed_defaults.append(key)

    return {
        "round": round_name,
        "mode": resolved_mode,
        "default_provider": default_provider,
        "default_model": default_model,
        "active_overrides": active_overrides,
        "suppressed_default_overrides": suppressed_defaults,
        "custom_nonbenchmark_overrides": custom_nonbenchmark,
    }


def _print_prompt_routing_summary(
    config: "AgeomConfig",
    round_name: str,
    prompt_keys: list[str],
    execution_mode: str | None = None,
) -> dict[str, Any]:
    """Print a compact routing audit and return the structured summary."""
    summary = _summarize_prompt_routing(
        config,
        round_name,
        prompt_keys,
        execution_mode,
    )
    active = ", ".join(
        f"{row['prompt_key']}={row['provider']}:{row['model']}"
        for row in summary["active_overrides"]
    ) or "none"
    suppressed = ", ".join(summary["suppressed_default_overrides"]) or "none"
    custom = ", ".join(summary["custom_nonbenchmark_overrides"]) or "none"
    print(
        f"LLM routing ({round_name}): "
        f"mode={summary['mode']} "
        f"default={summary['default_provider']}:{summary['default_model']} "
        f"active=[{active}] "
        f"suppressed_defaults=[{suppressed}] "
        f"custom_nonbenchmark=[{custom}]"
    )
    return summary


def _routing_metadata_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Shrink routing audit into dashboard-friendly run metadata."""
    return {
        "round": summary["round"],
        "mode": summary["mode"],
        "default_provider": summary["default_provider"],
        "default_model": summary["default_model"],
        "active_overrides": [
            {
                "prompt_key": row["prompt_key"],
                "provider": row["provider"],
                "model": row["model"],
            }
            for row in summary["active_overrides"]
        ],
        "suppressed_default_overrides": list(summary["suppressed_default_overrides"]),
        "custom_nonbenchmark_overrides": list(summary["custom_nonbenchmark_overrides"]),
    }


@dataclass(frozen=True)
class RetrievalPolicy:
    """Effective retrieval gates after combining mode and catalog-confidence signals."""

    catalog_confidence: float
    confidence_band: str
    confidence_source: str
    skill_index_enabled: bool
    graph_retrieval_enabled: bool
    semantic_index_backend_override: str | None
    hunter_mode: str


def _resolve_retrieval_policy(
    *,
    mode_settings: Any,
    catalog: Any | None,
    texts: list[str] | tuple[str, ...],
    config: Any | None = None,
) -> RetrievalPolicy:
    """Decide whether retrieval should stay enabled for the current task."""
    semantic_override = getattr(mode_settings, "semantic_index_backend_override", None)
    # Respect explicit config-level backend (e.g. AGEOM_SEMANTIC_INDEX_BACKEND=faiss)
    # so users can force FAISS regardless of confidence band.
    if semantic_override is None and config is not None:
        cfg_backend = str(getattr(config, "semantic_index_backend", "auto")).strip().lower()
        if cfg_backend not in {"auto", ""}:
            semantic_override = cfg_backend
    hunter_mode = str(getattr(mode_settings, "hunter_mode", "standard"))
    graph_enabled = bool(getattr(mode_settings, "graph_retrieval_enabled", False))
    skill_enabled = bool(getattr(mode_settings, "skill_index_enabled", False))

    if catalog is None or getattr(catalog, "size", 0) == 0:
        return RetrievalPolicy(
            catalog_confidence=0.0,
            confidence_band="none",
            confidence_source="empty_catalog",
            skill_index_enabled=False,
            graph_retrieval_enabled=False,
            semantic_index_backend_override=semantic_override or "lexical",
            hunter_mode="standard",
        )

    confidences = []
    for text in texts:
        stripped = str(text or "").strip()
        if not stripped:
            continue
        confidence = catalog.estimate_confidence(stripped)
        confidences.append((stripped, confidence))

    if not confidences:
        return RetrievalPolicy(
            catalog_confidence=0.0,
            confidence_band="low",
            confidence_source="no_text",
            skill_index_enabled=False,
            graph_retrieval_enabled=False,
            semantic_index_backend_override=semantic_override or "lexical",
            hunter_mode="standard",
        )

    best_text, best = max(confidences, key=lambda item: item[1].score)
    band = "low"
    if best.score >= 0.70:
        band = "high"
    elif best.score >= 0.40:
        band = "medium"

    if band == "low":
        return RetrievalPolicy(
            catalog_confidence=best.score,
            confidence_band=band,
            confidence_source=best_text,
            skill_index_enabled=False,
            graph_retrieval_enabled=False,
            semantic_index_backend_override=semantic_override or "lexical",
            hunter_mode="standard",
        )

    if band == "medium":
        return RetrievalPolicy(
            catalog_confidence=best.score,
            confidence_band=band,
            confidence_source=best_text,
            skill_index_enabled=skill_enabled,
            graph_retrieval_enabled=False,
            semantic_index_backend_override=semantic_override or "lexical",
            hunter_mode="standard",
        )

    return RetrievalPolicy(
        catalog_confidence=best.score,
        confidence_band=band,
        confidence_source=best_text,
        skill_index_enabled=skill_enabled,
        graph_retrieval_enabled=graph_enabled,
        semantic_index_backend_override=semantic_override,
        hunter_mode=hunter_mode,
    )


def _print_retrieval_policy(policy: RetrievalPolicy) -> None:
    """Print the effective retrieval policy for the current task."""
    print(
        "Retrieval policy: "
        f"catalog_confidence={policy.catalog_confidence:.2f} "
        f"({policy.confidence_band}), "
        f"skill_index={'on' if policy.skill_index_enabled else 'off'}, "
        f"graph_retrieval={'on' if policy.graph_retrieval_enabled else 'off'}, "
        f"semantic_backend={policy.semantic_index_backend_override or 'default'}, "
        f"hunter_mode={policy.hunter_mode}"
    )


def _parse_prompt_benchmark_provider_specs(
    raw_specs: list[str] | None,
) -> list[tuple[str, str]]:
    specs = [spec.strip() for spec in (raw_specs or []) if spec and spec.strip()]
    if not specs:
        raise ValueError("At least one --provider provider:model spec is required")

    parsed: list[tuple[str, str]] = []
    for spec in specs:
        provider, sep, model = spec.partition(":")
        provider = provider.strip()
        model = model.strip()
        if not sep or not provider or not model:
            raise ValueError(
                f"Invalid provider spec '{spec}'. Expected format provider:model"
            )
        parsed.append((provider, model))
    return parsed


def _add_mode_argument(parser: argparse.ArgumentParser) -> None:
    """Add a shared execution-mode argument to a subcommand parser."""
    parser.add_argument(
        "--mode",
        choices=["rapid", "structured", "single_agent", "verified"],
        default=None,
        help="Execution mode override (default: AGEOM_EXECUTION_MODE or verified)",
    )


def _mode_feature_summary(mode_settings: Any) -> dict[str, str]:
    """Render the resolved execution-mode feature gates for display/telemetry."""
    return {
        "mode": str(mode_settings.mode),
        "skill_index": "on" if mode_settings.skill_index_enabled else "off",
        "graph_retrieval": "on" if mode_settings.graph_retrieval_enabled else "off",
        "architect_context": "on"
        if mode_settings.architect_shared_context_enabled
        else "off",
        "hunter_context": "on" if mode_settings.hunter_shared_context_enabled else "off",
        "synth_context": "on"
        if mode_settings.synthesizer_shared_context_enabled
        else "off",
        "ingester_context": "on"
        if mode_settings.ingester_shared_context_enabled
        else "off",
        "hunter_mode": str(mode_settings.hunter_mode),
        "hunter_gbnf": "on" if mode_settings.hunter_use_gbnf else "off",
        "semantic_backend": (
            str(mode_settings.semantic_index_backend_override)
            if mode_settings.semantic_index_backend_override
            else "default"
        ),
    }


def _print_mode_summary(command_name: str, mode_settings: Any) -> None:
    """Print a compact execution-mode summary for the current command."""
    summary = _mode_feature_summary(mode_settings)
    print(
        f"Execution mode ({command_name}): {summary['mode']} "
        f"[skill_index={summary['skill_index']}, "
        f"graph_retrieval={summary['graph_retrieval']}, "
        f"architect_context={summary['architect_context']}, "
        f"hunter_context={summary['hunter_context']}, "
        f"synth_context={summary['synth_context']}, "
        f"ingester_context={summary['ingester_context']}, "
        f"hunter_mode={summary['hunter_mode']}, "
        f"hunter_gbnf={summary['hunter_gbnf']}, "
        f"semantic_backend={summary['semantic_backend']}]"
    )


def _create_proof_env(prover: "Prover", config: "AgeomConfig") -> "ProofEnvironment":
    """Create the appropriate ProofEnvironment for the given prover.

    Args:
        prover: The target proof assistant.
        config: The AgeomConfig instance.
    """
    if prover.value == "lean4":
        from ageom.judge.lean_env import LeanEnvironment

        return LeanEnvironment(config.lean_toolchain)
    elif prover.value == "python":
        from ageom.judge.python_env import PythonEnvironment

        return PythonEnvironment(
            mypy_path=config.python_mypy_path,
            python_path=config.python_path,
        )
    else:
        from ageom.judge.coq_env import CoqEnvironment

        return CoqEnvironment(config.coq_project_path)


def _load_semantic_index(
    index_dir: Path,
    config: "AgeomConfig",
    *,
    backend_override: str | None = None,
) -> tuple["SemanticIndex", str]:
    """Load semantic index with FAISS, falling back to lexical mode if needed."""
    from ageom.indexer.fallback_index import LexicalSemanticIndex

    backend = str(
        backend_override or getattr(config, "semantic_index_backend", "auto")
    ).strip().lower()
    if backend in {"lexical", "lexical_fallback"}:
        return LexicalSemanticIndex.load(index_dir), "lexical_forced"

    from ageom.indexer.builder import SemanticIndexImpl
    from ageom.indexer.embedder import create_embedder
    from ageom.indexer.faiss_store import FAISSStore

    try:
        store = FAISSStore.load(index_dir)
        metadata = store._metadata
        embedder = create_embedder(
            backend=metadata.embedding_backend if metadata is not None else config.embedding_backend,
            model_name=metadata.embedding_model if metadata is not None else config.embedding_model,
        )
        return SemanticIndexImpl(store, embedder), "faiss"
    except (ImportError, ModuleNotFoundError) as exc:
        if backend == "faiss":
            raise
        if "faiss" not in str(exc).lower():
            raise
        fallback = LexicalSemanticIndex.load(index_dir)
        return fallback, "lexical_fallback"


async def _create_shared_context(
    config: "AgeomConfig",
    *,
    enabled: bool,
) -> tuple["SharedContextStore | None", "SharedContextMetrics | None"]:
    """Create a shared context store and metrics wrapper for this command."""
    from ageom.shared_context import SharedContextMetrics, create_shared_context_store

    if not enabled:
        return None, None

    os.environ["AGEOM_SHARED_CONTEXT_INCLUDE_PROVENANCE"] = (
        "1" if config.shared_context_include_provenance else "0"
    )
    metrics = SharedContextMetrics()
    store = await create_shared_context_store(
        enabled=True,
        backend=config.shared_context_backend,
        postgres_uri=config.postgres_uri,
        postgres_table=config.shared_context_postgres_table,
        max_records_per_namespace=config.shared_context_max_records_per_namespace,
        ttl_hours=config.shared_context_ttl_hours,
        promotion_enabled=config.shared_context_promotion_enabled,
        promotion_min_confidence=config.shared_context_promotion_min_confidence,
        repo_namespace=config.shared_context_repo_namespace,
        metrics=metrics,
    )
    return store, metrics


def _load_architect_catalog(
    args: argparse.Namespace,
    config: "AgeomConfig",
):
    """Load the architect primitive catalog from built-ins, JSON catalogs, and source registries."""
    from ageom.architect.catalog import CatalogReport, PrimitiveCatalog, seed_builtin_primitives
    from ageom.architect.source_catalog import seed_catalog_from_sources
    from ageom.sources import load_sources

    catalog = PrimitiveCatalog()
    seed_builtin_primitives(catalog)

    if getattr(args, "catalog", None):
        catalog = PrimitiveCatalog.load(args.catalog)
        seed_builtin_primitives(catalog)
    else:
        search_dir = config.skill_index_dir
        if search_dir.exists():
            for cat_file in sorted(search_dir.glob("catalog_*.json")):
                print(f"Loading catalog: {cat_file.name}")
                partial = PrimitiveCatalog.load(cat_file)
                for prim in partial.all_primitives():
                    catalog.add(prim)

    report = CatalogReport()
    try:
        sources_cfg = load_sources(config.sources_file)
        derived = seed_catalog_from_sources(
            catalog,
            config=sources_cfg,
            base_dir=Path.cwd(),
            include_live_registries=False,
            report=report,
        )
        if derived or report.merged or report.structural_skips:
            parts = [f"{catalog.size} primitives"]
            if derived:
                parts.append(f"{derived} added")
            if report.merged:
                parts.append(f"{report.merged} merged")
            if report.structural_skips:
                parts.append(f"{report.structural_skips} structural skips")
            parts.append(f"from {report.total_candidates} candidates")
            print(f"Catalog: {', '.join(parts)}")
            source_parts: list[str] = []
            if report.source_live_registry_candidates:
                source_parts.append(f"{report.source_live_registry_candidates} live-registry")
            if report.source_ast_candidates:
                source_parts.append(f"{report.source_ast_candidates} ast-fallback")
            if report.source_cdg_metadata_matches:
                source_parts.append(f"{report.source_cdg_metadata_matches} cdg-matched")
            if report.source_witness_doc_fallbacks:
                source_parts.append(f"{report.source_witness_doc_fallbacks} witness-doc")
            if report.source_witness_signature_fallbacks:
                source_parts.append(
                    f"{report.source_witness_signature_fallbacks} witness-signature"
                )
            if source_parts:
                print(f"Catalog source alignment: {', '.join(source_parts)}")
    except Exception as exc:
        print(
            f"Warning: failed to derive primitives from configured sources: {exc}",
            file=sys.stderr,
        )

    report_payload = {
        "catalog_size": catalog.size,
        "total_candidates": report.total_candidates,
        "added": report.added,
        "merged": report.merged,
        "structural_skips": report.structural_skips,
        "source_live_registry_candidates": report.source_live_registry_candidates,
        "source_ast_candidates": report.source_ast_candidates,
        "source_cdg_metadata_matches": report.source_cdg_metadata_matches,
        "source_witness_doc_fallbacks": report.source_witness_doc_fallbacks,
        "source_witness_signature_fallbacks": report.source_witness_signature_fallbacks,
        "source_breakdown": report.source_breakdown,
        "merge_details": [
            {
                "candidate": candidate,
                "incumbent": incumbent,
                "similarity": similarity,
            }
            for candidate, incumbent, similarity in report.merge_details
        ],
    }
    return catalog, report_payload


def _load_skill_index_or_empty(
    config: "AgeomConfig",
    *,
    enabled: bool = True,
):
    """Load the persisted skill index unless explicitly disabled."""
    from ageom.architect.embedder import SkillIndex

    skill_index = SkillIndex(
        index_dir=config.skill_index_dir,
        embedding_backend=getattr(config, "embedding_backend", "fastembed"),
        embedding_model=getattr(config, "embedding_model", "BAAI/bge-small-en-v1.5"),
    )
    if not enabled:
        print("Warning: skill index disabled by execution mode.", file=sys.stderr)
        return skill_index
    if os.environ.get("AGEOM_DISABLE_SKILL_INDEX", "").strip() in {"1", "true", "yes"}:
        print("Warning: skill index disabled via AGEOM_DISABLE_SKILL_INDEX.", file=sys.stderr)
        return skill_index

    if config.skill_index_dir.exists():
        try:
            return SkillIndex.load(config.skill_index_dir)
        except Exception as exc:
            print(f"Warning: failed to load skill index: {exc}", file=sys.stderr)
    return skill_index


def _print_shared_context_metrics(
    label: str,
    metrics: "SharedContextMetrics | None",
) -> None:
    """Print a compact shared-context metrics line."""
    if metrics is None:
        return
    snap = metrics.snapshot()
    print(
        "  Shared context"
        f" ({label}): backend={snap['backend']} "
        f"searches={snap['searches_total']} "
        f"hit_rate={float(snap['search_hit_rate']):.2f} "
        f"avg_search_ms={float(snap['search_latency_ms_avg']):.1f} "
        f"puts={snap['puts_total']} "
        f"dup_supp_rate={float(snap['duplicate_suppression_rate']):.2f} "
        f"match_delta={float(snap['match_success_delta']):+.2f} "
        f"promotions={snap['promotions_total']} "
        f"injected_blocks={snap['injected_blocks']} "
        f"injected_chars={snap['injected_chars']} "
        f"template_hits={snap['template_search_hits']}/{snap['template_searches_total']} "
        f"template_puts={snap['template_puts_total']} "
        f"template_injected={snap['template_injected_blocks']}"
    )


def _snapshot_shared_context_metrics(
    metrics_by_label: dict[str, "SharedContextMetrics | None"],
) -> dict[str, dict[str, float | int | str]]:
    payload: dict[str, dict[str, float | int | str]] = {}
    for label, metrics in metrics_by_label.items():
        if metrics is None:
            continue
        payload[label] = metrics.snapshot()
    return payload


def _shared_context_metadata(
    metrics_by_label: dict[str, "SharedContextMetrics | None"],
    *,
    metrics_path: Path | None = None,
) -> dict[str, object]:
    """Build run-metadata payload for dashboard shared-context summaries."""
    contexts = _snapshot_shared_context_metrics(metrics_by_label)
    payload: dict[str, object] = {"contexts": contexts}
    if metrics_path is not None:
        payload["metrics_path"] = str(metrics_path)
    return payload


def _write_shared_context_metrics_file(
    path: Path,
    metrics_by_label: dict[str, "SharedContextMetrics | None"],
) -> Path | None:
    """Persist shared-context metrics JSON; return path when written."""
    contexts = _snapshot_shared_context_metrics(metrics_by_label)
    if not contexts:
        return None
    payload = {
        "generated_at_unix": time.time(),
        "contexts": contexts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def _run_async_command(coro: Any) -> None:
    """Run a CLI coroutine and close it if a mocked asyncio.run leaves it pending."""
    try:
        asyncio.run(coro)
    finally:
        if inspect.iscoroutine(coro) and getattr(coro, "cr_frame", None) is not None:
            coro.close()
