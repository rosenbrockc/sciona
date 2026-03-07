"""CLI entrypoint for AGEO-Matcher."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import shutil
import socketserver
import sys
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ageom.config import AgeomConfig, resolve_execution_mode
    from ageom.hunter.llm import LLMClient
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

    provider_attr = f"{round_name}_llm_provider"
    model_attr = f"{round_name}_llm_model"
    max_tokens_attr = f"{round_name}_llm_max_tokens" if round_name == "hunter" else None

    llm_provider = (
        getattr(args, "llm_provider", None)
        or getattr(config, provider_attr, "")
        or config.llm_provider
    )
    llm_model = (
        getattr(args, "llm_model", None)
        or getattr(config, model_attr, "")
        or config.llm_model
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
    )


def _create_llm_router(
    args: argparse.Namespace,
    config: "AgeomConfig",
    round_name: str,
    prompt_keys: list[str],
) -> "LLMClient":
    """Create an ``LLMRouter`` wrapping the default client with per-prompt overrides.

    For each *prompt_key* in *prompt_keys*, if the config has a non-empty
    ``{prompt_key}_llm_provider``, a dedicated ``LLMClient`` is created.
    Clients with matching (provider, model) pairs are deduplicated.
    """
    from ageom.config import should_apply_prompt_override
    from ageom.hunter.llm import create_llm_client
    from ageom.llm_router import LLMRouter

    default = _create_llm(args, config, round_name)
    overrides: dict[str, "LLMClient"] = {}
    # Cache by (provider, model) to avoid redundant connections
    client_cache: dict[tuple[str, str], "LLMClient"] = {}

    for key in prompt_keys:
        provider = getattr(config, f"{key}_llm_provider", "")
        model = getattr(config, f"{key}_llm_model", "")
        if not provider:
            continue
        if not should_apply_prompt_override(config, key):
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
            )
        overrides[key] = client_cache[cache_key]

    if not overrides:
        return default
    return LLMRouter(default=default, overrides=overrides)


def _summarize_prompt_routing(
    config: "AgeomConfig",
    round_name: str,
    prompt_keys: list[str],
) -> dict[str, Any]:
    """Summarize which prompt-key overrides are active vs suppressed."""
    from ageom.config import (
        BENCHMARK_JUSTIFIED_PROMPT_KEYS,
        prompt_override_matches_code_default,
        should_apply_prompt_override,
    )

    default_provider = getattr(config, f"{round_name}_llm_provider", "") or config.llm_provider
    default_model = getattr(config, f"{round_name}_llm_model", "") or config.llm_model
    active_overrides: list[dict[str, str]] = []
    suppressed_defaults: list[str] = []
    custom_nonbenchmark: list[str] = []

    for key in prompt_keys:
        provider = getattr(config, f"{key}_llm_provider", "")
        model = getattr(config, f"{key}_llm_model", "")
        if not provider:
            continue
        if should_apply_prompt_override(config, key):
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
) -> dict[str, Any]:
    """Print a compact routing audit and return the structured summary."""
    summary = _summarize_prompt_routing(config, round_name, prompt_keys)
    active = ", ".join(
        f"{row['prompt_key']}={row['provider']}:{row['model']}"
        for row in summary["active_overrides"]
    ) or "none"
    suppressed = ", ".join(summary["suppressed_default_overrides"]) or "none"
    custom = ", ".join(summary["custom_nonbenchmark_overrides"]) or "none"
    print(
        f"LLM routing ({round_name}): "
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
) -> RetrievalPolicy:
    """Decide whether retrieval should stay enabled for the current task."""
    semantic_override = getattr(mode_settings, "semantic_index_backend_override", None)
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
        choices=["rapid", "structured", "verified"],
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
    except Exception as exc:
        print(
            f"Warning: failed to derive primitives from configured sources: {exc}",
            file=sys.stderr,
        )

    return catalog


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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ageom",
        description="AGEO-Matcher: ground predicates into verified library functions",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- index build ---
    index_parser = subparsers.add_parser("index", help="Index management")
    index_sub = index_parser.add_subparsers(dest="index_command")

    build_parser = index_sub.add_parser("build", help="Build FAISS index from library")
    build_parser.add_argument(
        "--prover",
        choices=["lean4", "coq", "python"],
        required=True,
        help="Proof assistant",
    )
    build_parser.add_argument(
        "--path", type=str, default="", help="Path to Coq project (for --prover coq)"
    )
    build_parser.add_argument(
        "--packages",
        type=str,
        default=None,
        help="Comma-separated Python packages to index (for --prover python, default: numpy,scipy)",
    )
    build_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for index (default: from .env)",
    )

    # --- skill ---
    skill_parser = subparsers.add_parser(
        "skill", help="Manage the algorithmic skill catalog"
    )
    skill_sub = skill_parser.add_subparsers(dest="skill_command")

    ingest_parser = skill_sub.add_parser(
        "ingest", help="Ingest primitives from a source"
    )
    ingest_parser.add_argument(
        "--source",
        choices=["clrs", "coq100"],
        required=True,
        help="Source to ingest from",
    )
    ingest_parser.add_argument(
        "--path", type=str, required=True, help="Path to the cloned source repo"
    )
    ingest_parser.add_argument(
        "--output", type=str, default=None, help="Output path for catalog JSON"
    )

    skill_index_parser = skill_sub.add_parser(
        "index", help="Build FAISS skill index from catalog"
    )
    skill_index_parser.add_argument(
        "--catalog",
        type=str,
        default=None,
        help="Path to catalog JSON (default: auto-detect)",
    )
    skill_index_parser.add_argument(
        "--output", type=str, default=None, help="Output directory for skill index"
    )

    skill_search_parser = skill_sub.add_parser("search", help="Search the skill index")
    skill_search_parser.add_argument("query", type=str, help="Search query")
    skill_search_parser.add_argument(
        "--k", type=int, default=10, help="Number of results to return"
    )
    skill_search_parser.add_argument(
        "--index-dir", type=str, default=None, help="Skill index directory"
    )

    # --- decompose ---
    decompose_parser = subparsers.add_parser(
        "decompose", help="Decompose a goal into a Conceptual Dependency Graph"
    )
    decompose_parser.add_argument("goal", type=str, help="High-level goal to decompose")
    decompose_parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Max decomposition depth (default: from config)",
    )
    decompose_parser.add_argument(
        "--output", type=str, default=None, help="Output path for CDG JSON"
    )
    decompose_parser.add_argument(
        "--catalog",
        type=str,
        default=None,
        help="Path to catalog JSON (default: auto-detect)",
    )
    decompose_parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        help="Checkpoint thread ID (auto-generated if omitted)",
    )
    decompose_parser.add_argument(
        "--llm-provider",
        choices=["anthropic", "codex", "llama_cpp", "claude_cli", "codex_cli", "gemini_cli", "claude_shim", "codex_shim", "gemini_shim"],
        default=None,
        help="LLM provider override (default: from config)",
    )
    decompose_parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM model override for decomposition (default: from config)",
    )
    decompose_parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=None,
        help="Max output tokens for decomposition LLM calls",
    )
    decompose_parser.add_argument(
        "--no-persist",
        action="store_true",
        default=False,
        help="Disable PostgreSQL persistence (use in-memory only)",
    )
    decompose_parser.add_argument(
        "--trace",
        action="store_true",
        default=False,
        help="Write pipeline event trace to {output_dir}/trace.jsonl",
    )
    _add_mode_argument(decompose_parser)

    # --- history ---
    history_parser = subparsers.add_parser(
        "history", help="Show checkpoint history for a decomposition thread"
    )
    history_parser.add_argument("thread_id", type=str, help="Thread ID to inspect")

    # --- visualize ---
    viz_parser = subparsers.add_parser(
        "visualize", help="Open browser-based CDG visualization"
    )
    viz_parser.add_argument(
        "cdg_file",
        nargs="?",
        default=None,
        help="Path to CDG JSON to pre-load (optional)",
    )
    viz_parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="HTTP server port (default: auto-pick)",
    )
    viz_parser.add_argument(
        "--no-serve",
        action="store_true",
        default=False,
        help="Open file:// directly instead of starting a local server",
    )
    viz_parser.add_argument(
        "--api",
        action="store_true",
        default=False,
        help="Start FastAPI server with Memgraph CDG browsing (requires Memgraph connection)",
    )
    viz_parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable uvicorn auto-reload on code changes (--api mode only)",
    )

    # --- assemble ---
    assemble_parser = subparsers.add_parser(
        "assemble", help="Assemble CDG + match results into a compilable skeleton"
    )
    assemble_parser.add_argument("cdg_file", type=str, help="Path to CDG JSON")
    assemble_parser.add_argument(
        "matches_file", type=str, help="Path to match results JSON"
    )
    assemble_parser.add_argument(
        "--prover",
        choices=["lean4", "coq", "python"],
        default="lean4",
        help="Proof assistant",
    )
    assemble_parser.add_argument(
        "--output", type=str, default=None, help="Output path for generated source file"
    )
    assemble_parser.add_argument(
        "--check",
        action="store_true",
        default=False,
        help="Also compile the skeleton and report errors",
    )

    # --- synthesize ---
    synth_parser = subparsers.add_parser(
        "synthesize", help="Assemble, compile, and repair a skeleton (full Round 3)"
    )
    synth_parser.add_argument("cdg_file", type=str, help="Path to CDG JSON")
    synth_parser.add_argument(
        "matches_file", type=str, help="Path to match results JSON"
    )
    synth_parser.add_argument(
        "--prover",
        choices=["lean4", "coq", "python"],
        default="lean4",
        help="Proof assistant",
    )
    synth_parser.add_argument(
        "--output", type=str, default=None, help="Output path for final verified source"
    )
    synth_parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Max repair iterations (default: from config)",
    )
    synth_parser.add_argument(
        "--llm-provider",
        choices=["anthropic", "codex", "llama_cpp", "claude_cli", "codex_cli", "gemini_cli", "claude_shim", "codex_shim", "gemini_shim"],
        default=None,
        help="LLM provider override (default: from config)",
    )
    synth_parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM model override (default: from config)",
    )
    synth_parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=None,
        help="Max output tokens for LLM calls",
    )
    synth_parser.add_argument(
        "--trace",
        action="store_true",
        default=False,
        help="Write pipeline event trace to {output_dir}/trace.jsonl",
    )
    _add_mode_argument(synth_parser)

    # --- run (full orchestration) ---
    run_parser = subparsers.add_parser(
        "run", help="Run full orchestration: decompose -> match -> (refine) -> assemble"
    )
    run_parser.add_argument("goal", type=str, help="High-level goal")
    run_parser.add_argument(
        "--prover",
        choices=["lean4", "coq", "python"],
        default="lean4",
        help="Proof assistant",
    )
    run_parser.add_argument(
        "--max-rounds", type=int, default=3, help="Max refinement rounds (default: 3)"
    )
    run_parser.add_argument(
        "--output", type=str, default=None, help="Output directory for all artifacts"
    )
    run_parser.add_argument(
        "--catalog", type=str, default=None, help="Path to catalog JSON"
    )
    run_parser.add_argument(
        "--llm-provider",
        choices=["anthropic", "codex", "llama_cpp", "claude_cli", "codex_cli", "gemini_cli", "claude_shim", "codex_shim", "gemini_shim"],
        default=None,
        help="LLM provider override",
    )
    run_parser.add_argument(
        "--llm-model", type=str, default=None, help="LLM model override"
    )
    run_parser.add_argument(
        "--llm-max-tokens", type=int, default=None, help="Max output tokens"
    )
    run_parser.add_argument(
        "--trace", action="store_true", default=False, help="Write trace.jsonl"
    )
    _add_mode_argument(run_parser)

    # --- export ---
    export_parser = subparsers.add_parser(
        "export", help="Export verified source to compiled artifacts and FFI bindings"
    )
    export_parser.add_argument(
        "source_file",
        type=str,
        help="Path to verified .lean/.v file or SynthesisResult JSON",
    )
    export_parser.add_argument(
        "--target",
        choices=["lean-lib", "coq-lib", "rust-ffi", "c-header", "python-pkg"],
        default="lean-lib",
        help="Export target (default: lean-lib)",
    )
    export_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: from config)",
    )
    export_parser.add_argument(
        "--optimize",
        action="store_true",
        default=False,
        help="Run hot-path optimizer before export",
    )
    export_parser.add_argument(
        "--prover",
        choices=["lean4", "coq", "python"],
        default="lean4",
        help="Proof assistant (default: lean4)",
    )

    # --- optimize (Principal) ---
    optimize_parser = subparsers.add_parser(
        "optimize", help="Run NAS/AutoML optimisation loop (Principal role)"
    )
    optimize_parser.add_argument("goal", type=str, help="High-level goal to optimise")
    optimize_parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Path to benchmark dataset (CSV or JSON)",
    )
    optimize_parser.add_argument(
        "--metric",
        choices=["latency", "memory", "precision", "flop_count"],
        default="latency",
        help="Optimisation metric (default: latency)",
    )
    optimize_parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of optimisation trials (default: 50)",
    )
    optimize_parser.add_argument(
        "--prover",
        choices=["lean4", "coq", "python"],
        default="python",
        help="Proof assistant (default: python)",
    )
    optimize_parser.add_argument(
        "--catalog",
        type=str,
        default=None,
        help="Path to catalog JSON",
    )
    optimize_parser.add_argument(
        "--llm-provider",
        choices=["anthropic", "codex", "llama_cpp", "claude_cli", "codex_cli", "gemini_cli", "claude_shim", "codex_shim", "gemini_shim"],
        default=None,
        help="LLM provider override",
    )
    optimize_parser.add_argument(
        "--llm-model", type=str, default=None, help="LLM model override"
    )
    optimize_parser.add_argument(
        "--llm-max-tokens", type=int, default=None, help="Max output tokens"
    )
    optimize_parser.add_argument(
        "--no-persist",
        action="store_true",
        default=False,
        help="Disable PostgreSQL persistence",
    )
    optimize_parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-trial subprocess timeout in seconds (default: 120)",
    )
    _add_mode_argument(optimize_parser)

    # --- profile ---
    profile_parser = subparsers.add_parser(
        "profile", help="Evaluate an existing CDG and compiled artifact against a dataset"
    )
    profile_parser.add_argument(
        "--cdg", type=str, required=True, help="Path to the CDG JSON file"
    )
    profile_parser.add_argument(
        "--artifact", type=str, required=True, help="Path to the compiled artifact (Python file)"
    )
    profile_parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the benchmark dataset (CSV/JSON)"
    )
    profile_parser.add_argument(
        "--metric",
        choices=["latency", "memory", "precision", "flop_count"],
        default="precision",
        help="Optimization metric to profile (default: precision)",
    )

    # --- prompt-benchmark ---
    prompt_benchmark_parser = subparsers.add_parser(
        "prompt-benchmark",
        help="Benchmark prompt keys across providers on a small cross-domain suite",
    )
    prompt_benchmark_parser.add_argument(
        "--provider",
        action="append",
        default=[],
        help="Provider spec in the form provider:model. Repeat to compare multiple providers.",
    )
    prompt_benchmark_parser.add_argument(
        "--prompt-key",
        action="append",
        choices=["hunter_score", "hunter_reformulate", "hunter_analyze_failure"],
        default=[],
        help="Restrict the benchmark to one or more prompt keys (default: all benchmarked keys).",
    )
    prompt_benchmark_parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to run each provider/case pair (default: 1)",
    )
    prompt_benchmark_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max output tokens for benchmark calls (default: hunter config)",
    )
    prompt_benchmark_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON report path",
    )
    prompt_benchmark_parser.add_argument(
        "--compare-direct-baseline",
        action="store_true",
        default=False,
        help="Also run a simpler direct-baseline prompt variant for each provider/case pair",
    )

    # --- sources ---
    sources_parser = subparsers.add_parser(
        "sources", help="Manage multi-repo atom sources"
    )
    sources_sub = sources_parser.add_subparsers(dest="sources_command")

    sources_sub.add_parser("list", help="List resolved atom sources")
    sources_sync_parser = sources_sub.add_parser(
        "sync", help="Fetch / update git atom sources"
    )
    sources_sync_parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Sync only the named source (default: all)",
    )

    # --- ingest ---
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest source code into the atom framework (Round 0)",
    )
    ingest_parser.add_argument(
        "source", type=str, help="Path to source file (.py/.rs/.jl/.cpp/.h/.hpp)"
    )
    ingest_parser.add_argument(
        "--class",
        dest="class_name",
        type=str,
        required=True,
        help="Name of the class to ingest",
    )
    ingest_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for generated files",
    )
    ingest_parser.add_argument(
        "--llm-provider",
        choices=["anthropic", "codex", "llama_cpp", "claude_cli", "codex_cli", "gemini_cli", "claude_shim", "codex_shim", "gemini_shim"],
        default=None,
        help="LLM provider override (default: from config)",
    )
    ingest_parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM model override (default: from config)",
    )
    ingest_parser.add_argument(
        "--procedural",
        action="store_true",
        default=False,
        help="Use deterministic procedural extraction instead of LLM chunking",
    )
    ingest_parser.add_argument(
        "--trace",
        action="store_true",
        default=False,
        help="Write pipeline event trace to {output_dir}/trace.jsonl",
    )
    ingest_parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help="Print live ingestion status updates to stdout",
    )
    ingest_parser.add_argument(
        "--stale-seconds",
        type=int,
        default=120,
        help="Heartbeat threshold for stalled detection (default: 120)",
    )
    _add_mode_argument(ingest_parser)

    ingest_status_parser = subparsers.add_parser(
        "ingest-status",
        help="Inspect ingestion run state from monitor files",
    )
    ingest_status_parser.add_argument(
        "output",
        type=str,
        help="Ingestion output directory",
    )
    ingest_status_parser.add_argument(
        "--stale-seconds",
        type=int,
        default=120,
        help="Heartbeat threshold for stalled detection (default: 120)",
    )
    ingest_status_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Print full status payload as JSON",
    )

    # --- match ---
    match_parser = subparsers.add_parser(
        "match", help="Match predicates to library functions"
    )
    match_parser.add_argument("--statement", type=str, help="Single statement to match")
    match_parser.add_argument("--pdg-file", type=str, help="JSON file with PDG nodes")
    match_parser.add_argument(
        "--prover",
        choices=["lean4", "coq", "python"],
        default="lean4",
        help="Proof assistant",
    )
    match_parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Directory containing FAISS index (default: from .env)",
    )
    match_parser.add_argument(
        "--llm-provider",
        choices=["anthropic", "codex", "llama_cpp", "claude_cli", "codex_cli", "gemini_cli", "claude_shim", "codex_shim", "gemini_shim"],
        default=None,
        help="LLM provider override (default: from config)",
    )
    match_parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM model override for matching (default: from config)",
    )
    match_parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=None,
        help="Max output tokens for matching LLM calls",
    )
    match_parser.add_argument(
        "--trace",
        action="store_true",
        default=False,
        help="Write pipeline event trace to trace.jsonl",
    )
    _add_mode_argument(match_parser)

    # --- catalog-gaps ---
    gaps_parser = subparsers.add_parser(
        "catalog-gaps",
        help="Detect catalog coverage gaps from a CDG file",
    )
    gaps_parser.add_argument(
        "--cdg",
        type=str,
        required=True,
        help="Path to CDG JSON file",
    )
    gaps_parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Similarity ceiling below which a node is considered unmatched (default: 0.6)",
    )
    gaps_parser.add_argument(
        "--catalog",
        type=str,
        default=None,
        help="Path to a catalog JSON to load instead of built-ins",
    )

    # --- upsert-cdg ---
    upsert_cdg_parser = subparsers.add_parser(
        "upsert-cdg",
        help="Upsert CDG JSON files into Memgraph graph store",
    )
    upsert_cdg_parser.add_argument(
        "repo_path",
        type=str,
        help="Path to atoms repo directory (e.g. ~/personal/ageo-atoms/ageoa/biosppy)",
    )
    upsert_cdg_parser.add_argument(
        "--repo-name",
        type=str,
        default=None,
        help="Repo namespace override (default: directory basename)",
    )
    upsert_cdg_parser.add_argument(
        "--memgraph-uri",
        type=str,
        default=None,
        help="Memgraph bolt URI override (default: from config)",
    )

    args = parser.parse_args()

    if args.command == "index" and getattr(args, "index_command", None) == "build":
        _cmd_index_build(args)
    elif args.command == "skill":
        skill_cmd = getattr(args, "skill_command", None)
        if skill_cmd == "ingest":
            _cmd_skill_ingest(args)
        elif skill_cmd == "index":
            _cmd_skill_index(args)
        elif skill_cmd == "search":
            _cmd_skill_search(args)
        else:
            print(
                "Error: provide a skill subcommand (ingest, index, search)",
                file=sys.stderr,
            )
            sys.exit(1)
    elif args.command == "sources":
        sources_cmd = getattr(args, "sources_command", None)
        if sources_cmd == "list":
            _cmd_sources_list(args)
        elif sources_cmd == "sync":
            _cmd_sources_sync(args)
        else:
            print(
                "Error: provide a sources subcommand (list, sync)",
                file=sys.stderr,
            )
            sys.exit(1)
    elif args.command == "optimize":
        _run_async_command(_cmd_optimize(args))
    elif args.command == "profile":
        _run_async_command(_cmd_profile(args))
    elif args.command == "prompt-benchmark":
        _run_async_command(_cmd_prompt_benchmark(args))
    elif args.command == "decompose":
        _run_async_command(_cmd_decompose(args))
    elif args.command == "history":
        _run_async_command(_cmd_history(args))
    elif args.command == "ingest":
        _run_async_command(_cmd_ingest(args))
    elif args.command == "ingest-status":
        _cmd_ingest_status(args)
    elif args.command == "match":
        _run_async_command(_cmd_match(args))
    elif args.command == "assemble":
        _run_async_command(_cmd_assemble(args))
    elif args.command == "synthesize":
        _run_async_command(_cmd_synthesize(args))
    elif args.command == "run":
        _run_async_command(_cmd_run(args))
    elif args.command == "export":
        _run_async_command(_cmd_export(args))
    elif args.command == "visualize":
        _cmd_visualize(args)
    elif args.command == "catalog-gaps":
        _cmd_catalog_gaps(args)
    elif args.command == "upsert-cdg":
        _run_async_command(_cmd_upsert_cdg(args))
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_catalog_gaps(args: argparse.Namespace) -> None:
    """Detect catalog coverage gaps from a CDG file."""
    from ageom.architect.handoff import CDGExport
    from ageom.architect.models import AlgorithmicNode, NodeStatus
    from ageom.config import AgeomConfig

    config = AgeomConfig()
    mode_settings = resolve_execution_mode(config, getattr(args, "mode", None))
    catalog = _load_architect_catalog(args, config)

    cdg_path = Path(args.cdg)
    if not cdg_path.exists():
        print(f"CDG file not found: {cdg_path}", file=sys.stderr)
        sys.exit(1)

    with open(cdg_path) as f:
        cdg_data = json.load(f)
    cdg = CDGExport.model_validate(cdg_data)

    # Collect atomic nodes without a matched primitive
    fallback_nodes = [
        n for n in cdg.nodes
        if n.status == NodeStatus.ATOMIC and not n.matched_primitive
    ]

    if not fallback_nodes:
        print(f"No unmatched atomic nodes in {cdg_path.name}.")
        return

    # Try to use the skill index for similarity-based gap detection
    skill_index = None
    try:
        skill_index_obj = _load_skill_index_or_empty(config)
        if skill_index_obj is not None and hasattr(skill_index_obj, "search_by_embedding"):
            skill_index = skill_index_obj
    except Exception:
        pass

    clusters = catalog.find_gaps(
        fallback_nodes,
        skill_index=skill_index,
        similarity_ceiling=args.threshold,
    )

    print(f"Catalog: {catalog.size} primitives")
    print(f"Unmatched atomic nodes: {len(fallback_nodes)}")
    print(f"Gap clusters (2+ similar unmatched nodes): {len(clusters)}")

    for i, cluster in enumerate(clusters, 1):
        print(f"\n  Gap {i} ({len(cluster)} nodes):")
        for node in cluster:
            print(f"    - {node.name}: {node.description[:80]}")


def _cmd_sources_list(args: argparse.Namespace) -> None:
    """Print resolved atom sources table."""
    from ageom.config import AgeomConfig
    from ageom.sources import load_sources, resolve_source

    config = AgeomConfig()
    sources_cfg = load_sources(config.sources_file)

    if not sources_cfg.sources:
        print("No sources configured. Add entries to sources.yml.")
        return

    print(f"{'Name':<20} {'Package':<20} {'Type':<6} {'Resolved Path'}")
    print("-" * 80)
    for src in sources_cfg.sources:
        kind = "git" if src.git else "path"
        try:
            resolved = resolve_source(src)
            exists = resolved.exists()
            status = str(resolved) if exists else f"{resolved} (NOT FOUND)"
        except Exception as exc:
            status = f"ERROR: {exc}"
        print(f"{src.name:<20} {src.package:<20} {kind:<6} {status}")


def _cmd_sources_sync(args: argparse.Namespace) -> None:
    """Fetch / update git atom sources."""
    from ageom.config import AgeomConfig
    from ageom.sources import load_sources, sync_source

    config = AgeomConfig()
    sources_cfg = load_sources(config.sources_file)

    targets = sources_cfg.sources
    if args.name:
        targets = [s for s in targets if s.name == args.name]
        if not targets:
            print(f"Error: source '{args.name}' not found in sources.yml", file=sys.stderr)
            sys.exit(1)

    for src in targets:
        print(f"Syncing {src.name}...")
        try:
            resolved = sync_source(src)
            print(f"  -> {resolved}")
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)


def _run_async_command(coro: Any) -> None:
    """Run a CLI coroutine and close it if a mocked asyncio.run leaves it pending."""
    try:
        asyncio.run(coro)
    finally:
        if inspect.iscoroutine(coro) and getattr(coro, "cr_frame", None) is not None:
            coro.close()


def _cmd_skill_ingest(args: argparse.Namespace) -> None:
    """Ingest algorithmic primitives from a source repo."""
    from ageom.config import AgeomConfig

    config = AgeomConfig()
    source = args.source
    source_path = Path(args.path)

    if not source_path.exists():
        print(f"Error: source path {source_path} not found", file=sys.stderr)
        sys.exit(1)

    if source == "clrs":
        from ageom.architect.ingest_clrs import ingest_clrs

        print(f"Ingesting CLRS-30 from {source_path}...")
        catalog = ingest_clrs(source_path)
    elif source == "coq100":
        from ageom.architect.ingest_coq100 import ingest_coq100

        print(f"Ingesting coq-100-theorems from {source_path}...")
        catalog = ingest_coq100(source_path)
    else:
        print(f"Error: unknown source {source}", file=sys.stderr)
        sys.exit(1)

    output = args.output or str(config.skill_index_dir / f"catalog_{source}.json")
    catalog.save(output)
    print(f"Catalog saved to {output} ({catalog.size} primitives)")


def _cmd_skill_index(args: argparse.Namespace) -> None:
    """Build FAISS skill index from a catalog."""
    from ageom.architect.embedder import SkillIndex
    from ageom.config import AgeomConfig

    config = AgeomConfig()
    output_dir = Path(args.output) if args.output else config.skill_index_dir

    catalog = _load_architect_catalog(args, config)

    if catalog.size == 0:
        print(
            "Error: no primitives found. Run 'ageom skill ingest' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Building skill index from {catalog.size} primitives...")
    index = SkillIndex(
        index_dir=output_dir,
        embedding_backend=config.embedding_backend,
        embedding_model=config.embedding_model,
    )
    index.build_from_catalog(catalog)
    index.save()
    print(f"Skill index saved to {output_dir}")


def _cmd_skill_search(args: argparse.Namespace) -> None:
    """Search the skill index."""
    from ageom.architect.embedder import SkillIndex
    from ageom.config import AgeomConfig

    config = AgeomConfig()
    index_dir = Path(args.index_dir) if args.index_dir else config.skill_index_dir

    if not index_dir.exists():
        print(
            f"Error: skill index not found at {index_dir}. Run 'ageom skill index' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    index = SkillIndex.load(index_dir)
    results = index.search(args.query, k=args.k)

    if not results:
        print("No results found.")
        return

    for i, prim in enumerate(results, 1):
        print(f"{i}. [{prim.category.value}] {prim.name}")
        print(f"   {prim.description[:100]}")
        if prim.type_signature:
            print(f"   Type: {prim.type_signature[:80]}")
        print()


def _cmd_index_build(args: argparse.Namespace) -> None:
    """Build a FAISS index from library declarations."""
    from ageom.config import AgeomConfig
    from ageom.indexer.builder import IndexBuilder
    from ageom.types import Prover

    config = AgeomConfig()
    output_dir = args.output or str(config.index_dir)
    prover = Prover(args.prover)

    print(f"Building index for {prover.value}...")

    if prover == Prover.LEAN4:
        from ageom.indexer.lean_source import LeanDeclarationSource

        source = LeanDeclarationSource()
        declarations = source.get_all_declarations()
        print(f"Found {len(declarations)} declarations from Mathlib")
    elif prover == Prover.COQ:
        from ageom.indexer.coq_source import CoqDeclarationSource

        if not args.path:
            print("Error: --path is required for Coq", file=sys.stderr)
            sys.exit(1)
        source = CoqDeclarationSource()
        declarations = source.get_all_declarations(args.path)
        print(f"Found {len(declarations)} declarations from {args.path}")
    elif prover == Prover.PYTHON:
        from ageom.indexer.python_source import PythonDeclarationSource
        from ageom.sources import load_sources, resolve_source

        py_source = PythonDeclarationSource()
        declarations = []

        if getattr(args, "packages", None):
            # Explicit --packages flag: use legacy behaviour
            packages = [p.strip() for p in args.packages.split(",") if p.strip()]
            for pkg in packages:
                declarations.extend(py_source.get_declarations_from_package(pkg))
            label = ", ".join(packages)
        else:
            # Use sources.yml
            sources_cfg = load_sources(config.sources_file)
            if sources_cfg.sources:
                pkg_labels: list[str] = []
                for src in sources_cfg.sources:
                    root = resolve_source(src)
                    declarations.extend(
                        py_source.get_declarations_from_path(root, src.package)
                    )
                    pkg_labels.append(src.package)
                label = ", ".join(pkg_labels)
            else:
                # Fallback to config.python_packages
                packages = [
                    p.strip()
                    for p in config.python_packages.split(",")
                    if p.strip()
                ]
                for pkg in packages:
                    declarations.extend(
                        py_source.get_declarations_from_package(pkg)
                    )
                label = ", ".join(packages)
        print(f"Found {len(declarations)} declarations from {label}")
    else:
        print(f"Error: unsupported prover {prover}", file=sys.stderr)
        sys.exit(1)

    builder = IndexBuilder(
        embedding_backend=config.embedding_backend,
        embedding_model=config.embedding_model,
    )
    store = builder.build_from_declarations(
        declarations, source_lib=args.path or "Mathlib", prover=prover
    )
    store.save(output_dir)
    print(f"Index saved to {output_dir} ({store.size} entries)")


def _cmd_ingest_status(args: argparse.Namespace) -> None:
    """Inspect ingestion monitor status and return meaningful exit codes."""
    from ageom.ingester.monitor import COMPLETED_FILE, FAILED_FILE, IngestMonitor

    output_dir = Path(args.output)
    status = IngestMonitor.read_status(output_dir)
    derived_state = IngestMonitor.classify_state(
        status, stale_seconds=max(5, int(args.stale_seconds))
    )

    completed_path = output_dir / COMPLETED_FILE
    failed_path = output_dir / FAILED_FILE
    if completed_path.exists():
        derived_state = "completed"
    elif failed_path.exists():
        derived_state = "failed"

    payload = {
        "output_dir": str(output_dir),
        "derived_state": derived_state,
        "status": status,
        "has_completed_marker": completed_path.exists(),
        "has_failed_marker": failed_path.exists(),
    }

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        phase = str(status.get("phase", "")) if status else ""
        step = str(status.get("current_step", "")) if status else ""
        last_heartbeat = float(status.get("last_heartbeat_at") or 0.0) if status else 0.0
        heartbeat_age = max(0.0, time.time() - last_heartbeat) if last_heartbeat else 0.0
        print(f"state={derived_state}")
        print(f"output={output_dir}")
        if phase:
            print(f"phase={phase}")
        if step:
            print(f"step={step}")
        if last_heartbeat:
            print(f"heartbeat_age_sec={heartbeat_age:.1f}")
        if status.get("llm_call_inflight"):
            inflight = status["llm_call_inflight"]
            prompt_key = str(inflight.get("prompt_key", ""))
            print(f"llm_inflight={prompt_key}")
        if failed_path.exists():
            try:
                failed_payload = json.loads(failed_path.read_text())
                if isinstance(failed_payload, dict):
                    err = failed_payload.get("error")
                    if err:
                        print(f"error={err}")
            except json.JSONDecodeError:
                pass

    if derived_state in {"failed", "stalled"}:
        sys.exit(2)
    if derived_state in {"missing", "unknown"}:
        sys.exit(1)


async def _cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a source unit into the atom framework."""
    from ageom.config import AgeomConfig, resolve_execution_mode
    from ageom.ingester import IngesterAgent
    from ageom.ingester.monitor import IngestMonitor
    from ageom.types import Prover

    config = AgeomConfig()
    mode_settings = resolve_execution_mode(config, getattr(args, "mode", None))
    output_dir = Path(args.output) if args.output else Path("output") / args.class_name
    output_dir.mkdir(parents=True, exist_ok=True)
    _print_mode_summary("ingest", mode_settings)

    llm_provider = (
        getattr(args, "llm_provider", None)
        or config.ingester_llm_provider
        or config.llm_provider
    )
    llm_model = (
        getattr(args, "llm_model", None)
        or config.ingester_llm_model
        or config.llm_model
    )
    stale_seconds = max(5, int(getattr(args, "stale_seconds", 120)))
    monitor = IngestMonitor(
        output_dir,
        enable_trace=bool(getattr(args, "trace", False)),
        monitor_stdout=bool(getattr(args, "monitor", False)),
        stale_seconds=stale_seconds,
    )
    monitor.start(
        source_path=str(args.source),
        class_name=args.class_name,
        procedural=bool(getattr(args, "procedural", False)),
        llm_provider=llm_provider,
        llm_model=llm_model,
        max_depth=int(config.ingester_max_depth),
    )

    proof_env = None
    try:
        source_path = Path(args.source)
        if not source_path.exists():
            raise FileNotFoundError(f"source file not found: {source_path}")

        # Set up LLM
        from ageom.llm_router import (
            INGESTER_ABSTRACT,
            INGESTER_CHUNK,
            INGESTER_DECOMPOSE,
            INGESTER_FIX_GHOST,
            INGESTER_FIX_TYPE,
            INGESTER_HOIST_STATE,
            INGESTER_OPAQUE_WITNESS,
        )

        prompt_keys = [
            INGESTER_CHUNK,
            INGESTER_HOIST_STATE,
            INGESTER_ABSTRACT,
            INGESTER_FIX_TYPE,
            INGESTER_FIX_GHOST,
            INGESTER_OPAQUE_WITNESS,
            INGESTER_DECOMPOSE,
        ]
        _print_prompt_routing_summary(config, "ingester", prompt_keys)
        llm = _create_llm_router(args, config, "ingester", prompt_keys)

        # Set up proof environment (Python/mypy)
        proof_env = _create_proof_env(Prover.PYTHON, config)

        # Optionally load FAISS index
        faiss_index = None
        if config.index_dir.exists():
            try:
                faiss_index, _index_mode = _load_semantic_index(
                    config.index_dir,
                    config,
                    backend_override=mode_settings.semantic_index_backend_override,
                )
            except Exception as exc:
                print(f"Warning: failed to load semantic index: {exc}", file=sys.stderr)

        ingester_run_id = uuid.uuid4().hex
        shared_context, shared_context_metrics = await _create_shared_context(
            config,
            enabled=mode_settings.ingester_shared_context_enabled,
        )
        agent = IngesterAgent(
            llm=llm,
            proof_env=proof_env,
            faiss_index=faiss_index,
            output_dir=str(output_dir),
            max_depth=config.ingester_max_depth,
            line_threshold=config.ingester_decompose_line_threshold,
            monitor=monitor,
            shared_context=shared_context,
            shared_context_metrics=shared_context_metrics,
            context_namespace=f"ingester/{ingester_run_id}",
            context_budget_chars=config.ingester_shared_context_budget_chars,
            parallelism=config.ingester_parallelism,
            enable_cache=config.ingester_cache_enabled,
            cache_dir=str(config.ingester_cache_dir),
        )

        print(f"Ingesting {'class' if not getattr(args, 'procedural', False) else 'procedural'} '{args.class_name}' from {source_path}")
        if getattr(args, "procedural", False):
            bundle = await agent.ingest_procedural(str(source_path), args.class_name)
        else:
            bundle = await agent.ingest(
                str(source_path), args.class_name, raise_on_error=True
            )

        # Stage output files and publish atomically on successful completion.
        if bundle.generated_atoms:
            monitor.stage_file("atoms.py", bundle.generated_atoms)
        if bundle.generated_state_models:
            monitor.stage_file("state_models.py", bundle.generated_state_models)
        if bundle.generated_witnesses:
            monitor.stage_file("witnesses.py", bundle.generated_witnesses)
        monitor.stage_json("cdg.json", bundle.cdg.model_dump())

        if bundle.match_results:
            matches_data = [mr.to_dict() for mr in bundle.match_results]
            monitor.stage_json("matches.json", matches_data)

        published_files = monitor.publish_staged()
        summary = {
            "cdg_nodes": len(bundle.cdg.nodes),
            "cdg_edges": len(bundle.cdg.edges),
            "matches": len(bundle.match_results),
            "mypy_passed": bool(bundle.mypy_passed),
            "ghost_sim_passed": bool(bundle.ghost_sim_passed),
            "published_files": published_files,
        }
        monitor.complete(summary=summary)

        print("\nIngestion complete:")
        print(f"  CDG: {len(bundle.cdg.nodes)} nodes, {len(bundle.cdg.edges)} edges")
        print(f"  Matches: {len(bundle.match_results)}")
        print(f"  mypy passed: {bundle.mypy_passed}")
        print(f"  Ghost sim passed: {bundle.ghost_sim_passed}")
        print(f"  Output: {output_dir}/")
        print(f"  Status: {output_dir / '.ingest_status.json'}")
        print(f"  Marker: {output_dir / 'COMPLETED.json'}")
        _print_shared_context_metrics("ingester", shared_context_metrics)
        metrics_path = _write_shared_context_metrics_file(
            output_dir / "shared_context_metrics.json",
            {"ingester": shared_context_metrics},
        )
        if metrics_path is not None:
            print(f"  Shared context metrics: {metrics_path}")

        if getattr(args, "trace", False):
            print(f"  Trace: {output_dir / 'trace.jsonl'}")
    except Exception as exc:
        monitor.fail(error=str(exc))
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if proof_env is not None:
            await proof_env.close()


async def _run_decompose(
    agent: "DecompositionAgent",
    args: argparse.Namespace,
    max_depth: int,
    catalog: "PrimitiveCatalog",
) -> "CDGExport":
    """Run decomposition and print summary — shared by retrieval on/off paths."""
    from ageom.architect.handoff import CDGExport

    print(f"Decomposing: {args.goal}")
    print(f"  Max depth: {max_depth}, Catalog size: {catalog.size}")

    cdg = await agent.decompose(args.goal, thread_id=args.thread_id)

    thread_id = cdg.metadata.get("thread_id", "")
    print(f"  Thread ID: {thread_id}")

    by_status: dict[str, int] = {}
    for node in cdg.nodes:
        status = node.status.value
        by_status[status] = by_status.get(status, 0) + 1

    print("\nDecomposition complete:")
    print(f"  Nodes: {len(cdg.nodes)}, Edges: {len(cdg.edges)}")
    for status, count in sorted(by_status.items()):
        print(f"    {status}: {count}")
    print(f"  Complete: {cdg.is_complete()}")

    return cdg


async def _cmd_decompose(args: argparse.Namespace) -> None:
    """Decompose a goal into a Conceptual Dependency Graph."""
    from ageom.architect.checkpointer import create_checkpointer
    from ageom.architect.graph import DecompositionAgent
    from ageom.architect.handoff import save_json
    from ageom.config import AgeomConfig, resolve_execution_mode

    config = AgeomConfig()
    mode_settings = resolve_execution_mode(config, getattr(args, "mode", None))
    max_depth = args.max_depth or config.architect_max_depth
    _print_mode_summary("decompose", mode_settings)

    catalog = _load_architect_catalog(args, config)
    retrieval_policy = _resolve_retrieval_policy(
        mode_settings=mode_settings,
        catalog=catalog,
        texts=[args.goal],
    )
    _print_retrieval_policy(retrieval_policy)

    if catalog.size == 0:
        print(
            "Warning: no catalog loaded. Decomposition will have no atomic stop conditions.",
            file=sys.stderr,
        )

    skill_index = _load_skill_index_or_empty(
        config,
        enabled=retrieval_policy.skill_index_enabled,
    )

    # Set up LLM
    try:
        from ageom.llm_router import (
            ARCHITECT_CRITIQUE,
            ARCHITECT_DECOMPOSE,
            ARCHITECT_STRATEGY,
        )

        prompt_keys = [
            ARCHITECT_STRATEGY,
            ARCHITECT_DECOMPOSE,
            ARCHITECT_CRITIQUE,
        ]
        _print_prompt_routing_summary(config, "architect", prompt_keys)
        llm = _create_llm_router(args, config, "architect", prompt_keys)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ImportError as exc:
        print(f"Error: missing LLM dependency ({exc})", file=sys.stderr)
        sys.exit(1)

    # Determine persistence URI
    postgres_uri = "" if args.no_persist else config.postgres_uri
    architect_run_id = uuid.uuid4().hex
    architect_shared_context, architect_shared_metrics = await _create_shared_context(
        config,
        enabled=mode_settings.architect_shared_context_enabled,
    )
    architect_context_namespace = f"architect/{architect_run_id}"

    # Set up graph retriever (opt-in)
    retriever = None
    graph_store_ctx = None
    if retrieval_policy.graph_retrieval_enabled:
        from ageom.architect.graph_retrieval import make_retriever
        from ageom.graph_store import GraphStore

        graph_store_ctx = GraphStore(
            uri=config.memgraph_uri,
            user=config.memgraph_user,
            password=config.memgraph_password,
        )

    async with create_checkpointer(postgres_uri) as checkpointer:
        if graph_store_ctx is not None:
            async with graph_store_ctx as store:
                retriever = make_retriever(config, store, current_repo="")
                agent = DecompositionAgent(
                    catalog=catalog,
                    skill_index=skill_index,
                    llm=llm,
                    max_depth=max_depth,
                    checkpointer=checkpointer,
                    graph_retriever=retriever,
                    shared_context=architect_shared_context,
                    shared_context_metrics=architect_shared_metrics,
                    context_namespace=architect_context_namespace,
                    context_budget_chars=config.architect_shared_context_budget_chars,
                )
                cdg = await _run_decompose(agent, args, max_depth, catalog)
        else:
            agent = DecompositionAgent(
                catalog=catalog,
                skill_index=skill_index,
                llm=llm,
                max_depth=max_depth,
                checkpointer=checkpointer,
                shared_context=architect_shared_context,
                shared_context_metrics=architect_shared_metrics,
                context_namespace=architect_context_namespace,
                context_budget_chars=config.architect_shared_context_budget_chars,
            )
            cdg = await _run_decompose(agent, args, max_depth, catalog)

        # Save output
        if args.output:
            save_json(cdg, args.output)
            print(f"  Saved to: {args.output}")
            metrics_path = _write_shared_context_metrics_file(
                Path(args.output).parent / "shared_context_metrics.json",
                {"architect": architect_shared_metrics},
            )
            if metrics_path is not None:
                print(f"  Shared context metrics: {metrics_path}")
        _print_shared_context_metrics("architect", architect_shared_metrics)


async def _cmd_history(args: argparse.Namespace) -> None:
    """Show checkpoint history for a decomposition thread."""
    from ageom.architect.checkpointer import create_checkpointer
    from ageom.architect.graph import DecompositionAgent
    from ageom.config import AgeomConfig
    from ageom.hunter.llm import LLMClient

    config = AgeomConfig()

    # Minimal no-op deps — we only need the compiled graph for state queries
    class _Stub(LLMClient):
        async def complete(self, system: str, user: str) -> str:
            return ""

        async def complete_with_grammar(
            self, system: str, user: str, grammar: str
        ) -> str:
            return ""

    async with create_checkpointer(config.postgres_uri) as checkpointer:
        agent = DecompositionAgent(
            catalog=None,  # type: ignore[arg-type]
            skill_index=None,  # type: ignore[arg-type]
            llm=_Stub(),
            checkpointer=checkpointer,
        )
        history = await agent.get_state_history(args.thread_id)

    if not history:
        print(f"No checkpoints found for thread {args.thread_id}")
        return

    print(f"Thread {args.thread_id}: {len(history)} checkpoint(s)\n")
    for i, entry in enumerate(history):
        vals = entry["values"]
        node_count = len(vals.get("nodes", []))
        pending = len(vals.get("pending_node_ids", []))
        done = vals.get("done", False)
        cp_id = entry.get("checkpoint_id", "?")
        print(f"  [{i}] checkpoint_id={cp_id}")
        print(f"      nodes={node_count}  pending={pending}  done={done}")


async def _cmd_match(args: argparse.Namespace) -> None:
    """Match predicates to library functions."""
    from ageom.config import AgeomConfig, resolve_execution_mode
    from ageom.hunter.graph import HunterAgent
    from ageom.judge.checker import VerificationOracleImpl
    from ageom.types import PDGNode, Prover

    config = AgeomConfig()
    mode_settings = resolve_execution_mode(config, getattr(args, "mode", None))
    _print_mode_summary("match", mode_settings)

    # Build PDG nodes
    nodes: list[PDGNode] = []
    prover = Prover(args.prover)
    if args.statement:
        nodes.append(
            PDGNode(
                predicate_id="cli-0",
                statement=args.statement,
                prover=prover,
            )
        )
    elif args.pdg_file:
        pdg_path = Path(args.pdg_file)
        with open(pdg_path) as f:
            pdg_data = json.load(f)
        for item in pdg_data:
            nodes.append(
                PDGNode(
                    predicate_id=item.get("predicate_id", ""),
                    statement=item["statement"],
                    informal_desc=item.get("informal_desc", ""),
                    prover=prover,
                    context=item.get("context", {}),
                )
            )
    else:
        print("Error: provide --statement or --pdg-file", file=sys.stderr)
        sys.exit(1)

    catalog = _load_architect_catalog(args, config)
    retrieval_policy = _resolve_retrieval_policy(
        mode_settings=mode_settings,
        catalog=catalog,
        texts=[f"{node.statement} {node.informal_desc}".strip() for node in nodes],
    )
    _print_retrieval_policy(retrieval_policy)

    # Load index
    index_dir = Path(args.index_dir) if args.index_dir else config.index_dir
    if not index_dir.exists():
        print(
            f"Error: index directory {index_dir} not found. Run 'ageom index build' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    index, index_mode = _load_semantic_index(
        index_dir,
        config,
        backend_override=retrieval_policy.semantic_index_backend_override,
    )
    if index_mode != "faiss":
        print(
            "Warning: FAISS unavailable; using lexical fallback index for Hunter.",
            file=sys.stderr,
        )

    # Set up verification oracle
    env = _create_proof_env(prover, config)
    if prover == Prover.LEAN4:
        oracle = VerificationOracleImpl(lean_env=env)
    elif prover == Prover.PYTHON:
        oracle = VerificationOracleImpl(python_env=env)
    else:
        oracle = VerificationOracleImpl(coq_env=env)

    # Set up LLM
    try:
        from ageom.llm_router import (
            HUNTER_ANALYZE_FAILURE,
            HUNTER_REFORMULATE,
            HUNTER_SCORE,
        )

        prompt_keys = [
            HUNTER_SCORE,
            HUNTER_REFORMULATE,
            HUNTER_ANALYZE_FAILURE,
        ]
        _print_prompt_routing_summary(config, "hunter", prompt_keys)
        llm = _create_llm_router(args, config, "hunter", prompt_keys)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ImportError as exc:
        print(f"Error: missing LLM dependency ({exc})", file=sys.stderr)
        sys.exit(1)

    run_id = uuid.uuid4().hex
    shared_context, shared_context_metrics = await _create_shared_context(
        config,
        enabled=mode_settings.hunter_shared_context_enabled,
    )
    agent = HunterAgent(
        index=index,
        oracle=oracle,
        llm=llm,
        max_iterations=config.hunter_max_iterations,
        top_k_verify=config.hunter_top_k_verify,
        search_k=config.hunter_search_k,
        mode=retrieval_policy.hunter_mode,
        use_gbnf=mode_settings.hunter_use_gbnf,
        query_batch_size=config.hunter_query_batch_size,
        top_k_per_query=config.hunter_top_k_per_query,
        max_candidates_total=config.hunter_max_candidates_total,
        shared_context=shared_context,
        shared_context_metrics=shared_context_metrics,
        context_namespace="hunter",
        run_id=run_id,
        context_budget_chars=config.hunter_shared_context_budget_chars,
    )

    # Run matching
    for node in nodes:
        print(f"\nMatching: {node.statement}")
        result = await agent.find_match(node)
        if result.success:
            assert result.verified_match is not None
            print(f"  VERIFIED: {result.verified_match.candidate.declaration.name}")
            print(
                f"  Type: {result.verified_match.candidate.declaration.type_signature}"
            )
        else:
            print(f"  NO MATCH FOUND ({len(result.all_candidates)} candidates tried)")
            for vr in result.all_verifications:
                print(
                    f"    - {vr.candidate.declaration.name}: {vr.error_message[:100]}"
                )
    _print_shared_context_metrics("hunter", shared_context_metrics)


async def _cmd_assemble(args: argparse.Namespace) -> None:
    """Assemble CDG + match results into a compilable skeleton."""
    from ageom.architect.handoff import load_json
    from ageom.synthesizer.assembler import Assembler, AssemblyError
    from ageom.types import MatchResult, Prover

    # Load CDG
    cdg_path = Path(args.cdg_file)
    if not cdg_path.exists():
        print(f"Error: CDG file not found: {cdg_path}", file=sys.stderr)
        sys.exit(1)
    cdg = load_json(cdg_path)

    # Load match results
    matches_path = Path(args.matches_file)
    if not matches_path.exists():
        print(f"Error: matches file not found: {matches_path}", file=sys.stderr)
        sys.exit(1)
    with open(matches_path) as f:
        matches_data = json.load(f)
    if not isinstance(matches_data, list):
        print("Error: matches file must contain a JSON array", file=sys.stderr)
        sys.exit(1)
    match_results = [MatchResult.from_dict(d) for d in matches_data]

    prover = Prover(args.prover)

    # Assemble
    try:
        assembler = Assembler(prover)
        skeleton = assembler.assemble(cdg, match_results)
    except AssemblyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if prover == Prover.LEAN4:
        ext = ".lean"
    elif prover == Prover.PYTHON:
        ext = ".py"
    else:
        ext = ".v"
    output = args.output or (cdg_path.stem + "_skeleton" + ext)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(skeleton.source_code)

    print(f"Skeleton written to {output_path}")
    print(f"  Units: {len(skeleton.units)}, Sorry count: {skeleton.sorry_count}")

    # Optional compilation check
    if args.check:
        from ageom.config import AgeomConfig
        from ageom.synthesizer.compiler import SkeletonCompiler

        config = AgeomConfig()
        env = _create_proof_env(prover, config)

        try:
            compiler = SkeletonCompiler(env)
            result = await compiler.compile(skeleton)
            if result.compiled_ok:
                print("  Compilation: OK")
            else:
                print("  Compilation: FAILED")
                if result.feedback:
                    for err in result.feedback.errors:
                        print(f"    {err}")
        finally:
            await env.close()


async def _cmd_export(args: argparse.Namespace) -> None:
    """Export verified source to compiled artifacts and FFI bindings."""
    from ageom.config import AgeomConfig
    from ageom.synthesizer.extractor import ExportTarget, Extractor
    from ageom.synthesizer.models import SkeletonFile, SynthesisResult

    config = AgeomConfig()

    source_path = Path(args.source_file)
    if not source_path.exists():
        print(f"Error: source file not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    # Try loading as SynthesisResult JSON, else treat as raw source
    synthesis_result: SynthesisResult
    if source_path.suffix == ".json":
        with open(source_path) as f:
            data = json.load(f)
        synthesis_result = SynthesisResult(**data)
    else:
        source_code = source_path.read_text()
        skeleton = SkeletonFile(
            prover=args.prover,
            source_code=source_code,
        )
        synthesis_result = SynthesisResult(
            skeleton=skeleton,
            compiled_ok=True,
        )

    # Optional optimizer
    if args.optimize or config.optimize_by_default:
        from ageom.synthesizer.optimizer import Optimizer

        optimizer = Optimizer()
        candidates = optimizer.scan(synthesis_result.skeleton)
        if candidates:
            print(f"Optimizer found {len(candidates)} candidate(s) for hot-path swap")
            # Verify guards (without a real env, just apply comment-guards)
            verified = [c for c in candidates if c.rule.guard_check.startswith("--")]
            for c in verified:
                c.guard_verified = True
            if verified:
                synthesis_result.skeleton = optimizer.apply(
                    synthesis_result.skeleton, verified
                )
                print(f"  Applied {len(verified)} optimization(s)")

    target = ExportTarget(args.target)
    output_dir = Path(args.output_dir) if args.output_dir else config.export_output_dir

    extractor = Extractor(config)
    print(f"Exporting to {target.value} in {output_dir}/...")
    bundle = await extractor.extract(synthesis_result, target, output_dir)

    print("\nExport complete:")
    print(f"  Target: {bundle.target}")
    print(f"  Source: {bundle.source_path}")
    if bundle.compiled_artifact:
        print(f"  Artifact: {bundle.compiled_artifact}")
    if bundle.ffi_files:
        print("  FFI files:")
        for f in bundle.ffi_files:
            print(f"    {f}")
    if bundle.certificate:
        print(f"  Certificate: {output_dir / 'certificate.json'}")
        print(f"    Source hash: {bundle.certificate.source_hash[:16]}...")
    if bundle.errors:
        print("  Errors:")
        for err in bundle.errors:
            print(f"    {err}")


async def _cmd_synthesize(args: argparse.Namespace) -> None:
    """Assemble CDG + match results, then repair via the synthesizer agent."""
    from ageom.architect.handoff import load_json
    from ageom.config import AgeomConfig, resolve_execution_mode
    from ageom.synthesizer.agent import SynthesizerAgent
    from ageom.synthesizer.assembler import Assembler, AssemblyError
    from ageom.types import MatchResult, Prover

    config = AgeomConfig()
    mode_settings = resolve_execution_mode(config, getattr(args, "mode", None))
    _print_mode_summary("synthesize", mode_settings)

    # Load CDG
    cdg_path = Path(args.cdg_file)
    if not cdg_path.exists():
        print(f"Error: CDG file not found: {cdg_path}", file=sys.stderr)
        sys.exit(1)
    cdg = load_json(cdg_path)

    # Load match results
    matches_path = Path(args.matches_file)
    if not matches_path.exists():
        print(f"Error: matches file not found: {matches_path}", file=sys.stderr)
        sys.exit(1)
    with open(matches_path) as f:
        matches_data = json.load(f)
    if not isinstance(matches_data, list):
        print("Error: matches file must contain a JSON array", file=sys.stderr)
        sys.exit(1)
    match_results = [MatchResult.from_dict(d) for d in matches_data]

    prover = Prover(args.prover)

    # Phase 1: Assemble
    try:
        assembler = Assembler(prover)
        skeleton = assembler.assemble(cdg, match_results)
    except AssemblyError as exc:
        print(f"Error assembling skeleton: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Assembled skeleton: {len(skeleton.units)} units, {skeleton.sorry_count} sorrys"
    )

    # Set up ProofEnvironment
    env = _create_proof_env(prover, config)

    # Set up LLM
    try:
        from ageom.llm_router import SYNTHESIZER_REPAIR, SYNTHESIZER_TACTIC

        prompt_keys = [
            SYNTHESIZER_REPAIR,
            SYNTHESIZER_TACTIC,
        ]
        _print_prompt_routing_summary(config, "synthesizer", prompt_keys)
        llm = _create_llm_router(args, config, "synthesizer", prompt_keys)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ImportError as exc:
        print(f"Error: missing LLM dependency ({exc})", file=sys.stderr)
        sys.exit(1)

    max_iterations = args.max_iterations or config.synthesizer_max_iterations
    synth_run_id = uuid.uuid4().hex
    synth_shared_context, synth_shared_metrics = await _create_shared_context(
        config,
        enabled=mode_settings.synthesizer_shared_context_enabled,
    )

    try:
        agent = SynthesizerAgent(
            env=env,
            llm=llm,
            max_iterations=max_iterations,
            shared_context=synth_shared_context,
            shared_context_metrics=synth_shared_metrics,
            context_namespace=f"synthesizer/{synth_run_id}",
            context_budget_chars=config.synthesizer_shared_context_budget_chars,
        )
        print(f"Starting repair loop (max {max_iterations} iterations)...")
        result = await agent.synthesize(skeleton)

        # Output
        if prover == Prover.LEAN4:
            ext = ".lean"
        elif prover == Prover.PYTHON:
            ext = ".py"
        else:
            ext = ".v"
        output = args.output or (cdg_path.stem + "_verified" + ext)
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result.skeleton.source_code)

        print(f"\nResult written to {output_path}")
        print(f"  Compiled OK: {result.compiled_ok}")
        print(f"  Iterations used: {result.iterations_used}")
        print(f"  Patches applied: {result.patches_applied}")
        print(f"  Sorry remaining: {result.sorry_remaining}")
        _print_shared_context_metrics("synthesizer", synth_shared_metrics)
        metrics_path = _write_shared_context_metrics_file(
            output_path.parent / "shared_context_metrics.json",
            {"synthesizer": synth_shared_metrics},
        )
        if metrics_path is not None:
            print(f"  Shared context metrics: {metrics_path}")

        if result.error_history:
            print("  Errors encountered:")
            for it, cat, text in result.error_history:
                print(f"    [{it}] {cat}: {text[:80]}")

        # Write trace if requested
        if getattr(args, "trace", False):
            from ageom.telemetry import get_event_log

            trace_path = output_path.parent / "trace.jsonl"
            event_log = get_event_log()
            if len(event_log) > 0:
                event_log.save(trace_path)
                print(f"  Trace: {trace_path} ({len(event_log)} events)")
    finally:
        await env.close()


async def _cmd_run(args: argparse.Namespace) -> None:
    """Run the full orchestration loop: decompose -> match -> refine -> assemble."""
    from ageom.architect.catalog import PrimitiveCatalog, seed_builtin_primitives
    from ageom.architect.checkpointer import create_checkpointer
    from ageom.architect.graph import DecompositionAgent
    from ageom.architect.handoff import save_json
    from ageom.config import AgeomConfig, resolve_execution_mode
    from ageom.hunter.graph import HunterAgent
    from ageom.judge.checker import VerificationOracleImpl
    from ageom.orchestrator import run_orchestration
    from ageom.telemetry import (
        configure_dashboard_output,
        finish_run,
        get_event_log,
        start_run,
        telemetry_scope,
        telemetry_stage,
        update_stage,
    )
    from ageom.types import Prover

    config = AgeomConfig()
    mode_settings = resolve_execution_mode(config, getattr(args, "mode", None))
    prover = Prover(args.prover)
    output_dir = Path(args.output) if args.output else Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    _print_mode_summary("run", mode_settings)

    configure_dashboard_output(config.telemetry_runs_dir)
    event_log = get_event_log()
    event_log.configure_live_output(None)
    event_log.clear()
    if getattr(args, "trace", False):
        event_log.configure_live_output(output_dir / "trace.jsonl")
    catalog = _load_architect_catalog(args, config)
    retrieval_policy = _resolve_retrieval_policy(
        mode_settings=mode_settings,
        catalog=catalog,
        texts=[args.goal],
    )
    _print_retrieval_policy(retrieval_policy)
    architect_routing = _summarize_prompt_routing(
        config,
        "architect",
        [
            "architect_strategy",
            "architect_decompose",
            "architect_critique",
            "orchestrator_refine",
        ],
    )
    hunter_routing = _summarize_prompt_routing(
        config,
        "hunter",
        [
            "hunter_score",
            "hunter_reformulate",
            "hunter_analyze_failure",
        ],
    )
    telemetry_run_id = start_run(
        "algorithm_creation",
        metadata={
            "command": "run",
            "goal": args.goal,
            "prover": prover.value,
            "max_rounds": int(args.max_rounds),
            "execution_mode": mode_settings.mode,
            "mode_features": _mode_feature_summary(mode_settings),
            "retrieval_policy": {
                "catalog_confidence": retrieval_policy.catalog_confidence,
                "confidence_band": retrieval_policy.confidence_band,
                "skill_index": retrieval_policy.skill_index_enabled,
                "graph_retrieval": retrieval_policy.graph_retrieval_enabled,
                "semantic_backend": retrieval_policy.semantic_index_backend_override
                or "default",
                "hunter_mode": retrieval_policy.hunter_mode,
            },
            "llm_routing": {
                "architect": _routing_metadata_summary(architect_routing),
                "hunter": _routing_metadata_summary(hunter_routing),
            },
        },
    )

    try:
        with telemetry_scope(run_id=telemetry_run_id):
            update_stage(stage="setup", status="running", message="loading dependencies")

            skill_index = _load_skill_index_or_empty(
                config,
                enabled=retrieval_policy.skill_index_enabled,
            )

            # Set up LLM
            try:
                from ageom.llm_router import (
                    ARCHITECT_CRITIQUE,
                    ARCHITECT_DECOMPOSE,
                    ARCHITECT_STRATEGY,
                    ORCHESTRATOR_REFINE,
                )

                architect_prompt_keys = [
                    ARCHITECT_STRATEGY,
                    ARCHITECT_DECOMPOSE,
                    ARCHITECT_CRITIQUE,
                    ORCHESTRATOR_REFINE,
                ]
                _print_prompt_routing_summary(config, "architect", architect_prompt_keys)
                llm = _create_llm_router(
                    args,
                    config,
                    "architect",
                    architect_prompt_keys,
                )
            except (ValueError, ImportError) as exc:
                print(f"Error: {exc}", file=sys.stderr)
                finish_run(telemetry_run_id, status="failed", error=str(exc))
                sys.exit(1)
            update_stage(stage="setup", status="completed")

            # Step 1: Decompose
            print(f"Decomposing: {args.goal}")

            retriever = None
            graph_store_ctx = None
            if retrieval_policy.graph_retrieval_enabled:
                from ageom.architect.graph_retrieval import make_retriever
                from ageom.graph_store import GraphStore as _GraphStore

                graph_store_ctx = _GraphStore(
                    uri=config.memgraph_uri,
                    user=config.memgraph_user,
                    password=config.memgraph_password,
                )
            architect_run_id = uuid.uuid4().hex
            architect_shared_context, architect_shared_metrics = await _create_shared_context(
                config,
                enabled=mode_settings.architect_shared_context_enabled,
            )

            with telemetry_stage("architect_decompose", message="building initial CDG"):
                async with create_checkpointer(config.postgres_uri) as checkpointer:
                    if graph_store_ctx is not None:
                        async with graph_store_ctx as gstore:
                            retriever = make_retriever(config, gstore, current_repo="")
                            architect = DecompositionAgent(
                                catalog=catalog,
                                skill_index=skill_index,
                                llm=llm,
                                checkpointer=checkpointer,
                                graph_retriever=retriever,
                                shared_context=architect_shared_context,
                                shared_context_metrics=architect_shared_metrics,
                                context_namespace=f"architect/{architect_run_id}",
                                context_budget_chars=config.architect_shared_context_budget_chars,
                            )
                            cdg = await architect.decompose(args.goal)
                    else:
                        architect = DecompositionAgent(
                            catalog=catalog,
                            skill_index=skill_index,
                            llm=llm,
                            checkpointer=checkpointer,
                            shared_context=architect_shared_context,
                            shared_context_metrics=architect_shared_metrics,
                            context_namespace=f"architect/{architect_run_id}",
                            context_budget_chars=config.architect_shared_context_budget_chars,
                        )
                        cdg = await architect.decompose(args.goal)

            print(f"  Decomposed: {len(cdg.nodes)} nodes, {len(cdg.edges)} edges")

            # Step 2: Set up Hunter
            index_dir = config.index_dir
            if not index_dir.exists():
                print(
                    f"Error: index directory {index_dir} not found. Run 'ageom index build' first.",
                    file=sys.stderr,
                )
                finish_run(
                    telemetry_run_id,
                    status="failed",
                    error=f"missing index directory: {index_dir}",
                )
                sys.exit(1)

            with telemetry_stage("hunter_setup", message="loading retrieval index"):
                index, index_mode = _load_semantic_index(
                    index_dir,
                    config,
                    backend_override=retrieval_policy.semantic_index_backend_override,
                )
                if index_mode != "faiss":
                    print(
                        "Warning: FAISS unavailable; using lexical fallback index for Hunter.",
                        file=sys.stderr,
                    )

                env = _create_proof_env(prover, config)
                if prover == Prover.LEAN4:
                    oracle = VerificationOracleImpl(lean_env=env)
                elif prover == Prover.PYTHON:
                    oracle = VerificationOracleImpl(python_env=env)
                else:
                    oracle = VerificationOracleImpl(coq_env=env)

                try:
                    from ageom.llm_router import (
                        HUNTER_ANALYZE_FAILURE,
                        HUNTER_REFORMULATE,
                        HUNTER_SCORE,
                    )

                    hunter_prompt_keys = [
                        HUNTER_SCORE,
                        HUNTER_REFORMULATE,
                        HUNTER_ANALYZE_FAILURE,
                    ]
                    _print_prompt_routing_summary(config, "hunter", hunter_prompt_keys)
                    hunter_llm = _create_llm_router(
                        args,
                        config,
                        "hunter",
                        hunter_prompt_keys,
                    )
                except (ValueError, ImportError) as exc:
                    print(f"Error setting up hunter LLM: {exc}", file=sys.stderr)
                    finish_run(telemetry_run_id, status="failed", error=str(exc))
                    sys.exit(1)

                run_id = uuid.uuid4().hex
                hunter_shared_context, hunter_shared_metrics = await _create_shared_context(
                    config,
                    enabled=mode_settings.hunter_shared_context_enabled,
                )
                hunter = HunterAgent(
                    index=index,
                    oracle=oracle,
                    llm=hunter_llm,
                    max_iterations=config.hunter_max_iterations,
                    top_k_verify=config.hunter_top_k_verify,
                    search_k=config.hunter_search_k,
                    mode=retrieval_policy.hunter_mode,
                    use_gbnf=mode_settings.hunter_use_gbnf,
                    query_batch_size=config.hunter_query_batch_size,
                    top_k_per_query=config.hunter_top_k_per_query,
                    max_candidates_total=config.hunter_max_candidates_total,
                    shared_context=hunter_shared_context,
                    shared_context_metrics=hunter_shared_metrics,
                    context_namespace="hunter",
                    run_id=run_id,
                    context_budget_chars=config.hunter_shared_context_budget_chars,
                )

            # Step 3: Run orchestration loop
            print(f"Running orchestration (max {args.max_rounds} rounds)...")
            with telemetry_stage(
                "orchestration",
                message="architect->hunter refine loop",
                total=int(args.max_rounds),
            ):
                try:
                    result = await run_orchestration(
                        cdg,
                        hunter_agent=hunter,
                        llm=llm,
                        prover=prover,
                        max_rounds=args.max_rounds,
                        hunter_concurrency=config.orchestrator_hunter_concurrency,
                    )
                finally:
                    await env.close()
            update_stage(
                stage="orchestration",
                completed=result.rounds_used,
                total=int(args.max_rounds),
            )
    except Exception as exc:
        finish_run(telemetry_run_id, status="failed", error=str(exc))
        raise
    else:
        pass
    finally:
        event_log.configure_live_output(None)

    # Output
    print("\nOrchestration complete:")
    print(f"  Rounds used: {result.rounds_used}")
    print(
        f"  Matches: {sum(1 for mr in result.match_results if mr.success)}/{len(result.match_results)}"
    )
    if result.ungroundable:
        print(f"  Ungroundable: {result.ungroundable}")

    save_json(result.cdg, output_dir / "cdg.json")

    if result.match_results:

        matches_data = [mr.to_dict() for mr in result.match_results]
        with open(output_dir / "matches.json", "w") as f:
            json.dump(matches_data, f, indent=2)

    print(f"  Output: {output_dir}/")
    _print_shared_context_metrics("architect", architect_shared_metrics)
    _print_shared_context_metrics("hunter", hunter_shared_metrics)
    metrics_path = _write_shared_context_metrics_file(
        output_dir / "shared_context_metrics.json",
        {
            "architect": architect_shared_metrics,
            "hunter": hunter_shared_metrics,
        },
    )
    if metrics_path is not None:
        print(f"  Shared context metrics: {metrics_path}")

    if getattr(args, "trace", False):
        if len(event_log) > 0:
            event_log.save(output_dir / "trace.jsonl")
            print(f"  Trace: {output_dir / 'trace.jsonl'} ({len(event_log)} events)")

    finish_run(telemetry_run_id, status="completed")


async def _cmd_optimize(args: argparse.Namespace) -> None:
    """Run the Principal NAS/AutoML optimisation loop."""
    from ageom.architect.catalog import PrimitiveCatalog, seed_builtin_primitives
    from ageom.architect.checkpointer import create_checkpointer
    from ageom.architect.graph import DecompositionAgent
    from ageom.config import AgeomConfig, resolve_execution_mode
    from ageom.principal.evaluator import ExecutionSandbox
    from ageom.principal.graph import (
        PrincipalDeps,
        build_principal_graph,
    )
    from ageom.principal.models import OptimizationMetric

    config = AgeomConfig()
    mode_settings = resolve_execution_mode(config, getattr(args, "mode", None))
    _print_mode_summary("optimize", mode_settings)

    catalog = _load_architect_catalog(args, config)
    retrieval_policy = _resolve_retrieval_policy(
        mode_settings=mode_settings,
        catalog=catalog,
        texts=[args.goal],
    )
    _print_retrieval_policy(retrieval_policy)

    skill_index = _load_skill_index_or_empty(
        config,
        enabled=retrieval_policy.skill_index_enabled,
    )

    # LLM
    try:
        from ageom.llm_router import (
            ARCHITECT_CRITIQUE,
            ARCHITECT_DECOMPOSE,
            ARCHITECT_STRATEGY,
        )

        prompt_keys = [
            ARCHITECT_STRATEGY,
            ARCHITECT_DECOMPOSE,
            ARCHITECT_CRITIQUE,
        ]
        _print_prompt_routing_summary(config, "architect", prompt_keys)
        llm = _create_llm_router(args, config, "architect", prompt_keys)
    except (ValueError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    metric = OptimizationMetric(args.metric)
    postgres_uri = "" if args.no_persist else config.postgres_uri
    architect_run_id = uuid.uuid4().hex
    architect_shared_context, architect_shared_metrics = await _create_shared_context(
        config,
        enabled=mode_settings.architect_shared_context_enabled,
    )

    print("Principal optimisation loop")
    print(f"  Goal: {args.goal}")
    print(f"  Metric: {metric.value}")
    print(f"  Trials: {args.trials}")
    print(f"  Benchmark: {args.benchmark}")
    print()

    async with create_checkpointer(postgres_uri) as checkpointer:
        architect = DecompositionAgent(
            catalog=catalog,
            skill_index=skill_index,
            llm=llm,
            checkpointer=checkpointer,
            shared_context=architect_shared_context,
            shared_context_metrics=architect_shared_metrics,
            context_namespace=f"architect/{architect_run_id}",
            context_budget_chars=config.architect_shared_context_budget_chars,
        )

        sandbox = ExecutionSandbox(timeout_s=args.timeout)
        deps = PrincipalDeps(architect=architect, sandbox=sandbox)

        graph = build_principal_graph().compile()

        initial_state = {
            "goal": args.goal,
            "metric": metric,
            "dataset_path": args.benchmark,
            "max_trials": args.trials,
        }

        config_dict = {"configurable": {"deps": deps}}

        final_state = await graph.ainvoke(initial_state, config=config_dict)

    # Report
    print("\nOptimisation complete:")
    print(f"  Trials run: {final_state.get('current_trial', 0)}")
    print(f"  Best loss: {final_state.get('best_loss', float('inf')):.6f}")
    history = final_state.get("trial_history", [])
    if history:
        print("  Trial history:")
        for entry in history:
            print(f"    Trial {entry['trial']}: loss={entry['loss']:.6f}")
    _print_shared_context_metrics("architect", architect_shared_metrics)
    metrics_out_dir = Path("output")
    metrics_path = _write_shared_context_metrics_file(
        metrics_out_dir / "optimize_shared_context_metrics.json",
        {"architect": architect_shared_metrics},
    )
    if metrics_path is not None:
        print(f"  Shared context metrics: {metrics_path}")


async def _cmd_upsert_cdg(args: argparse.Namespace) -> None:
    """Upsert CDG JSON files into Memgraph graph store."""
    from ageom.config import AgeomConfig
    from ageom.upsert_cdg import upsert_repo

    config = AgeomConfig()
    if args.memgraph_uri:
        config.memgraph_uri = args.memgraph_uri

    repo_path = Path(args.repo_path).expanduser().resolve()
    if not repo_path.is_dir():
        print(f"Error: {repo_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    repo_name = args.repo_name or repo_path.name
    print(f"Upserting CDGs from {repo_path} as repo '{repo_name}'")

    summary = await upsert_repo(repo_path, repo_name, config)
    if summary:
        total_atoms = sum(c["atoms"] for c in summary.values())
        print(f"\nDone — {len(summary)} CDG(s), {total_atoms} atom(s) upserted.")
    else:
        print("No CDGs processed.")


def _cmd_visualize(args: argparse.Namespace) -> None:
    """Open browser-based CDG visualization."""

    static_dir = Path(__file__).resolve().parent / "static"
    if not static_dir.exists():
        print(f"Error: static directory not found at {static_dir}", file=sys.stderr)
        sys.exit(1)

    # API mode: start FastAPI with uvicorn
    if getattr(args, "api", False):
        try:
            import uvicorn
        except ImportError:
            print(
                "Error: uvicorn not installed. Install with: pip install 'ageo-matcher[visualizer]'",
                file=sys.stderr,
            )
            sys.exit(1)

        port = args.port or 8080
        url = f"http://127.0.0.1:{port}"
        print(f"Starting CDG Visualizer API at {url}")
        print(f"Telemetry dashboard: {url}/dashboard.html")
        print("Press Ctrl+C to stop")

        threading.Thread(
            target=webbrowser.open, args=(url,), daemon=True
        ).start()

        uvicorn.run(
            "ageom.visualizer_api:app",
            host="127.0.0.1",
            port=port,
            log_level="info",
            reload=getattr(args, "reload", False),
        )
        return

    default_cdg = static_dir / "default_cdg.json"

    # If a CDG file was provided, validate and copy it
    if args.cdg_file:
        cdg_path = Path(args.cdg_file)
        if not cdg_path.exists():
            print(f"Error: CDG file not found: {cdg_path}", file=sys.stderr)
            sys.exit(1)
        try:
            with open(cdg_path) as f:
                data = json.load(f)
            if not isinstance(data.get("nodes"), list):
                print("Error: CDG JSON must contain a 'nodes' array", file=sys.stderr)
                sys.exit(1)
            if not isinstance(data.get("edges"), list):
                print("Error: CDG JSON must contain an 'edges' array", file=sys.stderr)
                sys.exit(1)
        except json.JSONDecodeError as exc:
            print(f"Error: invalid JSON in {cdg_path}: {exc}", file=sys.stderr)
            sys.exit(1)
        shutil.copy2(str(cdg_path), str(default_cdg))

    try:
        if args.no_serve:
            # Open file:// directly
            index_html = static_dir / "index.html"
            url = index_html.as_uri()
            print(f"Opening {url}")
            webbrowser.open(url)
        else:
            # Start local HTTP server
            original_dir = os.getcwd()
            os.chdir(str(static_dir))

            handler = SimpleHTTPRequestHandler

            with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
                port = httpd.server_address[1]
                url = f"http://127.0.0.1:{port}/index.html"
                print(f"Serving CDG visualizer at {url}")
                print("Press Ctrl+C to stop")

                # Open browser in a thread so we don't block the server
                threading.Thread(
                    target=webbrowser.open, args=(url,), daemon=True
                ).start()

                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    print("\nShutting down server")
                finally:
                    os.chdir(original_dir)
    finally:
        # Clean up default_cdg.json
        if default_cdg.exists():
            default_cdg.unlink()


async def _cmd_profile(args: argparse.Namespace) -> None:
    """Evaluate an existing CDG against a dataset and rank error contributors."""
    from ageom.architect.handoff import load_json
    from ageom.principal.models import OptimizationMetric
    from ageom.principal.profiler import profile_algorithm_error
    from ageom.synthesizer.models import ExportBundle

    cdg_path = Path(args.cdg)
    if not cdg_path.exists():
        print(f"Error: CDG file not found at {cdg_path}", file=sys.stderr)
        sys.exit(1)

    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        print(f"Error: Artifact file not found at {artifact_path}", file=sys.stderr)
        sys.exit(1)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset file not found at {dataset_path}", file=sys.stderr)
        sys.exit(1)

    cdg = load_json(cdg_path)
    metric = OptimizationMetric(args.metric)

    bundle = ExportBundle(
        target="python-pkg",
        output_dir=artifact_path.parent,
        source_path=artifact_path,
        compiled_artifact=artifact_path,
    )

    print(f"Profiling {artifact_path.name} against {dataset_path.name} using metric {metric.value}...")
    
    try:
        gradients = await profile_algorithm_error(
            cdg=cdg,
            bundle=bundle,
            dataset_path=str(dataset_path),
            metric=metric,
        )

        if not gradients:
            print("No gradients were computed. Ensure trace.jsonl is emitted properly.")
            return

        print("\n=== Profiling Results ===")
        print(f"{'Node ID':<20} | {'Score (%)':<10} | {'Reason'}")
        print("-" * 80)
        for g in gradients:
            print(f"{g.node_id:<20} | {g.gradient_score:<10.2f} | {g.bottleneck_reason}")

    except Exception as exc:
        print(f"Error during profiling: {exc}", file=sys.stderr)
        sys.exit(1)


async def _cmd_prompt_benchmark(args: argparse.Namespace) -> None:
    """Run prompt-key A/B benchmarks across a small cross-domain suite."""
    from ageom.config import AgeomConfig
    from ageom.hunter.llm import create_llm_client
    from ageom.prompt_benchmark import (
        PromptBenchmarkProvider,
        format_prompt_benchmark_summary,
        run_prompt_benchmark,
        save_prompt_benchmark_report,
        select_prompt_benchmark_cases,
        summarize_prompt_benchmark,
    )

    config = AgeomConfig()
    try:
        provider_specs = _parse_prompt_benchmark_provider_specs(args.provider)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    providers: list[PromptBenchmarkProvider] = []
    try:
        for provider_name, model_name in provider_specs:
            client = create_llm_client(
                provider=provider_name,
                model=model_name,
                max_tokens=args.max_tokens or config.hunter_llm_max_tokens,
                anthropic_api_key=config.anthropic_api_key,
                openai_api_key=config.openai_api_key,
                openai_base_url=config.openai_base_url,
                llama_cpp_base_url=config.llama_cpp_base_url,
                llama_cpp_api_key=config.llama_cpp_api_key,
                use_agent_layer=config.use_agent_layer,
            )
            providers.append(
                PromptBenchmarkProvider(
                    name=f"{provider_name}:{model_name}",
                    client=client,
                )
            )

        cases = select_prompt_benchmark_cases(prompt_keys=args.prompt_key)
        results = await run_prompt_benchmark(
            providers=providers,
            cases=cases,
            repeats=args.repeats,
            compare_direct_baseline=bool(args.compare_direct_baseline),
        )
        aggregates = summarize_prompt_benchmark(results)
        print(format_prompt_benchmark_summary(aggregates))

        if args.output:
            save_prompt_benchmark_report(
                args.output,
                results=results,
                aggregates=aggregates,
            )
            print(f"\nSaved report: {args.output}")
    finally:
        for provider in providers:
            close = getattr(provider.client, "close", None)
            if not callable(close):
                continue
            maybe_result = close()
            if inspect.isawaitable(maybe_result):
                await maybe_result


if __name__ == "__main__":
    main()
