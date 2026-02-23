"""CLI entrypoint for AGEO-Matcher."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import socketserver
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ageom.config import AgeomConfig
    from ageom.hunter.llm import LLMClient
    from ageom.protocols import ProofEnvironment
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
        choices=["anthropic", "codex", "llama_cpp", "claude_cli", "codex_cli", "gemini_cli"],
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
        choices=["anthropic", "codex", "llama_cpp", "claude_cli", "codex_cli", "gemini_cli"],
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
        choices=["anthropic", "codex", "llama_cpp", "claude_cli", "codex_cli", "gemini_cli"],
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
        choices=["anthropic", "codex", "llama_cpp", "claude_cli", "codex_cli", "gemini_cli"],
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
        help="Ingest an existing Python class into the atom framework (Round 0)",
    )
    ingest_parser.add_argument("source", type=str, help="Path to Python source file")
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
        choices=["anthropic", "codex", "llama_cpp", "claude_cli", "codex_cli", "gemini_cli"],
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
        choices=["anthropic", "codex", "llama_cpp", "claude_cli", "codex_cli", "gemini_cli"],
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
        asyncio.run(_cmd_optimize(args))
    elif args.command == "decompose":
        asyncio.run(_cmd_decompose(args))
    elif args.command == "history":
        asyncio.run(_cmd_history(args))
    elif args.command == "ingest":
        asyncio.run(_cmd_ingest(args))
    elif args.command == "match":
        asyncio.run(_cmd_match(args))
    elif args.command == "assemble":
        asyncio.run(_cmd_assemble(args))
    elif args.command == "synthesize":
        asyncio.run(_cmd_synthesize(args))
    elif args.command == "run":
        asyncio.run(_cmd_run(args))
    elif args.command == "export":
        asyncio.run(_cmd_export(args))
    elif args.command == "visualize":
        _cmd_visualize(args)
    else:
        parser.print_help()
        sys.exit(1)


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
    from ageom.architect.catalog import PrimitiveCatalog
    from ageom.architect.embedder import SkillIndex
    from ageom.config import AgeomConfig

    config = AgeomConfig()
    output_dir = Path(args.output) if args.output else config.skill_index_dir

    # Auto-detect catalogs if no explicit path given
    catalog = PrimitiveCatalog()
    if args.catalog:
        catalog = PrimitiveCatalog.load(args.catalog)
    else:
        # Load all catalog_*.json files from the skill index dir
        search_dir = config.skill_index_dir
        if search_dir.exists():
            for cat_file in sorted(search_dir.glob("catalog_*.json")):
                print(f"Loading catalog: {cat_file.name}")
                partial = PrimitiveCatalog.load(cat_file)
                for prim in partial.all_primitives():
                    catalog.add(prim)

    if catalog.size == 0:
        print(
            "Error: no primitives found. Run 'ageom skill ingest' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Building skill index from {catalog.size} primitives...")
    index = SkillIndex(index_dir=output_dir)
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

    builder = IndexBuilder()
    store = builder.build_from_declarations(
        declarations, source_lib=args.path or "Mathlib", prover=prover
    )
    store.save(output_dir)
    print(f"Index saved to {output_dir} ({store.size} entries)")


async def _cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest an existing Python class into the atom framework."""
    from ageom.config import AgeomConfig
    from ageom.ingester import IngesterAgent
    from ageom.types import Prover

    config = AgeomConfig()

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: source file not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    # Set up LLM
    try:
        from ageom.llm_router import (
            INGESTER_ABSTRACT,
            INGESTER_CHUNK,
            INGESTER_DECOMPOSE,
            INGESTER_FIX_GHOST,
            INGESTER_FIX_TYPE,
            INGESTER_HOIST_STATE,
            INGESTER_OPAQUE_WITNESS,
        )

        llm = _create_llm_router(
            args,
            config,
            "ingester",
            [
                INGESTER_CHUNK,
                INGESTER_HOIST_STATE,
                INGESTER_ABSTRACT,
                INGESTER_FIX_TYPE,
                INGESTER_FIX_GHOST,
                INGESTER_OPAQUE_WITNESS,
                INGESTER_DECOMPOSE,
            ],
        )
    except (ValueError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Set up proof environment (Python/mypy)
    proof_env = _create_proof_env(Prover.PYTHON, config)

    # Optionally load FAISS index
    faiss_index = None
    if config.index_dir.exists():
        try:
            from ageom.indexer.builder import SemanticIndexImpl
            from ageom.indexer.embedder import UniXcoderEmbedder
            from ageom.indexer.faiss_store import FAISSStore

            store = FAISSStore.load(config.index_dir)
            embedder = UniXcoderEmbedder(config.embedding_model)
            faiss_index = SemanticIndexImpl(store, embedder)
        except Exception as exc:
            print(f"Warning: failed to load FAISS index: {exc}", file=sys.stderr)

    output_dir = Path(args.output) if args.output else Path("output") / args.class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        agent = IngesterAgent(
            llm=llm,
            proof_env=proof_env,
            faiss_index=faiss_index,
            output_dir=str(output_dir),
            max_depth=config.ingester_max_depth,
            line_threshold=config.ingester_decompose_line_threshold,
        )

        print(f"Ingesting {'class' if not getattr(args, 'procedural', False) else 'procedural'} '{args.class_name}' from {source_path}")
        if getattr(args, "procedural", False):
            bundle = await agent.ingest_procedural(str(source_path), args.class_name)
        else:
            bundle = await agent.ingest(str(source_path), args.class_name)

        # Write output files
        if bundle.generated_atoms:
            (output_dir / "atoms.py").write_text(bundle.generated_atoms)
        if bundle.generated_state_models:
            (output_dir / "state_models.py").write_text(bundle.generated_state_models)
        if bundle.generated_witnesses:
            (output_dir / "witnesses.py").write_text(bundle.generated_witnesses)

        # Write CDG JSON
        from ageom.architect.handoff import save_json

        save_json(bundle.cdg, output_dir / "cdg.json")

        # Write match results
        if bundle.match_results:
            matches_data = [mr.to_dict() for mr in bundle.match_results]
            with open(output_dir / "matches.json", "w") as f:
                json.dump(matches_data, f, indent=2)

        print("\nIngestion complete:")
        print(f"  CDG: {len(bundle.cdg.nodes)} nodes, {len(bundle.cdg.edges)} edges")
        print(f"  Matches: {len(bundle.match_results)}")
        print(f"  mypy passed: {bundle.mypy_passed}")
        print(f"  Ghost sim passed: {bundle.ghost_sim_passed}")
        print(f"  Output: {output_dir}/")

        # Write trace if requested
        if getattr(args, "trace", False):
            from ageom.telemetry import get_event_log

            event_log = get_event_log()
            if len(event_log) > 0:
                event_log.save(output_dir / "trace.jsonl")
                print(
                    f"  Trace: {output_dir / 'trace.jsonl'} ({len(event_log)} events)"
                )
    finally:
        await proof_env.close()


async def _cmd_decompose(args: argparse.Namespace) -> None:
    """Decompose a goal into a Conceptual Dependency Graph."""
    from ageom.architect.catalog import PrimitiveCatalog
    from ageom.architect.checkpointer import create_checkpointer
    from ageom.architect.embedder import SkillIndex
    from ageom.architect.graph import DecompositionAgent
    from ageom.architect.handoff import save_json
    from ageom.config import AgeomConfig

    config = AgeomConfig()
    max_depth = args.max_depth or config.architect_max_depth

    # Load catalog
    catalog = PrimitiveCatalog()
    if args.catalog:
        catalog = PrimitiveCatalog.load(args.catalog)
    else:
        search_dir = config.skill_index_dir
        if search_dir.exists():
            for cat_file in sorted(search_dir.glob("catalog_*.json")):
                print(f"Loading catalog: {cat_file.name}")
                partial = PrimitiveCatalog.load(cat_file)
                for prim in partial.all_primitives():
                    catalog.add(prim)

    if catalog.size == 0:
        print(
            "Warning: no catalog loaded. Decomposition will have no atomic stop conditions.",
            file=sys.stderr,
        )

    # Load skill index (falls back to empty)
    skill_index = SkillIndex(index_dir=config.skill_index_dir)
    if config.skill_index_dir.exists():
        try:
            skill_index = SkillIndex.load(config.skill_index_dir)
        except Exception as exc:
            print(f"Warning: failed to load skill index: {exc}", file=sys.stderr)

    # Set up LLM
    try:
        from ageom.llm_router import (
            ARCHITECT_CRITIQUE,
            ARCHITECT_DECOMPOSE,
            ARCHITECT_STRATEGY,
        )

        llm = _create_llm_router(
            args,
            config,
            "architect",
            [
                ARCHITECT_STRATEGY,
                ARCHITECT_DECOMPOSE,
                ARCHITECT_CRITIQUE,
            ],
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ImportError as exc:
        print(f"Error: missing LLM dependency ({exc})", file=sys.stderr)
        sys.exit(1)

    # Determine persistence URI
    postgres_uri = "" if args.no_persist else config.postgres_uri

    async with create_checkpointer(postgres_uri) as checkpointer:
        agent = DecompositionAgent(
            catalog=catalog,
            skill_index=skill_index,
            llm=llm,
            max_depth=max_depth,
            checkpointer=checkpointer,
        )

        print(f"Decomposing: {args.goal}")
        print(f"  Max depth: {max_depth}, Catalog size: {catalog.size}")

        cdg = await agent.decompose(args.goal, thread_id=args.thread_id)

        thread_id = cdg.metadata.get("thread_id", "")
        print(f"  Thread ID: {thread_id}")

        # Print summary
        by_status: dict[str, int] = {}
        for node in cdg.nodes:
            status = node.status.value
            by_status[status] = by_status.get(status, 0) + 1

        print("\nDecomposition complete:")
        print(f"  Nodes: {len(cdg.nodes)}, Edges: {len(cdg.edges)}")
        for status, count in sorted(by_status.items()):
            print(f"    {status}: {count}")
        print(f"  Complete: {cdg.is_complete()}")

        # Save output
        if args.output:
            save_json(cdg, args.output)
            print(f"  Saved to: {args.output}")


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
    from ageom.config import AgeomConfig
    from ageom.hunter.graph import HunterAgent
    from ageom.indexer.builder import SemanticIndexImpl
    from ageom.indexer.embedder import UniXcoderEmbedder
    from ageom.indexer.faiss_store import FAISSStore
    from ageom.judge.checker import VerificationOracleImpl
    from ageom.types import PDGNode, Prover

    config = AgeomConfig()

    # Load index
    index_dir = Path(args.index_dir) if args.index_dir else config.index_dir
    if not index_dir.exists():
        print(
            f"Error: index directory {index_dir} not found. Run 'ageom index build' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    store = FAISSStore.load(index_dir)
    embedder = UniXcoderEmbedder(config.embedding_model)
    index = SemanticIndexImpl(store, embedder)

    # Set up verification oracle
    prover = Prover(args.prover)
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

        llm = _create_llm_router(
            args,
            config,
            "hunter",
            [
                HUNTER_SCORE,
                HUNTER_REFORMULATE,
                HUNTER_ANALYZE_FAILURE,
            ],
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ImportError as exc:
        print(f"Error: missing LLM dependency ({exc})", file=sys.stderr)
        sys.exit(1)

    agent = HunterAgent(
        index=index,
        oracle=oracle,
        llm=llm,
        max_iterations=config.hunter_max_iterations,
        top_k_verify=config.hunter_top_k_verify,
        search_k=config.hunter_search_k,
        mode=config.hunter_mode,
        use_gbnf=config.hunter_use_gbnf,
        query_batch_size=config.hunter_query_batch_size,
        top_k_per_query=config.hunter_top_k_per_query,
        max_candidates_total=config.hunter_max_candidates_total,
    )

    # Build PDG nodes
    nodes: list[PDGNode] = []
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
    from ageom.config import AgeomConfig
    from ageom.synthesizer.agent import SynthesizerAgent
    from ageom.synthesizer.assembler import Assembler, AssemblyError
    from ageom.types import MatchResult, Prover

    config = AgeomConfig()

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

        llm = _create_llm_router(
            args,
            config,
            "synthesizer",
            [
                SYNTHESIZER_REPAIR,
                SYNTHESIZER_TACTIC,
            ],
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ImportError as exc:
        print(f"Error: missing LLM dependency ({exc})", file=sys.stderr)
        sys.exit(1)

    max_iterations = args.max_iterations or config.synthesizer_max_iterations

    try:
        agent = SynthesizerAgent(env=env, llm=llm, max_iterations=max_iterations)
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
    from ageom.architect.catalog import PrimitiveCatalog
    from ageom.architect.checkpointer import create_checkpointer
    from ageom.architect.embedder import SkillIndex
    from ageom.architect.graph import DecompositionAgent
    from ageom.architect.handoff import save_json
    from ageom.config import AgeomConfig
    from ageom.hunter.graph import HunterAgent
    from ageom.indexer.builder import SemanticIndexImpl
    from ageom.indexer.embedder import UniXcoderEmbedder
    from ageom.indexer.faiss_store import FAISSStore
    from ageom.judge.checker import VerificationOracleImpl
    from ageom.orchestrator import run_orchestration
    from ageom.types import Prover

    config = AgeomConfig()
    prover = Prover(args.prover)

    # Load catalog
    catalog = PrimitiveCatalog()
    if args.catalog:
        catalog = PrimitiveCatalog.load(args.catalog)
    else:
        search_dir = config.skill_index_dir
        if search_dir.exists():
            for cat_file in sorted(search_dir.glob("catalog_*.json")):
                partial = PrimitiveCatalog.load(cat_file)
                for prim in partial.all_primitives():
                    catalog.add(prim)

    # Load skill index
    skill_index = SkillIndex(index_dir=config.skill_index_dir)
    if config.skill_index_dir.exists():
        try:
            skill_index = SkillIndex.load(config.skill_index_dir)
        except Exception as exc:
            print(f"Warning: failed to load skill index: {exc}", file=sys.stderr)

    # Set up LLM
    try:
        from ageom.llm_router import (
            ARCHITECT_CRITIQUE,
            ARCHITECT_DECOMPOSE,
            ARCHITECT_STRATEGY,
            ORCHESTRATOR_REFINE,
        )

        llm = _create_llm_router(
            args,
            config,
            "architect",
            [
                ARCHITECT_STRATEGY,
                ARCHITECT_DECOMPOSE,
                ARCHITECT_CRITIQUE,
                ORCHESTRATOR_REFINE,
            ],
        )
    except (ValueError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Step 1: Decompose
    print(f"Decomposing: {args.goal}")

    async with create_checkpointer("") as checkpointer:
        architect = DecompositionAgent(
            catalog=catalog,
            skill_index=skill_index,
            llm=llm,
            checkpointer=checkpointer,
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
        sys.exit(1)

    store = FAISSStore.load(index_dir)
    embedder = UniXcoderEmbedder(config.embedding_model)
    index = SemanticIndexImpl(store, embedder)

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

        hunter_llm = _create_llm_router(
            args,
            config,
            "hunter",
            [
                HUNTER_SCORE,
                HUNTER_REFORMULATE,
                HUNTER_ANALYZE_FAILURE,
            ],
        )
    except (ValueError, ImportError) as exc:
        print(f"Error setting up hunter LLM: {exc}", file=sys.stderr)
        sys.exit(1)

    hunter = HunterAgent(
        index=index,
        oracle=oracle,
        llm=hunter_llm,
        max_iterations=config.hunter_max_iterations,
        top_k_verify=config.hunter_top_k_verify,
        search_k=config.hunter_search_k,
        mode=config.hunter_mode,
        use_gbnf=config.hunter_use_gbnf,
        query_batch_size=config.hunter_query_batch_size,
        top_k_per_query=config.hunter_top_k_per_query,
        max_candidates_total=config.hunter_max_candidates_total,
    )

    # Step 3: Run orchestration loop
    print(f"Running orchestration (max {args.max_rounds} rounds)...")
    try:
        result = await run_orchestration(
            cdg,
            hunter_agent=hunter,
            llm=llm,
            prover=prover,
            max_rounds=args.max_rounds,
        )
    finally:
        await env.close()

    # Output
    print("\nOrchestration complete:")
    print(f"  Rounds used: {result.rounds_used}")
    print(
        f"  Matches: {sum(1 for mr in result.match_results if mr.success)}/{len(result.match_results)}"
    )
    if result.ungroundable:
        print(f"  Ungroundable: {result.ungroundable}")

    output_dir = Path(args.output) if args.output else Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(result.cdg, output_dir / "cdg.json")

    if result.match_results:

        matches_data = [mr.to_dict() for mr in result.match_results]
        with open(output_dir / "matches.json", "w") as f:
            json.dump(matches_data, f, indent=2)

    print(f"  Output: {output_dir}/")

    if getattr(args, "trace", False):
        from ageom.telemetry import get_event_log

        event_log = get_event_log()
        if len(event_log) > 0:
            event_log.save(output_dir / "trace.jsonl")
            print(f"  Trace: {output_dir / 'trace.jsonl'} ({len(event_log)} events)")


async def _cmd_optimize(args: argparse.Namespace) -> None:
    """Run the Principal NAS/AutoML optimisation loop."""
    from ageom.architect.catalog import PrimitiveCatalog
    from ageom.architect.checkpointer import create_checkpointer
    from ageom.architect.embedder import SkillIndex
    from ageom.architect.graph import DecompositionAgent
    from ageom.config import AgeomConfig
    from ageom.principal.evaluator import ExecutionSandbox
    from ageom.principal.graph import (
        PrincipalDeps,
        build_principal_graph,
    )
    from ageom.principal.models import OptimizationMetric

    config = AgeomConfig()

    # Load catalog
    catalog = PrimitiveCatalog()
    if args.catalog:
        catalog = PrimitiveCatalog.load(args.catalog)
    else:
        search_dir = config.skill_index_dir
        if search_dir.exists():
            for cat_file in sorted(search_dir.glob("catalog_*.json")):
                partial = PrimitiveCatalog.load(cat_file)
                for prim in partial.all_primitives():
                    catalog.add(prim)

    # Load skill index
    skill_index = SkillIndex(index_dir=config.skill_index_dir)
    if config.skill_index_dir.exists():
        try:
            skill_index = SkillIndex.load(config.skill_index_dir)
        except Exception as exc:
            print(f"Warning: failed to load skill index: {exc}", file=sys.stderr)

    # LLM
    try:
        from ageom.llm_router import (
            ARCHITECT_CRITIQUE,
            ARCHITECT_DECOMPOSE,
            ARCHITECT_STRATEGY,
        )

        llm = _create_llm_router(
            args,
            config,
            "architect",
            [
                ARCHITECT_STRATEGY,
                ARCHITECT_DECOMPOSE,
                ARCHITECT_CRITIQUE,
            ],
        )
    except (ValueError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    metric = OptimizationMetric(args.metric)
    postgres_uri = "" if args.no_persist else config.postgres_uri

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


def _cmd_visualize(args: argparse.Namespace) -> None:
    """Open browser-based CDG visualization."""

    static_dir = Path(__file__).resolve().parent / "static"
    if not static_dir.exists():
        print(f"Error: static directory not found at {static_dir}", file=sys.stderr)
        sys.exit(1)

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


if __name__ == "__main__":
    main()
