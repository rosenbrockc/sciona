"""CLI entrypoint for AGEO-Matcher."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


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
        "--prover", choices=["lean4", "coq", "python"], required=True, help="Proof assistant"
    )
    build_parser.add_argument(
        "--path", type=str, default="", help="Path to Coq project (for --prover coq)"
    )
    build_parser.add_argument(
        "--packages", type=str, default=None,
        help="Comma-separated Python packages to index (for --prover python, default: numpy,scipy)",
    )
    build_parser.add_argument(
        "--output", type=str, default=None, help="Output directory for index (default: from .env)"
    )

    # --- skill ---
    skill_parser = subparsers.add_parser("skill", help="Manage the algorithmic skill catalog")
    skill_sub = skill_parser.add_subparsers(dest="skill_command")

    ingest_parser = skill_sub.add_parser("ingest", help="Ingest primitives from a source")
    ingest_parser.add_argument(
        "--source", choices=["clrs", "coq100"], required=True, help="Source to ingest from"
    )
    ingest_parser.add_argument(
        "--path", type=str, required=True, help="Path to the cloned source repo"
    )
    ingest_parser.add_argument(
        "--output", type=str, default=None, help="Output path for catalog JSON"
    )

    skill_index_parser = skill_sub.add_parser("index", help="Build FAISS skill index from catalog")
    skill_index_parser.add_argument(
        "--catalog", type=str, default=None, help="Path to catalog JSON (default: auto-detect)"
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
        "--max-depth", type=int, default=None, help="Max decomposition depth (default: from config)"
    )
    decompose_parser.add_argument(
        "--output", type=str, default=None, help="Output path for CDG JSON"
    )
    decompose_parser.add_argument(
        "--catalog", type=str, default=None, help="Path to catalog JSON (default: auto-detect)"
    )
    decompose_parser.add_argument(
        "--thread-id", type=str, default=None,
        help="Checkpoint thread ID (auto-generated if omitted)",
    )
    decompose_parser.add_argument(
        "--llm-provider",
        choices=["anthropic", "codex"],
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
        "--no-persist", action="store_true", default=False,
        help="Disable PostgreSQL persistence (use in-memory only)",
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
        "cdg_file", nargs="?", default=None,
        help="Path to CDG JSON to pre-load (optional)",
    )
    viz_parser.add_argument(
        "--port", type=int, default=0,
        help="HTTP server port (default: auto-pick)",
    )
    viz_parser.add_argument(
        "--no-serve", action="store_true", default=False,
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
        "--prover", choices=["lean4", "coq", "python"], default="lean4", help="Proof assistant"
    )
    assemble_parser.add_argument(
        "--output", type=str, default=None, help="Output path for generated source file"
    )
    assemble_parser.add_argument(
        "--check", action="store_true", default=False,
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
        "--prover", choices=["lean4", "coq", "python"], default="lean4", help="Proof assistant"
    )
    synth_parser.add_argument(
        "--output", type=str, default=None, help="Output path for final verified source"
    )
    synth_parser.add_argument(
        "--max-iterations", type=int, default=None,
        help="Max repair iterations (default: from config)",
    )
    synth_parser.add_argument(
        "--llm-provider",
        choices=["anthropic", "codex"],
        default=None,
        help="LLM provider override (default: from config)",
    )
    synth_parser.add_argument(
        "--llm-model", type=str, default=None,
        help="LLM model override (default: from config)",
    )
    synth_parser.add_argument(
        "--llm-max-tokens", type=int, default=None,
        help="Max output tokens for LLM calls",
    )

    # --- export ---
    export_parser = subparsers.add_parser(
        "export", help="Export verified source to compiled artifacts and FFI bindings"
    )
    export_parser.add_argument(
        "source_file", type=str,
        help="Path to verified .lean/.v file or SynthesisResult JSON",
    )
    export_parser.add_argument(
        "--target",
        choices=["lean-lib", "coq-lib", "rust-ffi", "c-header", "python-pkg"],
        default="lean-lib",
        help="Export target (default: lean-lib)",
    )
    export_parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: from config)",
    )
    export_parser.add_argument(
        "--optimize", action="store_true", default=False,
        help="Run hot-path optimizer before export",
    )
    export_parser.add_argument(
        "--prover", choices=["lean4", "coq", "python"], default="lean4",
        help="Proof assistant (default: lean4)",
    )

    # --- match ---
    match_parser = subparsers.add_parser("match", help="Match predicates to library functions")
    match_parser.add_argument("--statement", type=str, help="Single statement to match")
    match_parser.add_argument("--pdg-file", type=str, help="JSON file with PDG nodes")
    match_parser.add_argument(
        "--prover", choices=["lean4", "coq", "python"], default="lean4", help="Proof assistant"
    )
    match_parser.add_argument(
        "--index-dir", type=str, default=None, help="Directory containing FAISS index (default: from .env)"
    )
    match_parser.add_argument(
        "--llm-provider",
        choices=["anthropic", "codex"],
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
            print("Error: provide a skill subcommand (ingest, index, search)", file=sys.stderr)
            sys.exit(1)
    elif args.command == "decompose":
        asyncio.run(_cmd_decompose(args))
    elif args.command == "history":
        asyncio.run(_cmd_history(args))
    elif args.command == "match":
        asyncio.run(_cmd_match(args))
    elif args.command == "assemble":
        asyncio.run(_cmd_assemble(args))
    elif args.command == "synthesize":
        asyncio.run(_cmd_synthesize(args))
    elif args.command == "export":
        asyncio.run(_cmd_export(args))
    elif args.command == "visualize":
        _cmd_visualize(args)
    else:
        parser.print_help()
        sys.exit(1)


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
        print("Error: no primitives found. Run 'ageom skill ingest' first.", file=sys.stderr)
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
        print(f"Error: skill index not found at {index_dir}. Run 'ageom skill index' first.",
              file=sys.stderr)
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

        packages = args.packages.split(",") if getattr(args, "packages", None) else config.python_packages.split(",")
        py_source = PythonDeclarationSource()
        declarations = []
        for pkg in packages:
            pkg = pkg.strip()
            if pkg:
                declarations.extend(py_source.get_declarations_from_package(pkg))
        print(f"Found {len(declarations)} declarations from {', '.join(packages)}")
    else:
        print(f"Error: unsupported prover {prover}", file=sys.stderr)
        sys.exit(1)

    builder = IndexBuilder()
    store = builder.build_from_declarations(
        declarations, source_lib=args.path or "Mathlib", prover=prover
    )
    store.save(output_dir)
    print(f"Index saved to {output_dir} ({store.size} entries)")


async def _cmd_decompose(args: argparse.Namespace) -> None:
    """Decompose a goal into a Conceptual Dependency Graph."""
    from ageom.architect.catalog import PrimitiveCatalog
    from ageom.architect.checkpointer import create_checkpointer
    from ageom.architect.embedder import SkillIndex
    from ageom.architect.graph import DecompositionAgent
    from ageom.architect.handoff import save_json
    from ageom.architect.models import NodeStatus
    from ageom.config import AgeomConfig
    from ageom.hunter.llm import create_llm_client

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
        print("Warning: no catalog loaded. Decomposition will have no atomic stop conditions.",
              file=sys.stderr)

    # Load skill index (falls back to empty)
    skill_index = SkillIndex(index_dir=config.skill_index_dir)
    if config.skill_index_dir.exists():
        try:
            skill_index = SkillIndex.load(config.skill_index_dir)
        except Exception:
            pass  # Use empty index

    # Set up LLM
    llm_provider = args.llm_provider or config.architect_llm_provider or config.llm_provider
    llm_model = args.llm_model or config.architect_llm_model
    llm_max_tokens = args.llm_max_tokens or config.llm_max_tokens
    try:
        llm = create_llm_client(
            provider=llm_provider,
            model=llm_model,
            max_tokens=llm_max_tokens,
            anthropic_api_key=config.anthropic_api_key,
            openai_api_key=config.openai_api_key,
            openai_base_url=config.openai_base_url,
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

        print(f"\nDecomposition complete:")
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
    from ageom.hunter.llm import create_llm_client
    from ageom.indexer.builder import SemanticIndexImpl
    from ageom.indexer.embedder import UniXcoderEmbedder
    from ageom.indexer.faiss_store import FAISSStore
    from ageom.judge.checker import VerificationOracleImpl
    from ageom.judge.lean_env import LeanEnvironment
    from ageom.types import PDGNode, Prover

    config = AgeomConfig()

    # Load index
    index_dir = Path(args.index_dir) if args.index_dir else config.index_dir
    if not index_dir.exists():
        print(f"Error: index directory {index_dir} not found. Run 'ageom index build' first.", file=sys.stderr)
        sys.exit(1)

    store = FAISSStore.load(index_dir)
    embedder = UniXcoderEmbedder(config.embedding_model)
    index = SemanticIndexImpl(store, embedder)

    # Set up verification oracle
    prover = Prover(args.prover)
    if prover == Prover.LEAN4:
        lean_env = LeanEnvironment(config.lean_toolchain)
        oracle = VerificationOracleImpl(lean_env=lean_env)
    elif prover == Prover.PYTHON:
        from ageom.judge.python_env import PythonEnvironment

        python_env = PythonEnvironment(
            mypy_path=config.python_mypy_path,
            python_path=config.python_path,
        )
        oracle = VerificationOracleImpl(python_env=python_env)
    else:
        from ageom.judge.coq_env import CoqEnvironment

        coq_env = CoqEnvironment(config.coq_project_path)
        oracle = VerificationOracleImpl(coq_env=coq_env)

    # Set up LLM
    llm_provider = args.llm_provider or config.llm_provider
    llm_model = args.llm_model or config.llm_model
    llm_max_tokens = args.llm_max_tokens or config.llm_max_tokens
    try:
        llm = create_llm_client(
            provider=llm_provider,
            model=llm_model,
            max_tokens=llm_max_tokens,
            anthropic_api_key=config.anthropic_api_key,
            openai_api_key=config.openai_api_key,
            openai_base_url=config.openai_base_url,
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
            print(f"  Type: {result.verified_match.candidate.declaration.type_signature}")
        else:
            print(f"  NO MATCH FOUND ({len(result.all_candidates)} candidates tried)")
            for vr in result.all_verifications:
                print(f"    - {vr.candidate.declaration.name}: {vr.error_message[:100]}")


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
        if prover == Prover.LEAN4:
            from ageom.judge.lean_env import LeanEnvironment

            env = LeanEnvironment(config.lean_toolchain)
        elif prover == Prover.PYTHON:
            from ageom.judge.python_env import PythonEnvironment

            env = PythonEnvironment(
                mypy_path=config.python_mypy_path,
                python_path=config.python_path,
            )
        else:
            from ageom.judge.coq_env import CoqEnvironment

            env = CoqEnvironment(config.coq_project_path)

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

    print(f"\nExport complete:")
    print(f"  Target: {bundle.target}")
    print(f"  Source: {bundle.source_path}")
    if bundle.compiled_artifact:
        print(f"  Artifact: {bundle.compiled_artifact}")
    if bundle.ffi_files:
        print(f"  FFI files:")
        for f in bundle.ffi_files:
            print(f"    {f}")
    if bundle.certificate:
        print(f"  Certificate: {output_dir / 'certificate.json'}")
        print(f"    Source hash: {bundle.certificate.source_hash[:16]}...")
    if bundle.errors:
        print(f"  Errors:")
        for err in bundle.errors:
            print(f"    {err}")


async def _cmd_synthesize(args: argparse.Namespace) -> None:
    """Assemble CDG + match results, then repair via the synthesizer agent."""
    from ageom.architect.handoff import load_json
    from ageom.config import AgeomConfig
    from ageom.hunter.llm import create_llm_client
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

    print(f"Assembled skeleton: {len(skeleton.units)} units, {skeleton.sorry_count} sorrys")

    # Set up ProofEnvironment
    if prover == Prover.LEAN4:
        from ageom.judge.lean_env import LeanEnvironment

        env = LeanEnvironment(config.lean_toolchain)
    elif prover == Prover.PYTHON:
        from ageom.judge.python_env import PythonEnvironment

        env = PythonEnvironment(
            mypy_path=config.python_mypy_path,
            python_path=config.python_path,
        )
    else:
        from ageom.judge.coq_env import CoqEnvironment

        env = CoqEnvironment(config.coq_project_path)

    # Set up LLM
    llm_provider = (
        args.llm_provider
        or config.synthesizer_llm_provider
        or config.llm_provider
    )
    llm_model = args.llm_model or config.synthesizer_llm_model
    llm_max_tokens = args.llm_max_tokens or config.llm_max_tokens
    try:
        llm = create_llm_client(
            provider=llm_provider,
            model=llm_model,
            max_tokens=llm_max_tokens,
            anthropic_api_key=config.anthropic_api_key,
            openai_api_key=config.openai_api_key,
            openai_base_url=config.openai_base_url,
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
        result = agent_result = await agent.synthesize(skeleton)

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
            print(f"  Errors encountered:")
            for it, cat, text in result.error_history:
                print(f"    [{it}] {cat}: {text[:80]}")
    finally:
        await env.close()


def _cmd_visualize(args: argparse.Namespace) -> None:
    """Open browser-based CDG visualization."""
    import shutil
    import socketserver
    import threading
    import webbrowser
    from http.server import SimpleHTTPRequestHandler

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
