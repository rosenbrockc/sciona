"""CLI entrypoint for AGEO-Matcher."""

from __future__ import annotations

import argparse
import asyncio
import json
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
        "--prover", choices=["lean4", "coq"], required=True, help="Proof assistant"
    )
    build_parser.add_argument(
        "--path", type=str, default="", help="Path to Coq project (for --prover coq)"
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
        "--no-persist", action="store_true", default=False,
        help="Disable PostgreSQL persistence (use in-memory only)",
    )

    # --- history ---
    history_parser = subparsers.add_parser(
        "history", help="Show checkpoint history for a decomposition thread"
    )
    history_parser.add_argument("thread_id", type=str, help="Thread ID to inspect")

    # --- match ---
    match_parser = subparsers.add_parser("match", help="Match predicates to library functions")
    match_parser.add_argument("--statement", type=str, help="Single statement to match")
    match_parser.add_argument("--pdg-file", type=str, help="JSON file with PDG nodes")
    match_parser.add_argument(
        "--prover", choices=["lean4", "coq"], default="lean4", help="Proof assistant"
    )
    match_parser.add_argument(
        "--index-dir", type=str, default=None, help="Directory containing FAISS index (default: from .env)"
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
    from ageom.hunter.llm import ClaudeLLMClient

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
    if not config.anthropic_api_key:
        print("Error: AGEOM_ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    llm = ClaudeLLMClient(
        api_key=config.anthropic_api_key,
        model=config.architect_llm_model,
    )

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
    from ageom.hunter.llm import ClaudeLLMClient
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
    else:
        from ageom.judge.coq_env import CoqEnvironment

        coq_env = CoqEnvironment(config.coq_project_path)
        oracle = VerificationOracleImpl(coq_env=coq_env)

    # Set up LLM
    llm = ClaudeLLMClient(
        api_key=config.anthropic_api_key,
        model=config.llm_model,
        max_tokens=config.llm_max_tokens,
    )

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


if __name__ == "__main__":
    main()
