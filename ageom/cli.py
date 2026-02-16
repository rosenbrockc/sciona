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
    elif args.command == "match":
        asyncio.run(_cmd_match(args))
    else:
        parser.print_help()
        sys.exit(1)


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
