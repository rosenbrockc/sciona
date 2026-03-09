"""Commands for index management, skill catalog, and catalog gap detection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ageom.commands._helpers import (
    _load_architect_catalog,
    _load_skill_index_or_empty,
)


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

    catalog, _catalog_alignment = _load_architect_catalog(args, config)

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


def _cmd_catalog_gaps(args: argparse.Namespace) -> None:
    """Detect catalog coverage gaps from a CDG file."""
    from ageom.architect.handoff import CDGExport
    from ageom.architect.models import AlgorithmicNode, NodeStatus
    from ageom.config import AgeomConfig

    config = AgeomConfig()
    catalog, _catalog_alignment = _load_architect_catalog(args, config)

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
