"""Command for upserting CDG files into Memgraph."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
