"""Commands for managing multi-repo atom sources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
