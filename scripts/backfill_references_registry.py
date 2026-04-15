#!/usr/bin/env python3
"""Compatibility wrapper for the provider-owned references registry backfill."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sciona.atoms.provider_inventory import discover_references_registry_path
from sciona.atoms.supabase_backfill import (
    backfill_references_registry,
    build_registry_row,
    create_supabase_client_from_env,
    load_registry,
)

log = logging.getLogger(__name__)

backfill_registry = backfill_references_registry


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=discover_references_registry_path(),
        help="Path to the canonical references registry JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log planned upserts without writing to Supabase.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    stats = backfill_references_registry(
        create_supabase_client_from_env(),
        registry_path=args.registry_path,
        dry_run=args.dry_run,
    )
    log.info("Registry backfill complete: %s", stats)
    return 1 if stats.get("errors") else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
