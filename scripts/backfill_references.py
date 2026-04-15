#!/usr/bin/env python3
"""Compatibility wrapper for the provider-owned atom references backfill."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sciona.atoms.provider_inventory import discover_references_registry_path
from sciona.atoms.supabase_backfill import (
    backfill_references,
    build_atom_reference_row,
    build_ref_key,
    create_supabase_client_from_env,
    extract_fqdn,
    iter_reference_files,
    load_registry,
    map_source,
)

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atoms-root",
        type=Path,
        default=None,
        help="Optional single artifact root containing references.json files.",
    )
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
    stats = backfill_references(
        create_supabase_client_from_env(),
        atoms_root=args.atoms_root,
        registry_path=args.registry_path,
        dry_run=args.dry_run,
    )
    log.info("Atom reference backfill complete: %s", stats)
    return 1 if stats.get("errors") else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
