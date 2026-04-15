"""Compatibility wrapper for the provider-owned audit rollups backfill."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from sciona.atoms.provider_inventory import discover_audit_manifest_path
from sciona.atoms.supabase_backfill import (
    DEFAULT_BATCH_SIZE,
    backfill_audit_rollups,
    build_rollup_row,
    create_supabase_client_from_env,
    fetch_atom_lookup,
    load_manifest_entries,
)

log = logging.getLogger(__name__)

load_manifest = load_manifest_entries


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=discover_audit_manifest_path(),
        help="Path to audit_manifest.json",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("AUDIT_ROLLUP_BATCH_SIZE", DEFAULT_BATCH_SIZE)),
        help="Upsert batch size",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build rows without upserting them")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    stats = backfill_audit_rollups(
        create_supabase_client_from_env(),
        manifest_path=args.manifest_path,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
    log.info("Audit rollup backfill complete: %s", stats)


if __name__ == "__main__":
    main()
