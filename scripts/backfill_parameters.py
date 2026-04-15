"""Compatibility wrapper for the provider-owned parameters backfill."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sciona.atoms.provider_inventory import discover_audit_manifest_path
from sciona.atoms.supabase_backfill import (
    backfill_parameters,
    build_parameter_rows,
    create_supabase_client_from_env,
)

log = logging.getLogger(__name__)


def create_supabase_client():
    """Create a service-role Supabase client lazily."""
    return create_supabase_client_from_env()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Log intended writes without mutating Supabase")
    parser.add_argument(
        "--audit-manifest",
        type=Path,
        default=discover_audit_manifest_path(),
        help="Path to audit_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    """Run the parameters backfill."""
    args = parse_args()
    stats = backfill_parameters(
        create_supabase_client(),
        audit_manifest_path=args.audit_manifest,
        dry_run=args.dry_run,
    )
    log.info("Parameters backfill complete: %s", stats)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
