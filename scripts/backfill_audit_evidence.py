"""Compatibility wrapper for the provider-owned audit evidence backfill."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from sciona.atoms.provider_inventory import discover_audit_manifest_path
from sciona.atoms.supabase_backfill import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_RUNNER_VERSION,
    backfill_audit_evidence,
    build_evidence_rows,
    create_supabase_client_from_env,
    fetch_atom_lookup,
    load_manifest_entries,
)

log = logging.getLogger(__name__)

load_manifest = load_manifest_entries


def fetch_existing_backfill_keys(
    supabase,
    *,
    runner_version: str,
) -> set[tuple[str, str]]:
    """Fetch existing backfill evidence keys so reruns remain pragmatic."""
    response = (
        supabase.table("atom_audit_evidence")
        .select("atom_id, audit_type")
        .eq("runner_version", runner_version)
        .execute()
    )
    return {
        (row["atom_id"], row["audit_type"])
        for row in (response.data or [])
        if row.get("atom_id") and row.get("audit_type")
    }


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
        default=int(os.environ.get("AUDIT_BACKFILL_BATCH_SIZE", DEFAULT_BATCH_SIZE)),
        help="Insert batch size",
    )
    parser.add_argument(
        "--runner-version",
        default=os.environ.get("AUDIT_BACKFILL_RUNNER_VERSION", DEFAULT_RUNNER_VERSION),
        help="Synthetic runner_version tag used for idempotent reruns",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build rows without inserting them")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    stats = backfill_audit_evidence(
        create_supabase_client_from_env(),
        manifest_path=args.manifest_path,
        batch_size=args.batch_size,
        runner_version=args.runner_version,
        dry_run=args.dry_run,
    )
    log.info("Audit evidence backfill complete: %s", stats)


if __name__ == "__main__":
    main()
