"""CLI commands for catalog management."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sciona.api.snapshot import fetch_manifest_data, generate_manifest_sqlite


async def _cmd_catalog_sync(args: argparse.Namespace) -> None:
    """Build the local manifest.sqlite from Supabase data."""

    from sciona.commands.login_cmds import _load_token

    token, default_api_url = _load_token()
    supabase_url = (
        os.environ.get("SUPABASE_URL")
        or os.environ.get("SCIONA_SUPABASE_URL")
        or getattr(args, "api_url", None)
        or default_api_url
        or ""
    ).rstrip("/")
    if not supabase_url:
        print(
            "Error: SUPABASE_URL is not configured.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_path = Path(args.output) if args.output else Path.home() / ".sciona" / "manifest.sqlite"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    access_token = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
        or token
    )
    if not access_token:
        print(
            "Error: no Supabase token configured. Set SUPABASE_SERVICE_ROLE_KEY, "
            "SUPABASE_ANON_KEY, or run `sciona login`.",
            file=sys.stderr,
        )
        sys.exit(1)

    manifest_data = await fetch_manifest_data(
        supabase_url,
        access_token,
    )
    con = generate_manifest_sqlite(manifest_data, output_path=output_path)
    con.close()
    print(f"Manifest written to {output_path}")
