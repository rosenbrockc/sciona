"""CLI commands for catalog management."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


async def _cmd_catalog_sync(args: argparse.Namespace) -> None:
    """Download latest manifest.sqlite from the platform."""
    try:
        import httpx
    except ImportError:
        print("Error: httpx is required. Install with: pip install httpx", file=sys.stderr)
        sys.exit(1)

    from sciona.commands.login_cmds import _load_token

    token, default_api_url = _load_token()
    api_url = (args.api_url or default_api_url or "https://api.sciona.dev").rstrip("/")

    output_path = Path(args.output) if args.output else Path.home() / ".sciona" / "manifest.sqlite"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{api_url}/catalog/manifest",
            headers=headers,
            follow_redirects=True,
        )
        if resp.status_code == 404:
            print("No manifest available on the platform yet.", file=sys.stderr)
            sys.exit(1)
        if resp.status_code != 200:
            print(f"Error: {resp.status_code} {resp.text}", file=sys.stderr)
            sys.exit(1)

        output_path.write_bytes(resp.content)
        print(f"Manifest downloaded to {output_path}")
