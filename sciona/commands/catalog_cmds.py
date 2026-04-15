"""CLI commands for catalog management."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_MANIFEST_BUCKET = "sciona-platform"

from sciona.api.snapshot import (
    DEFAULT_MANIFEST_TIER,
    DEVELOPER_MANIFEST_TIER,
    MANIFEST_TIERS,
    manifest_artifact_key,
)


def _developer_mode_enabled() -> bool:
    return os.environ.get("SCIONA_DEVELOPER_MODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _resolve_manifest_url(args: argparse.Namespace) -> str:
    """Resolve the published manifest artifact URL."""
    explicit_url = getattr(args, "manifest_url", None)
    if explicit_url:
        return str(explicit_url).rstrip("/")

    env_url = os.environ.get("SCIONA_MANIFEST_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")

    tier = (
        str(getattr(args, "tier", "") or "").strip()
        or os.environ.get("SCIONA_MANIFEST_TIER", "").strip()
    )
    if not tier and _developer_mode_enabled():
        tier = DEVELOPER_MANIFEST_TIER
    valid_tiers = set(MANIFEST_TIERS)
    valid_tiers.add(DEVELOPER_MANIFEST_TIER)
    if tier not in valid_tiers:
        tier = DEFAULT_MANIFEST_TIER

    bucket = (
        os.environ.get("SCIONA_S3_BUCKET", "").strip()
        or os.environ.get("SCIONA_CATALOG_BUCKET", "").strip()
        or DEFAULT_MANIFEST_BUCKET
    )
    key = os.environ.get("SCIONA_MANIFEST_KEY", manifest_artifact_key(tier)).lstrip("/")
    return f"https://{bucket}.s3.amazonaws.com/{key}"


async def _download_manifest_bytes(manifest_url: str) -> bytes:
    """Fetch the published manifest bytes."""
    import httpx

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        response = await client.get(manifest_url)
        response.raise_for_status()
        return response.content


async def _cmd_catalog_sync(args: argparse.Namespace) -> None:
    """Download the published manifest.sqlite artifact."""
    output_path = Path(args.output) if args.output else Path.home() / ".sciona" / "manifest.sqlite"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_url = _resolve_manifest_url(args)

    try:
        payload = await _download_manifest_bytes(manifest_url)
    except ImportError:
        print("Error: httpx is required. Install with: pip install httpx", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error: failed to download manifest from {manifest_url}: {exc}", file=sys.stderr)
        sys.exit(1)

    output_path.write_bytes(payload)
    print(f"Manifest written to {output_path}")
