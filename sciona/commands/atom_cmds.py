"""CLI commands for atom publishing."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import sys
import tarfile
from pathlib import Path


async def _cmd_atom_publish(args: argparse.Namespace) -> None:
    """Publish an atom to the global registry."""
    try:
        import httpx
    except ImportError:
        print("Error: httpx is required. Install with: pip install httpx", file=sys.stderr)
        sys.exit(1)

    from sciona.commands.login_cmds import _load_token

    token, default_api_url = _load_token()
    if not token:
        print("Error: not authenticated. Run `sciona login` first.", file=sys.stderr)
        sys.exit(1)

    api_url = (getattr(args, "api_url", None) or default_api_url or "https://api.sciona.dev").rstrip("/")
    source_path = Path(args.path)

    if not source_path.exists():
        print(f"Error: path not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    # Create tar.gz of the source directory
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(source_path), arcname=source_path.name)
    source_bytes = buf.getvalue()
    source_b64 = base64.b64encode(source_bytes).decode()

    # Compute fingerprint from Python files
    fingerprint = ""
    py_files = sorted(source_path.rglob("*.py")) if source_path.is_dir() else [source_path]
    if py_files:
        try:
            from sciona.architect.atom_similarity import fingerprint_function

            combined = "\n".join(f.read_text() for f in py_files if f.exists())
            fingerprint = fingerprint_function(combined)
        except Exception:
            h = hashlib.sha256()
            for f in py_files:
                h.update(f.read_bytes())
            fingerprint = h.hexdigest()

    body = {
        "fqdn": source_path.stem if source_path.is_dir() else source_path.stem,
        "semver": args.semver,
        "source_tar_b64": source_b64,
        "fingerprint": fingerprint,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{api_url}/atoms",
            json=body,
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp.status_code == 409:
            print(f"Atom already exists with this content hash.", file=sys.stderr)
            sys.exit(1)
        if resp.status_code != 200:
            print(f"Error: {resp.status_code} {resp.text}", file=sys.stderr)
            sys.exit(1)

        data = resp.json()
        print(f"Published {data['fqdn']} v{data['semver']}")
        print(f"  atom_id: {data['atom_id']}")
        print(f"  version_id: {data['version_id']}")
        print(f"  content_hash: {data['content_hash']}")
        if data.get("is_new_atom"):
            print("  (new atom created)")
