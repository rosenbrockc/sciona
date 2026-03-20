"""Discipline repo webhook synchronization."""

from __future__ import annotations

import hashlib
import hmac
import logging
import sqlite3
from pathlib import Path
from typing import Any

from sciona.ecosystem.models import DisciplineRepo

logger = logging.getLogger(__name__)


def validate_webhook_signature(
    payload: bytes,
    signature: str,
    secret: str,
) -> bool:
    """Validate a GitHub webhook HMAC-SHA256 signature.

    Parameters
    ----------
    payload
        Raw request body bytes.
    signature
        The ``X-Hub-Signature-256`` header value (``sha256=...``).
    secret
        The shared webhook secret.
    """
    if not signature.startswith("sha256="):
        return False

    expected = hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(f"sha256={expected}", signature)


def parse_manifest_sqlite(
    db_path: Path,
) -> list[dict[str, Any]]:
    """Parse atoms from a discipline repo's manifest.sqlite.

    Returns a list of atom dicts with keys: fqdn, content_hash, status,
    domain_tags, description.
    """
    if not db_path.exists():
        logger.warning("Manifest not found: %s", db_path)
        return []

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        tables = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "atoms" not in tables:
            return []

        rows = con.execute(
            "SELECT atom_id, fqdn, status, domain_tags, description FROM atoms"
        ).fetchall()
    finally:
        con.close()

    atoms = []
    for row in rows:
        tags = row["domain_tags"].split(",") if row["domain_tags"] else []
        atoms.append({
            "atom_id": row["atom_id"],
            "fqdn": row["fqdn"],
            "status": row["status"] or "approved",
            "domain_tags": tags,
            "description": row["description"] or "",
        })

    return atoms


def diff_atoms(
    local_atoms: list[dict[str, Any]],
    global_fqdns: frozenset[str],
) -> tuple[list[dict], list[dict]]:
    """Diff discipline repo atoms against the global registry.

    Returns (new_atoms, updated_atoms).
    """
    new_atoms = []
    updated_atoms = []

    for atom in local_atoms:
        if atom["fqdn"] not in global_fqdns:
            new_atoms.append(atom)
        else:
            updated_atoms.append(atom)

    return new_atoms, updated_atoms


def should_sync(repo: DisciplineRepo, pushed_commit: str) -> bool:
    """Check if a webhook delivery requires processing.

    Idempotency: skip if the pushed commit matches the last synced commit.
    """
    return pushed_commit != repo.last_synced_commit
