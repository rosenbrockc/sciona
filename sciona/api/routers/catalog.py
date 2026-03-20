"""Catalog search and manifest download endpoints."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from sciona.api.deps import get_db
from sciona.api.models import CatalogEntry

router = APIRouter()


@router.get("/search")
async def catalog_search(
    q: str,
    domain_tag: str | None = None,
    limit: int = Query(default=50, le=200),
    db=Depends(get_db),
) -> list[CatalogEntry]:
    """Full-text search across the atom catalog."""
    conditions = ["a.status = 'approved'"]
    params: list = []
    idx = 1

    if q:
        conditions.append(
            f"(a.fqdn ILIKE ${idx} OR a.description ILIKE ${idx})"
        )
        params.append(f"%{q}%")
        idx += 1

    if domain_tag:
        conditions.append(f"${idx} = ANY(a.domain_tags)")
        params.append(domain_tag)
        idx += 1

    where = " AND ".join(conditions)
    params.append(limit)

    rows = await db.fetch(
        f"""SELECT a.fqdn, a.description, a.domain_tags, a.status
            FROM atoms a
            WHERE {where}
            ORDER BY a.updated_at DESC
            LIMIT ${idx}""",
        *params,
    )

    return [
        CatalogEntry(
            fqdn=r["fqdn"],
            description=r["description"],
            domain_tags=r["domain_tags"],
            status=r["status"],
        )
        for r in rows
    ]


@router.get("/manifest")
async def download_manifest() -> FileResponse:
    """Download latest manifest.sqlite snapshot."""
    # In production, this would proxy from S3.
    # For local dev, serve from a configured path.
    manifest_path = os.environ.get(
        "SCIONA_MANIFEST_PATH", "data/manifest.sqlite"
    )
    path = Path(manifest_path)
    if not path.exists():
        raise HTTPException(404, "Manifest not available")

    return FileResponse(
        path=str(path),
        media_type="application/x-sqlite3",
        filename="manifest.sqlite",
    )
