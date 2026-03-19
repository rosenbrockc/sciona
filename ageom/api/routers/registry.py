"""Atom registry CRUD endpoints."""

from __future__ import annotations

import hashlib
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from ageom.api.deps import UserRow, get_db, require_auth
from ageom.api.models import (
    AtomDetailResponse,
    AtomPublishRequest,
    AtomPublishResponse,
    AtomSummaryResponse,
    AtomVersionResponse,
    PaginatedResponse,
)

router = APIRouter()


@router.post("")
async def publish_atom(
    body: AtomPublishRequest,
    user: UserRow = Depends(require_auth),
    db=Depends(get_db),
) -> AtomPublishResponse:
    """Publish a new atom or new version of an existing atom."""
    import base64

    source_bytes = base64.b64decode(body.source_tar_b64)
    content_hash = hashlib.sha256(source_bytes).hexdigest()

    # Check for duplicate content hash
    existing = await db.fetchrow(
        "SELECT version_id FROM atom_versions WHERE content_hash = $1",
        content_hash,
    )
    if existing:
        raise HTTPException(409, f"Content hash {content_hash[:16]}… already exists")

    # Upsert atom
    atom_row = await db.fetchrow(
        "SELECT atom_id FROM atoms WHERE fqdn = $1", body.fqdn
    )
    is_new = atom_row is None

    if is_new:
        atom_row = await db.fetchrow(
            """INSERT INTO atoms (fqdn, owner_id, domain_tags, description)
               VALUES ($1, $2::uuid, $3, $4)
               RETURNING atom_id""",
            body.fqdn,
            user.user_id,
            body.domain_tags,
            body.description,
        )
    atom_id = atom_row["atom_id"]

    # Mark previous latest as non-latest
    await db.execute(
        "UPDATE atom_versions SET is_latest = FALSE WHERE atom_id = $1",
        atom_id,
    )

    # TODO: Upload source_bytes to S3 at atoms/{content_hash}.tar.gz
    s3_key = f"atoms/{content_hash}.tar.gz"

    version_row = await db.fetchrow(
        """INSERT INTO atom_versions
           (atom_id, content_hash, semver, is_latest, s3_key, fingerprint)
           VALUES ($1, $2, $3, TRUE, $4, $5)
           RETURNING version_id""",
        atom_id,
        content_hash,
        body.semver,
        s3_key,
        body.fingerprint,
    )

    return AtomPublishResponse(
        atom_id=atom_id,
        version_id=version_row["version_id"],
        fqdn=body.fqdn,
        content_hash=content_hash,
        semver=body.semver,
        is_new_atom=is_new,
    )


@router.get("/{fqdn:path}/versions")
async def list_versions(fqdn: str, db=Depends(get_db)) -> list[AtomVersionResponse]:
    """List all versions of an atom."""
    rows = await db.fetch(
        """SELECT v.version_id, v.content_hash, v.semver, v.is_latest,
                  v.fingerprint, v.created_at
           FROM atom_versions v
           JOIN atoms a ON v.atom_id = a.atom_id
           WHERE a.fqdn = $1
           ORDER BY v.created_at DESC""",
        fqdn,
    )
    return [AtomVersionResponse(**dict(r)) for r in rows]


@router.get("")
async def search_atoms(
    q: str = "",
    domain_tag: str | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    db=Depends(get_db),
) -> PaginatedResponse:
    """Search/list atoms with optional filters."""
    conditions = ["a.status = 'approved'"]
    params: list = []
    idx = 1

    if q:
        conditions.append(f"(a.fqdn ILIKE ${idx} OR a.description ILIKE ${idx})")
        params.append(f"%{q}%")
        idx += 1

    if domain_tag:
        conditions.append(f"${idx} = ANY(a.domain_tags)")
        params.append(domain_tag)
        idx += 1

    where = " AND ".join(conditions)

    count_row = await db.fetchrow(
        f"SELECT COUNT(*) AS cnt FROM atoms a WHERE {where}", *params
    )
    total = count_row["cnt"] if count_row else 0

    params.extend([limit, offset])
    rows = await db.fetch(
        f"""SELECT a.atom_id, a.fqdn, a.description, a.domain_tags, a.status
            FROM atoms a
            WHERE {where}
            ORDER BY a.updated_at DESC
            LIMIT ${idx} OFFSET ${idx + 1}""",
        *params,
    )

    items = [
        AtomSummaryResponse(
            atom_id=r["atom_id"],
            fqdn=r["fqdn"],
            description=r["description"],
            domain_tags=r["domain_tags"],
            status=r["status"],
        )
        for r in rows
    ]

    return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)


@router.get("/{fqdn:path}")
async def get_atom(fqdn: str, db=Depends(get_db)) -> AtomDetailResponse:
    """Get atom metadata + latest version."""
    row = await db.fetchrow(
        """SELECT a.atom_id, a.fqdn, a.description, a.domain_tags, a.status,
                  a.created_at, u.github_login AS owner_github_login
           FROM atoms a
           JOIN users u ON a.owner_id = u.user_id
           WHERE a.fqdn = $1""",
        fqdn,
    )
    if not row:
        raise HTTPException(404, f"Atom {fqdn!r} not found")

    latest = await db.fetchrow(
        """SELECT version_id, content_hash, semver, is_latest, fingerprint, created_at
           FROM atom_versions
           WHERE atom_id = $1 AND is_latest = TRUE""",
        row["atom_id"],
    )

    return AtomDetailResponse(
        atom_id=row["atom_id"],
        fqdn=row["fqdn"],
        description=row["description"],
        domain_tags=row["domain_tags"],
        status=row["status"],
        owner_github_login=row["owner_github_login"],
        latest_version=AtomVersionResponse(**dict(latest)) if latest else None,
        created_at=row["created_at"],
    )
