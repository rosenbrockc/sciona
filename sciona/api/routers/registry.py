"""Atom registry CRUD endpoints."""

from __future__ import annotations

import hashlib
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from sciona.api import deps as api_deps
from sciona.api.models import (
    AtomDetailResponse,
    AtomPublishRequest,
    AtomPublishResponse,
    AtomSummaryResponse,
    AtomVersionResponse,
    PaginatedResponse,
)

UserRow = getattr(api_deps, "UserProfile", None) or api_deps.UserRow
get_db = api_deps.get_db
require_auth = api_deps.require_auth

router = APIRouter()


async def _get_supabase(request: Request) -> Any | None:
    if not api_deps.use_supabase_db():
        return None
    return getattr(request.app.state, "supabase", None)


def _first_row(data: Any) -> dict[str, Any] | None:
    if data is None:
        return None
    if isinstance(data, list):
        return data[0] if data else None
    if isinstance(data, dict):
        return data
    return None


@router.post("")
async def publish_atom(
    body: AtomPublishRequest,
    user: UserRow = Depends(require_auth),
    supabase=Depends(_get_supabase),
    db=Depends(get_db),
) -> AtomPublishResponse:
    """Publish a new atom or new version of an existing atom."""
    import base64

    source_bytes = base64.b64decode(body.source_tar_b64)
    content_hash = hashlib.sha256(source_bytes).hexdigest()
    user_id = str(user.user_id)

    if supabase is not None:
        existing = await (
            supabase.table("atom_versions")
            .select("version_id")
            .eq("content_hash", content_hash)
            .maybe_single()
            .execute()
        )
        if _first_row(existing.data):
            raise HTTPException(409, f"Content hash {content_hash[:16]}… already exists")

        atom_result = await (
            supabase.table("atoms")
            .select("atom_id")
            .eq("fqdn", body.fqdn)
            .maybe_single()
            .execute()
        )
        atom_row = _first_row(atom_result.data)
        is_new = atom_row is None
        if is_new:
            inserted = await (
                supabase.table("atoms")
                .insert(
                    {
                        "fqdn": body.fqdn,
                        "owner_id": user_id,
                        "domain_tags": body.domain_tags,
                        "description": body.description,
                    }
                )
                .execute()
            )
            atom_row = _first_row(inserted.data)
        if not atom_row:
            raise HTTPException(500, "Failed to create atom")

        atom_id = atom_row["atom_id"]
        await (
            supabase.table("atom_versions")
            .update({"is_latest": False})
            .eq("atom_id", atom_id)
            .execute()
        )
        version_row = _first_row(
            (
                await (
                    supabase.table("atom_versions")
                    .insert(
                        {
                            "atom_id": atom_id,
                            "content_hash": content_hash,
                            "semver": body.semver,
                            "is_latest": True,
                            "s3_key": f"atoms/{content_hash}.tar.gz",
                            "fingerprint": body.fingerprint,
                        }
                    )
                    .execute()
                )
            ).data
        )
        if not version_row:
            raise HTTPException(500, "Failed to create version")

        return AtomPublishResponse(
            atom_id=atom_id,
            version_id=version_row["version_id"],
            fqdn=body.fqdn,
            content_hash=content_hash,
            semver=body.semver,
            is_new_atom=is_new,
        )

    existing = await db.fetchrow(
        "SELECT version_id FROM atom_versions WHERE content_hash = $1",
        content_hash,
    )
    if existing:
        raise HTTPException(409, f"Content hash {content_hash[:16]}… already exists")

    atom_row = await db.fetchrow("SELECT atom_id FROM atoms WHERE fqdn = $1", body.fqdn)
    is_new = atom_row is None

    if is_new:
        atom_row = await db.fetchrow(
            """INSERT INTO atoms (fqdn, owner_id, domain_tags, description)
               VALUES ($1, $2::uuid, $3, $4)
               RETURNING atom_id""",
            body.fqdn,
            user_id,
            body.domain_tags,
            body.description,
        )
    atom_id = atom_row["atom_id"]

    await db.execute("UPDATE atom_versions SET is_latest = FALSE WHERE atom_id = $1", atom_id)

    version_row = await db.fetchrow(
        """INSERT INTO atom_versions
           (atom_id, content_hash, semver, is_latest, s3_key, fingerprint)
           VALUES ($1, $2, $3, TRUE, $4, $5)
           RETURNING version_id""",
        atom_id,
        content_hash,
        body.semver,
        f"atoms/{content_hash}.tar.gz",
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
async def list_versions(
    fqdn: str,
    supabase=Depends(_get_supabase),
    db=Depends(get_db),
) -> list[AtomVersionResponse]:
    """List all versions of an atom."""
    if supabase is not None:
        atom_result = await (
            supabase.table("atoms")
            .select("atom_id")
            .eq("fqdn", fqdn)
            .maybe_single()
            .execute()
        )
        atom_row = _first_row(atom_result.data)
        if not atom_row:
            return []

        rows = await (
            supabase.table("atom_versions")
            .select("version_id, content_hash, semver, is_latest, fingerprint, created_at")
            .eq("atom_id", atom_row["atom_id"])
            .order("created_at", desc=True)
            .execute()
        )
        return [AtomVersionResponse(**dict(r)) for r in (rows.data or [])]

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
    supabase=Depends(_get_supabase),
    db=Depends(get_db),
) -> PaginatedResponse:
    """Search/list atoms with optional filters."""
    limit = int(getattr(limit, "default", limit))
    offset = int(getattr(offset, "default", offset))

    if supabase is not None:
        query = (
            supabase.table("atoms")
            .select("atom_id, fqdn, description, domain_tags, status", count="exact")
            .eq("status", "approved")
        )
        if q:
            query = query.or_(f"fqdn.ilike.%{q}%,description.ilike.%{q}%")
        if domain_tag:
            query = query.contains("domain_tags", [domain_tag])
        result = await query.order("updated_at", desc=True).range(
            offset, offset + limit - 1
        ).execute()
        rows = result.data or []
        total = int(result.count or len(rows))
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

    conditions = ["a.status = 'approved'"]
    params: list[Any] = []
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
async def get_atom(
    fqdn: str,
    supabase=Depends(_get_supabase),
    db=Depends(get_db),
) -> AtomDetailResponse:
    """Get atom metadata + latest version."""
    if supabase is not None:
        atom_result = await (
            supabase.table("atoms")
            .select(
                "atom_id, fqdn, description, domain_tags, status, created_at, owner_id"
            )
            .eq("fqdn", fqdn)
            .maybe_single()
            .execute()
        )
        atom = _first_row(atom_result.data)
        if not atom:
            raise HTTPException(404, f"Atom {fqdn!r} not found")

        owner_result = await (
            supabase.table("users")
            .select("github_login")
            .eq("user_id", atom["owner_id"])
            .maybe_single()
            .execute()
        )
        owner = _first_row(owner_result.data)
        if not owner:
            raise HTTPException(404, f"Atom {fqdn!r} not found")

        latest_result = await (
            supabase.table("atom_versions")
            .select("version_id, content_hash, semver, is_latest, fingerprint, created_at")
            .eq("atom_id", atom["atom_id"])
            .eq("is_latest", True)
            .maybe_single()
            .execute()
        )
        latest = _first_row(latest_result.data)

        return AtomDetailResponse(
            atom_id=atom["atom_id"],
            fqdn=atom["fqdn"],
            description=atom["description"],
            domain_tags=atom["domain_tags"],
            status=atom["status"],
            owner_github_login=owner["github_login"],
            latest_version=AtomVersionResponse(**dict(latest)) if latest else None,
            created_at=atom["created_at"],
        )

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
