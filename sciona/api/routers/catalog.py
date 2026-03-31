"""Catalog search and atom-document endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from sciona.api.deps import get_db, use_supabase_db
from sciona.api.models import CatalogEntry

router = APIRouter()


@router.get("/search")
async def catalog_search(
    request: Request,
    q: str,
    domain_tag: str | None = None,
    limit: int = Query(default=50, le=200),
    db=Depends(get_db),
) -> list[CatalogEntry]:
    """Full-text search across the atom catalog."""
    if use_supabase_db() and getattr(request.app.state, "supabase", None) is not None:
        supabase = request.app.state.supabase
        if q:
            try:
                rpc_result = await supabase.rpc(
                    "search_atoms_hybrid",
                    {
                        "query_text": q,
                        "mode": "fts",
                        "result_limit": limit,
                        "result_offset": 0,
                    },
                ).execute()
                rows = rpc_result.data or []
                if domain_tag:
                    rows = [
                        row
                        for row in rows
                        if domain_tag in (row.get("domain_tags") or [])
                    ]
                return [
                    CatalogEntry(
                        fqdn=row["fqdn"],
                        description=row.get("technical_description", "") or "",
                        domain_tags=row.get("domain_tags", []) or [],
                        status="approved",
                    )
                    for row in rows[:limit]
                ]
            except Exception:
                pass
        query = supabase.table("catalog_atoms_served").select(
            "fqdn, technical_description, domain_tags"
        )
        if q:
            query = query.or_(
                f"fqdn.ilike.%{q}%,technical_description.ilike.%{q}%"
            )
        if domain_tag:
            query = query.contains("domain_tags", [domain_tag])
        result = await query.limit(limit).execute()
        return [
            CatalogEntry(
                fqdn=row["fqdn"],
                description=row.get("technical_description", "") or "",
                domain_tags=row.get("domain_tags", []) or [],
                status="approved",
            )
            for row in (result.data or [])
        ]

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


@router.get("/atom/{fqdn:path}")
async def get_atom_document(
    request: Request,
    fqdn: str,
    db=Depends(get_db),
) -> dict:
    """Fetch the full atom documentation bundle via the database RPC."""
    if use_supabase_db() and getattr(request.app.state, "supabase", None) is not None:
        supabase = request.app.state.supabase
        result = await supabase.rpc(
            "get_atom_document",
            {"request_fqdn": fqdn},
        ).execute()
        document = result.data
    else:
        document = await db.fetchval("SELECT public.get_atom_document($1)", fqdn)
    if not document:
        raise HTTPException(404, f"Atom {fqdn!r} not found")
    return document
