"""Catalog search and atom-document endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from sciona.api import deps as api_deps
from sciona.api.models import CatalogEntry

router = APIRouter()


def _catalog_entry_from_row(row: dict, *, default_kind: str) -> CatalogEntry:
    return CatalogEntry(
        fqdn=row["fqdn"],
        description=row.get("technical_description", "") or "",
        artifact_kind=row.get("artifact_kind", default_kind) or default_kind,
        domain_tags=row.get("domain_tags", []) or [],
        status="approved",
        overall_verdict=row.get("overall_verdict", "") or "",
        risk_tier=row.get("risk_tier", "") or "",
        trust_readiness=row.get("trust_readiness", "") or "",
    )


@router.get("/search")
async def catalog_search(
    q: str,
    domain_tag: str | None = None,
    limit: int = Query(default=50, le=200),
    supabase=Depends(api_deps.get_supabase),
) -> list[CatalogEntry]:
    """Full-text search across the atom catalog."""
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
                _catalog_entry_from_row(row, default_kind="atom")
                for row in rows[:limit]
            ]
        except Exception:
            pass
    query = supabase.table("catalog_atoms_served").select(
        "fqdn, technical_description, domain_tags, overall_verdict, risk_tier, trust_readiness"
    )
    if q:
        query = query.or_(
            f"fqdn.ilike.%{q}%,technical_description.ilike.%{q}%"
        )
    if domain_tag:
        query = query.contains("domain_tags", [domain_tag])
    result = await query.limit(limit).execute()
    return [
        _catalog_entry_from_row(row, default_kind="atom")
        for row in (result.data or [])
    ]


@router.get("/atom/{fqdn:path}")
async def get_atom_document(
    fqdn: str,
    supabase=Depends(api_deps.get_supabase),
) -> dict:
    """Fetch the full atom documentation bundle via the database RPC."""
    result = await supabase.rpc(
        "get_atom_document",
        {"request_fqdn": fqdn},
    ).execute()
    document = result.data
    if not document:
        raise HTTPException(404, f"Atom {fqdn!r} not found")
    return document


@router.get("/search-artifacts")
async def artifact_search(
    q: str,
    domain_tag: str | None = None,
    limit: int = Query(default=50, le=200),
    supabase=Depends(api_deps.get_supabase),
) -> list[CatalogEntry]:
    """Search across artifact kinds, falling back to the atom catalog when needed."""
    if q:
        try:
            rpc_result = await supabase.rpc(
                "search_artifacts_hybrid",
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
            return [_catalog_entry_from_row(row, default_kind="artifact") for row in rows[:limit]]
        except Exception:
            pass
    try:
        query = supabase.table("catalog_artifacts_served").select(
            "fqdn, artifact_kind, technical_description, domain_tags, overall_verdict, risk_tier, trust_readiness"
        )
        if q:
            query = query.or_(f"fqdn.ilike.%{q}%,technical_description.ilike.%{q}%")
        if domain_tag:
            query = query.contains("domain_tags", [domain_tag])
        result = await query.limit(limit).execute()
        return [
            _catalog_entry_from_row(row, default_kind="artifact")
            for row in (result.data or [])
        ]
    except Exception:
        return await catalog_search(q=q, domain_tag=domain_tag, limit=limit, supabase=supabase)


@router.get("/artifact/{fqdn:path}")
async def get_artifact_document(
    fqdn: str,
    supabase=Depends(api_deps.get_supabase),
) -> dict:
    """Fetch the full artifact documentation bundle via the database RPC."""
    try:
        result = await supabase.rpc(
            "get_artifact_document",
            {"request_fqdn": fqdn},
        ).execute()
        document = result.data
    except Exception:
        document = None
    if not document:
        return await get_atom_document(fqdn, supabase=supabase)
    return document
