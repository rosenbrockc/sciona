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


async def _fetch_artifact_benchmarks(
    fqdn: str,
    *,
    supabase,
) -> list[dict]:
    artifact_result = await (
        supabase.table("artifacts")
        .select("artifact_id")
        .eq("fqdn", fqdn)
        .limit(1)
        .execute()
    )
    artifact_rows = artifact_result.data or []
    if not artifact_rows:
        return []
    artifact_id = artifact_rows[0].get("artifact_id")
    if not artifact_id:
        return []

    version_result = await (
        supabase.table("artifact_versions")
        .select("version_id,content_hash")
        .eq("artifact_id", artifact_id)
        .execute()
    )
    version_rows = version_result.data or []
    if not version_rows:
        return []
    content_hash_by_version = {
        str(row["version_id"]): str(row.get("content_hash", ""))
        for row in version_rows
        if row.get("version_id")
    }
    version_ids = sorted(content_hash_by_version)
    if not version_ids:
        return []

    benchmark_result = await (
        supabase.table("artifact_benchmarks")
        .select("version_id,benchmark_name,metric_name,metric_value,dataset_tag,measured_at")
        .in_("version_id", version_ids)
        .execute()
    )
    benchmark_rows = benchmark_result.data or []
    rows: list[dict] = []
    for row in benchmark_rows:
        version_id = str(row.get("version_id", ""))
        rows.append(
            {
                "artifact_fqdn": fqdn,
                "content_hash": content_hash_by_version.get(version_id, ""),
                "benchmark_id": row.get("benchmark_name", "") or "",
                "benchmark_name": row.get("benchmark_name", "") or "",
                "metric_name": row.get("metric_name", "") or "",
                "metric_value": row.get("metric_value"),
                "dataset_tag": row.get("dataset_tag", "") or "",
                "measured_at": row.get("measured_at", "") or "",
            }
        )
    rows.sort(
        key=lambda row: (
            str(row.get("benchmark_name", "")),
            str(row.get("metric_name", "")),
            str(row.get("content_hash", "")),
            str(row.get("measured_at", "")),
        )
    )
    return rows


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
    if not document.get("benchmarks"):
        try:
            document["benchmarks"] = await _fetch_artifact_benchmarks(
                fqdn,
                supabase=supabase,
            )
        except Exception:
            pass
    return document
