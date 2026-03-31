"""Dashboard API endpoints — impact factor, benchmarks, BibTeX, compute preserved."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request

from sciona.api import deps as api_deps
from sciona.ecosystem.dashboard import (
    compute_impact_factor,
    estimate_compute_preserved,
    generate_bibtex,
)

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


@router.get("/dashboard/originator/{originator_id}/impact")
async def get_originator_impact(
    originator_id: UUID,
    supabase=Depends(_get_supabase),
    db=Depends(api_deps.get_db),
) -> dict:
    """Algorithmic Impact Factor for an originator."""
    if supabase is not None:
        impact_result = await supabase.rpc(
            "get_originator_impact", {"p_user_id": str(originator_id)}
        ).execute()
        row = _first_row(impact_result.data)
        if not row:
            raise HTTPException(404, "Originator not found")

        bounty_result = await supabase.rpc(
            "get_originator_bounty_values", {"p_user_id": str(originator_id)}
        ).execute()
        bounty_values = [
            float(r["escrow_amount"]) for r in (bounty_result.data or [])
        ]
        impact = compute_impact_factor(
            bounty_values,
            atom_count=int(row.get("atom_count", 0)),
            originator_id=str(originator_id),
            github_username=row.get("github_login", ""),
        )
        return impact.model_dump()

    row = await db.fetchrow(
        """SELECT originator_id, github_login, bounty_count,
                  total_bounty_value, atom_count
           FROM originator_impact
           WHERE originator_id = $1""",
        originator_id,
    )
    if not row:
        raise HTTPException(404, "Originator not found")

    bounty_rows = await db.fetch(
        """SELECT b.escrow_amount
           FROM atom_authors aa
           JOIN atoms a ON a.atom_id = aa.atom_id
           JOIN submissions s ON s.atom_versions ? a.fqdn AND s.is_winner = true
           JOIN bounties b ON b.bounty_id = s.bounty_id AND b.status = 'settled'
           WHERE aa.user_id = $1""",
        originator_id,
    )
    bounty_values = [float(r["escrow_amount"]) for r in bounty_rows]

    impact = compute_impact_factor(
        bounty_values,
        atom_count=row["atom_count"],
        originator_id=str(originator_id),
        github_username=row["github_login"],
    )

    return impact.model_dump()


@router.get("/dashboard/atom/{fqdn}/benchmarks")
async def get_atom_benchmarks(
    fqdn: str,
    supabase=Depends(_get_supabase),
    db=Depends(api_deps.get_db),
) -> list[dict]:
    """All benchmark results for an atom."""
    if supabase is not None:
        result = await supabase.rpc(
            "get_atom_benchmarks", {"p_fqdn": fqdn}
        ).execute()
        return list(result.data or [])

    rows = await db.fetch(
        """SELECT ab.benchmark_name, ab.metric_name, ab.metric_value,
                  ab.dataset_tag, ab.measured_at
           FROM atom_benchmarks ab
           JOIN atom_versions av ON av.version_id = ab.version_id
           JOIN atoms a ON a.atom_id = av.atom_id
           WHERE a.fqdn = $1
           ORDER BY ab.measured_at DESC""",
        fqdn,
    )
    return [dict(r) for r in rows]


@router.get("/dashboard/atom/{fqdn}/bibtex")
async def get_atom_bibtex(
    fqdn: str,
    supabase=Depends(_get_supabase),
    db=Depends(api_deps.get_db),
) -> dict:
    """Auto-generated BibTeX entry for an atom."""
    if supabase is not None:
        atom_result = await (
            supabase.table("atoms")
            .select("atom_id, fqdn, description")
            .eq("fqdn", fqdn)
            .maybe_single()
            .execute()
        )
        atom = _first_row(atom_result.data)
        if not atom:
            raise HTTPException(404, "Atom not found")

        author_ids_result = await (
            supabase.table("atom_authors")
            .select("user_id")
            .eq("atom_id", atom["atom_id"])
            .execute()
        )
        author_ids = [str(r["user_id"]) for r in (author_ids_result.data or [])]
        authors = []
        if author_ids:
            author_rows = await (
                supabase.table("users")
                .select("github_login")
                .in_("user_id", author_ids)
                .execute()
            )
            authors = [r["github_login"] for r in (author_rows.data or [])]

        bibtex = generate_bibtex(
            atom_fqdn=fqdn,
            authors=authors,
            description=atom["description"],
        )
        return {"fqdn": fqdn, "bibtex": bibtex}

    atom = await db.fetchrow(
        "SELECT atom_id, fqdn, description FROM atoms WHERE fqdn = $1", fqdn
    )
    if not atom:
        raise HTTPException(404, "Atom not found")

    authors_rows = await db.fetch(
        """SELECT u.github_login
           FROM atom_authors aa
           JOIN users u ON u.user_id = aa.user_id
           WHERE aa.atom_id = $1""",
        atom["atom_id"],
    )
    authors = [r["github_login"] for r in authors_rows]

    bibtex = generate_bibtex(
        atom_fqdn=fqdn,
        authors=authors,
        description=atom["description"],
    )

    return {"fqdn": fqdn, "bibtex": bibtex}


@router.get("/dashboard/compute-preserved")
async def get_compute_preserved(
    supabase=Depends(_get_supabase),
    db=Depends(api_deps.get_db),
) -> dict:
    """Aggregate compute-preserved metrics."""
    if supabase is not None:
        result = await (
            supabase.table("compute_preserved")
            .select("bounty_id, escrow_amount, cdg_node_count")
            .execute()
        )
        rows = result.data or []
    else:
        rows = await db.fetch(
            """SELECT bounty_id, escrow_amount, cdg_node_count
               FROM compute_preserved"""
        )
        rows = [dict(r) for r in rows]

    bounties = [
        {
            "escrow_amount": float(r["escrow_amount"]),
            "cdg_node_count": r["cdg_node_count"],
        }
        for r in rows
    ]

    result = estimate_compute_preserved(bounties)
    return result.model_dump()


@router.get("/dashboard/leaderboard")
async def get_leaderboard(
    limit: int = 50,
    supabase=Depends(_get_supabase),
    db=Depends(api_deps.get_db),
) -> list[dict]:
    """Top originators by impact factor."""
    if supabase is not None:
        result = await (
            supabase.table("originator_impact")
            .select(
                "originator_id, github_login, bounty_count, total_bounty_value, atom_count"
            )
            .order("total_bounty_value", desc=True)
            .limit(limit)
            .execute()
        )
        return list(result.data or [])

    rows = await db.fetch(
        """SELECT originator_id, github_login, bounty_count,
                  total_bounty_value, atom_count
           FROM originator_impact
           ORDER BY total_bounty_value DESC
           LIMIT $1""",
        limit,
    )
    return [dict(r) for r in rows]
