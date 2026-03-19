"""Dashboard API endpoints — impact factor, benchmarks, BibTeX, compute preserved."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from ageom.api.deps import get_db
from ageom.ecosystem.dashboard import (
    compute_h_index,
    compute_impact_factor,
    estimate_compute_preserved,
    generate_bibtex,
)
from ageom.ecosystem.models import ComputePreserved, OriginatorImpact

router = APIRouter()


@router.get("/dashboard/originator/{originator_id}/impact")
async def get_originator_impact(
    originator_id: UUID,
    db=Depends(get_db),
) -> dict:
    """Algorithmic Impact Factor for an originator."""
    row = await db.fetchrow(
        """SELECT originator_id, github_login, bounty_count,
                  total_bounty_value, atom_count
           FROM originator_impact
           WHERE originator_id = $1""",
        originator_id,
    )
    if not row:
        raise HTTPException(404, "Originator not found")

    # Fetch individual bounty values for h-index
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
    db=Depends(get_db),
) -> list[dict]:
    """All benchmark results for an atom."""
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
    db=Depends(get_db),
) -> dict:
    """Auto-generated BibTeX entry for an atom."""
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
    db=Depends(get_db),
) -> dict:
    """Aggregate compute-preserved metrics."""
    rows = await db.fetch(
        """SELECT bounty_id, escrow_amount, cdg_node_count
           FROM compute_preserved"""
    )
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
    db=Depends(get_db),
) -> list[dict]:
    """Top originators by impact factor."""
    rows = await db.fetch(
        """SELECT originator_id, github_login, bounty_count,
                  total_bounty_value, atom_count
           FROM originator_impact
           ORDER BY total_bounty_value DESC
           LIMIT $1""",
        limit,
    )
    return [dict(r) for r in rows]
