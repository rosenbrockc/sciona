"""Verification and settlement API endpoints."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from sciona.api import deps as api_deps
from sciona.api.models import PaginatedResponse
from sciona.clearinghouse.models import LeaderboardEntry

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


@router.get("/submissions/{submission_id}/status")
async def get_submission_status(
    submission_id: UUID,
    supabase=Depends(_get_supabase),
    db=Depends(api_deps.get_db),
) -> dict:
    """Poll verification progress for a submission."""
    if supabase is not None:
        submission_result = await (
            supabase.table("submissions")
            .select("*")
            .eq("submission_id", str(submission_id))
            .maybe_single()
            .execute()
        )
        submission = _first_row(submission_result.data)
        if not submission:
            raise HTTPException(404, "Submission not found")

        runs_result = await (
            supabase.table("verification_runs")
            .select("status, metric_values, output_hash, is_deterministic")
            .eq("submission_id", str(submission_id))
            .order("created_at", desc=True)
            .execute()
        )
        return {
            "submission_id": str(submission_id),
            "verification_status": submission["verification_status"],
            "runs": runs_result.data or [],
        }

    row = await db.fetchrow(
        "SELECT * FROM submissions WHERE submission_id = $1", submission_id
    )
    if not row:
        raise HTTPException(404, "Submission not found")

    runs = await db.fetch(
        """SELECT status, metric_values, output_hash, is_deterministic
           FROM verification_runs
           WHERE submission_id = $1
           ORDER BY created_at DESC""",
        submission_id,
    )

    return {
        "submission_id": str(submission_id),
        "verification_status": row["verification_status"],
        "runs": [dict(r) for r in runs],
    }


@router.get("/bounties/{bounty_id}/leaderboard")
async def get_leaderboard(
    bounty_id: UUID,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    supabase=Depends(_get_supabase),
    db=Depends(api_deps.get_db),
) -> PaginatedResponse:
    """Current ranking of verified submissions for a bounty."""
    limit = int(getattr(limit, "default", limit))
    offset = int(getattr(offset, "default", offset))

    if supabase is not None:
        bounty_result = await (
            supabase.table("bounties")
            .select("bounty_id")
            .eq("bounty_id", str(bounty_id))
            .maybe_single()
            .execute()
        )
        if not _first_row(bounty_result.data):
            raise HTTPException(404, "Bounty not found")

        rpc_result = await supabase.rpc(
            "get_bounty_leaderboard",
            {
                "p_bounty_id": str(bounty_id),
                "p_limit": limit,
                "p_offset": offset,
            },
        ).execute()
        rows = rpc_result.data or []
        total = int(rows[0]["total_count"]) if rows else 0
        items = [
            LeaderboardEntry(
                submission_id=str(r["submission_id"]),
                architect_id=str(r["architect_id"]),
                metric_values=r["metric_values"] or {},
                verified_at=r["verified_at"],
                rank=offset + i + 1,
            )
            for i, r in enumerate(rows)
        ]
        return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)

    row = await db.fetchrow(
        "SELECT * FROM bounties WHERE bounty_id = $1", bounty_id
    )
    if not row:
        raise HTTPException(404, "Bounty not found")

    count_row = await db.fetchrow(
        """SELECT COUNT(*) AS cnt FROM verification_runs
           WHERE bounty_id = $1 AND status = 'completed'""",
        bounty_id,
    )
    total = count_row["cnt"] if count_row else 0

    runs = await db.fetch(
        """SELECT vr.submission_id, s.architect_id, vr.metric_values, vr.completed_at
           FROM verification_runs vr
           JOIN submissions s ON s.submission_id = vr.submission_id
           WHERE vr.bounty_id = $1 AND vr.status = 'completed' AND vr.split_type = 'blind'
           ORDER BY vr.completed_at DESC
           LIMIT $2 OFFSET $3""",
        bounty_id,
        limit,
        offset,
    )

    items = [
        LeaderboardEntry(
            submission_id=str(r["submission_id"]),
            architect_id=str(r["architect_id"]),
            metric_values=r["metric_values"] or {},
            verified_at=r["completed_at"],
            rank=offset + i + 1,
        )
        for i, r in enumerate(runs)
    ]

    return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)


@router.get("/bounties/{bounty_id}/settlement")
async def get_settlement(
    bounty_id: UUID,
    supabase=Depends(_get_supabase),
    db=Depends(api_deps.get_db),
) -> dict:
    """Retrieve settlement details for a settled bounty."""
    if supabase is not None:
        bounty_result = await (
            supabase.table("bounties")
            .select("*")
            .eq("bounty_id", str(bounty_id))
            .maybe_single()
            .execute()
        )
        row = _first_row(bounty_result.data)
        if not row:
            raise HTTPException(404, "Bounty not found")
        if row["status"] != "settled":
            raise HTTPException(409, "Bounty is not yet settled")

        payouts_result = await (
            supabase.table("settlement_payouts")
            .select("recipient_id, role, amount, atom_fqdn, cdg_hash")
            .eq("bounty_id", str(bounty_id))
            .order("role")
            .order("amount", desc=True)
            .execute()
        )
        return {
            "bounty_id": str(bounty_id),
            "status": "settled",
            "escrow_amount": float(row["escrow_amount"]),
            "payouts": payouts_result.data or [],
        }

    row = await db.fetchrow(
        "SELECT * FROM bounties WHERE bounty_id = $1", bounty_id
    )
    if not row:
        raise HTTPException(404, "Bounty not found")

    if row["status"] != "settled":
        raise HTTPException(409, "Bounty is not yet settled")

    payouts = await db.fetch(
        """SELECT recipient_id, role, amount, atom_fqdn, cdg_hash
           FROM settlement_payouts
           WHERE bounty_id = $1
           ORDER BY role, amount DESC""",
        bounty_id,
    )

    return {
        "bounty_id": str(bounty_id),
        "status": "settled",
        "escrow_amount": float(row["escrow_amount"]),
        "payouts": [dict(p) for p in payouts],
    }
