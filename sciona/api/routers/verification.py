"""Verification and settlement API endpoints."""

from __future__ import annotations

from decimal import Decimal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from sciona.api.deps import UserRow, get_db, require_auth
from sciona.api.models import PaginatedResponse
from sciona.clearinghouse.models import (
    BestScore,
    LeaderboardEntry,
    VerificationBudget,
    VerificationRun,
)
from sciona.clearinghouse.verification import (
    remaining_slots,
    should_trigger_verification,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Submission status
# ---------------------------------------------------------------------------


@router.get("/submissions/{submission_id}/status")
async def get_submission_status(
    submission_id: UUID,
    db=Depends(get_db),
) -> dict:
    """Poll verification progress for a submission."""
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


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


@router.get("/bounties/{bounty_id}/leaderboard")
async def get_leaderboard(
    bounty_id: UUID,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    db=Depends(get_db),
) -> PaginatedResponse:
    """Current ranking of verified submissions for a bounty."""
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


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------


@router.get("/bounties/{bounty_id}/settlement")
async def get_settlement(
    bounty_id: UUID,
    db=Depends(get_db),
) -> dict:
    """Retrieve settlement details for a settled bounty."""
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
