"""Bounty lifecycle endpoints."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from sciona.api import deps as api_deps
from sciona.api.bounty_state import (
    InvalidTransition,
    compute_cancellation_fee,
    validate_transition,
)
from sciona.api.models import (
    BountyCancelResponse,
    BountyCreateRequest,
    BountyFundResponse,
    BountyResponse,
    BountySummaryResponse,
    PaginatedResponse,
    SubmissionRequest,
    SubmissionResponse,
    UpdateTargetRequest,
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


def _json_obj(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        import json

        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return value


@router.post("")
async def create_bounty(
    body: BountyCreateRequest,
    user: UserRow = Depends(require_auth),
    supabase=Depends(_get_supabase),
    db=Depends(get_db),
) -> BountyResponse:
    """Create a draft bounty."""
    user_id = str(user.user_id)
    if supabase is not None:
        result = await (
            supabase.table("bounties")
            .insert(
                {
                    "principal_id": user_id,
                    "title": body.title,
                    "escrow_amount": body.escrow_amount,
                    "deadline": body.deadline,
                    "tier": body.tier,
                    "config_yml": body.config_yml,
                    "flare_payload": body.flare_payload,
                }
            )
            .execute()
        )
        created = _first_row(result.data)
        if not created:
            raise HTTPException(500, "Failed to create bounty")
        return _bounty_response(created)

    row = await db.fetchrow(
        """INSERT INTO bounties
           (principal_id, title, escrow_amount, deadline, tier, config_yml, flare_payload)
           VALUES ($1::uuid, $2, $3, $4, $5, $6::jsonb, $7::jsonb)
           RETURNING *""",
        user_id,
        body.title,
        body.escrow_amount,
        body.deadline,
        body.tier,
        __import__("json").dumps(body.config_yml),
        __import__("json").dumps(body.flare_payload) if body.flare_payload else None,
    )

    return _bounty_response(row)


@router.post("/{bounty_id}/fund")
async def fund_bounty(
    bounty_id: UUID,
    user: UserRow = Depends(require_auth),
    supabase=Depends(_get_supabase),
    db=Depends(get_db),
) -> BountyFundResponse:
    """Fund a bounty (transitions draft -> open)."""
    row = await _fetch_bounty(bounty_id, supabase=supabase, db=db)
    if not row:
        raise HTTPException(404, "Bounty not found")
    if str(row["principal_id"]) != str(user.user_id):
        raise HTTPException(403, "Only the bounty creator can fund it")

    try:
        new_status = validate_transition(row["status"], "fund")
    except InvalidTransition as e:
        raise HTTPException(409, str(e))

    if supabase is not None:
        await (
            supabase.table("bounties")
            .update({"status": new_status})
            .eq("bounty_id", str(bounty_id))
            .execute()
        )
    else:
        await db.execute(
            "UPDATE bounties SET status = $1, updated_at = now() WHERE bounty_id = $2",
            new_status,
            bounty_id,
        )

    return BountyFundResponse(
        bounty_id=bounty_id,
        status=new_status,
        checkout_url="",
    )


@router.post("/{bounty_id}/submit")
async def submit_to_bounty(
    bounty_id: UUID,
    body: SubmissionRequest,
    user: UserRow = Depends(require_auth),
    supabase=Depends(_get_supabase),
    db=Depends(get_db),
) -> SubmissionResponse:
    """Submit a CDG solution with signed receipt."""
    row = await _fetch_bounty(bounty_id, supabase=supabase, db=db)
    if not row:
        raise HTTPException(404, "Bounty not found")

    if row["status"] == "open":
        try:
            validate_transition(row["status"], "submit")
        except InvalidTransition as e:
            raise HTTPException(409, str(e))
        if supabase is not None:
            await (
                supabase.table("bounties")
                .update({"status": "submitted"})
                .eq("bounty_id", str(bounty_id))
                .execute()
            )
        else:
            await db.execute(
                "UPDATE bounties SET status = 'submitted', updated_at = now() WHERE bounty_id = $1",
                bounty_id,
            )
    elif row["status"] != "submitted":
        raise HTTPException(409, f"Cannot submit to bounty in {row['status']!r} state")

    if supabase is not None:
        sub_result = await (
            supabase.table("submissions")
            .insert(
                {
                    "bounty_id": str(bounty_id),
                    "architect_id": str(user.user_id),
                    "cdg_hash": body.cdg_hash,
                    "atom_versions": body.atom_versions,
                    "receipt_s3": "",
                    "receipt_json": body.receipt_json,
                    "claimed_metric_name": body.claimed_metric_name,
                    "claimed_metric_value": body.claimed_metric_value,
                }
            )
            .execute()
        )
        created = _first_row(sub_result.data)
        if not created:
            raise HTTPException(500, "Failed to create submission")
        sub_row = await (
            supabase.table("submissions")
            .select("submission_id, bounty_id, verification_status, submitted_at")
            .eq("submission_id", created["submission_id"])
            .maybe_single()
            .execute()
        )
        row = _first_row(sub_row.data) or created
        return SubmissionResponse(**dict(row))

    import json

    sub_row = await db.fetchrow(
        """INSERT INTO submissions
           (bounty_id, architect_id, cdg_hash, atom_versions, receipt_s3,
            receipt_json, claimed_metric_name, claimed_metric_value)
           VALUES ($1, $2::uuid, $3, $4::jsonb, '', $5::jsonb, $6, $7)
           RETURNING submission_id, bounty_id, verification_status, submitted_at""",
        bounty_id,
        user.user_id,
        body.cdg_hash,
        json.dumps(body.atom_versions),
        json.dumps(body.receipt_json),
        body.claimed_metric_name,
        body.claimed_metric_value,
    )

    return SubmissionResponse(**dict(sub_row))


@router.post("/{bounty_id}/cancel")
async def cancel_bounty(
    bounty_id: UUID,
    user: UserRow = Depends(require_auth),
    supabase=Depends(_get_supabase),
    db=Depends(get_db),
) -> BountyCancelResponse:
    """Cancel a bounty (with fee per design decision 4.13)."""
    row = await _fetch_bounty(bounty_id, supabase=supabase, db=db)
    if not row:
        raise HTTPException(404, "Bounty not found")
    if str(row["principal_id"]) != str(user.user_id):
        raise HTTPException(403, "Only the bounty creator can cancel it")

    if supabase is not None:
        submissions_result = await (
            supabase.table("submissions")
            .select("submission_id", count="exact")
            .eq("bounty_id", str(bounty_id))
            .execute()
        )
        has_submissions = bool(submissions_result.count or submissions_result.data)
    else:
        has_submissions = await db.fetchval(
            "SELECT EXISTS(SELECT 1 FROM submissions WHERE bounty_id = $1)",
            bounty_id,
        )

    try:
        action = "cancel_open" if row["status"] == "open" else "cancel_draft"
        new_status = validate_transition(row["status"], action)
        fee = compute_cancellation_fee(
            Decimal(str(row["escrow_amount"])),
            row["status"],
            has_submissions,
        )
    except InvalidTransition as e:
        raise HTTPException(409, str(e))

    if supabase is not None:
        await (
            supabase.table("bounties")
            .update({"status": new_status, "cancellation_fee": float(fee)})
            .eq("bounty_id", str(bounty_id))
            .execute()
        )
    else:
        await db.execute(
            """UPDATE bounties
               SET status = $1, cancellation_fee = $2, updated_at = now()
               WHERE bounty_id = $3""",
            new_status,
            float(fee),
            bounty_id,
        )

    return BountyCancelResponse(
        bounty_id=bounty_id,
        status=new_status,
        cancellation_fee=float(fee),
    )


@router.post("/{bounty_id}/target")
async def update_target(
    bounty_id: UUID,
    body: UpdateTargetRequest,
    user: UserRow = Depends(require_auth),
    supabase=Depends(_get_supabase),
    db=Depends(get_db),
) -> BountyResponse:
    """Principal updates minimum metric target between verifications."""
    row = await _fetch_bounty(bounty_id, supabase=supabase, db=db)
    if not row:
        raise HTTPException(404, "Bounty not found")
    if str(row["principal_id"]) != str(user.user_id):
        raise HTTPException(403, "Only the bounty creator can update the target")
    if row["status"] not in ("open", "submitted"):
        raise HTTPException(409, "Can only update target for open/submitted bounties")

    config = _json_obj(row["config_yml"])
    config["min_metric_value"] = body.min_metric_value

    if supabase is not None:
        await (
            supabase.table("bounties")
            .update({"config_yml": config})
            .eq("bounty_id", str(bounty_id))
            .execute()
        )
        updated = await _fetch_bounty(bounty_id, supabase=supabase, db=db)
        return _bounty_response(updated)

    import json

    await db.execute(
        "UPDATE bounties SET config_yml = $1::jsonb, updated_at = now() WHERE bounty_id = $2",
        json.dumps(config),
        bounty_id,
    )

    updated = await db.fetchrow(
        "SELECT * FROM bounties WHERE bounty_id = $1", bounty_id
    )
    return _bounty_response(updated)


@router.get("/{bounty_id}")
async def get_bounty(
    bounty_id: UUID,
    supabase=Depends(_get_supabase),
    db=Depends(get_db),
) -> BountyResponse:
    """Get bounty details including submission count and status."""
    row = await _fetch_bounty(bounty_id, supabase=supabase, db=db)
    if not row:
        raise HTTPException(404, "Bounty not found")
    return _bounty_response(row, db=db, bounty_id=bounty_id)


@router.get("")
async def list_bounties(
    status: str | None = None,
    domain_tag: str | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    supabase=Depends(_get_supabase),
    db=Depends(get_db),
) -> PaginatedResponse:
    """List bounties with optional filters."""
    limit = int(getattr(limit, "default", limit))
    offset = int(getattr(offset, "default", offset))

    if supabase is not None:
        query = supabase.table("bounties").select(
            "bounty_id, title, escrow_amount, status, deadline, tier, created_at",
            count="exact",
        )
        if status:
            query = query.eq("status", status)
        # Domain tags are not stored on bounties in the current schema; keep the
        # parameter for API compatibility but do not filter on it here.
        result = await query.order("created_at", desc=True).range(
            offset, offset + limit - 1
        ).execute()
        rows = result.data or []
        total = int(result.count or len(rows))
        items = [
            BountySummaryResponse(
                bounty_id=r["bounty_id"],
                title=r["title"],
                escrow_amount=float(r["escrow_amount"]),
                status=r["status"],
                deadline=r["deadline"],
                tier=r["tier"],
            )
            for r in rows
        ]
        return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)

    conditions = ["1=1"]
    params: list[Any] = []
    idx = 1

    if status:
        conditions.append(f"b.status = ${idx}")
        params.append(status)
        idx += 1

    where = " AND ".join(conditions)

    count_row = await db.fetchrow(
        f"SELECT COUNT(*) AS cnt FROM bounties b WHERE {where}", *params
    )
    total = count_row["cnt"] if count_row else 0

    params.extend([limit, offset])
    rows = await db.fetch(
        f"""SELECT b.bounty_id, b.title, b.escrow_amount, b.status,
                   b.deadline, b.tier, b.created_at
            FROM bounties b
            WHERE {where}
            ORDER BY b.created_at DESC
            LIMIT ${idx} OFFSET ${idx + 1}""",
        *params,
    )

    items = [
        BountySummaryResponse(
            bounty_id=r["bounty_id"],
            title=r["title"],
            escrow_amount=float(r["escrow_amount"]),
            status=r["status"],
            deadline=r["deadline"],
            tier=r["tier"],
        )
        for r in rows
    ]

    return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)


def _bounty_response(row, *, db=None, bounty_id=None) -> BountyResponse:
    """Convert a database row to a BountyResponse."""
    return BountyResponse(
        bounty_id=row["bounty_id"],
        principal_id=row["principal_id"],
        title=row["title"],
        escrow_amount=float(row["escrow_amount"]),
        status=row["status"],
        deadline=row["deadline"],
        tier=row["tier"],
        verification_budget=row["verification_budget"],
        verifications_used=row["verifications_used"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def _fetch_bounty(
    bounty_id: UUID,
    *,
    supabase=None,
    db=None,
) -> dict[str, Any] | None:
    if supabase is not None:
        result = await (
            supabase.table("bounties")
            .select("*")
            .eq("bounty_id", str(bounty_id))
            .maybe_single()
            .execute()
        )
        return _first_row(result.data)
    return await db.fetchrow("SELECT * FROM bounties WHERE bounty_id = $1", bounty_id)
