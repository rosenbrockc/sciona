"""Cypher parameter builders and constraint/index definitions for provenance."""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Constraints & indexes (Memgraph DDL)
# ---------------------------------------------------------------------------

PROVENANCE_CONSTRAINTS: list[str] = [
    "CREATE CONSTRAINT ON (o:Originator) ASSERT o.originator_id IS UNIQUE",
    "CREATE CONSTRAINT ON (b:Bounty) ASSERT b.bounty_id IS UNIQUE",
    "CREATE CONSTRAINT ON (s:CDGSubmission) ASSERT s.cdg_id IS UNIQUE",
]

PROVENANCE_INDEXES: list[str] = [
    "CREATE INDEX ON :Bounty(status)",
    "CREATE INDEX ON :Bounty(deadline)",
    "CREATE INDEX ON :CDGSubmission(topo_hash)",
    "CREATE INDEX ON :Originator(github_username)",
]


# ---------------------------------------------------------------------------
# Node parameter builders
# ---------------------------------------------------------------------------


def build_originator_params(
    originator_id: str,
    *,
    github_username: str = "",
    affiliation: str = "",
) -> dict[str, Any]:
    """Build Cypher property dict for an :Originator MERGE."""
    return {
        "originator_id": originator_id,
        "github_username": github_username,
        "affiliation": affiliation,
    }


def build_bounty_params(
    bounty_id: str,
    *,
    escrow_amount: float = 0.0,
    status: str = "open",
    created_at: str = "",
    deadline: str = "",
    verification_budget: int = 5,
) -> dict[str, Any]:
    """Build Cypher property dict for a :Bounty MERGE."""
    return {
        "bounty_id": bounty_id,
        "escrow_amount": escrow_amount,
        "status": status,
        "created_at": created_at,
        "deadline": deadline,
        "verification_budget": verification_budget,
    }


def build_submission_params(
    cdg_id: str,
    *,
    topo_hash: str = "",
    verified: bool = False,
    created_at: str = "",
) -> dict[str, Any]:
    """Build Cypher property dict for a :CDGSubmission MERGE."""
    return {
        "cdg_id": cdg_id,
        "topo_hash": topo_hash,
        "verified": verified,
        "created_at": created_at,
    }


# ---------------------------------------------------------------------------
# Relationship parameter builders
# ---------------------------------------------------------------------------


def build_authored_by_params(
    atom_fqn: str,
    originator_id: str,
    contribution_share: float = 1.0,
) -> dict[str, Any]:
    """Build Cypher property dict for an :AUTHORED_BY relationship."""
    return {
        "atom_fqn": atom_fqn,
        "originator_id": originator_id,
        "contribution_share": contribution_share,
    }


def build_depends_on_params(
    cdg_id: str,
    atom_fqn: str,
    content_hash: str = "",
) -> dict[str, Any]:
    """Build Cypher property dict for a :DEPENDS_ON relationship (version-pinned)."""
    return {
        "cdg_id": cdg_id,
        "atom_fqn": atom_fqn,
        "content_hash": content_hash,
    }


def build_solved_by_params(
    bounty_id: str,
    cdg_id: str,
    metric_value: float = 0.0,
) -> dict[str, Any]:
    """Build Cypher property dict for a :SOLVED_BY relationship."""
    return {
        "bounty_id": bounty_id,
        "cdg_id": cdg_id,
        "metric_value": metric_value,
    }


def build_derives_from_params(
    child_cdg_id: str,
    parent_cdg_id: str,
) -> dict[str, Any]:
    """Build Cypher property dict for a :DERIVES_FROM relationship."""
    return {
        "child_cdg_id": child_cdg_id,
        "parent_cdg_id": parent_cdg_id,
    }
