"""Pydantic models for the provenance graph."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Originator(BaseModel):
    """An individual or organisation that authored one or more atoms."""

    originator_id: str
    github_username: str = ""
    affiliation: str = ""


class CDGSubmission(BaseModel):
    """A CDG submitted as a bounty solution."""

    cdg_id: str
    topo_hash: str
    verified: bool = False
    created_at: str = ""


class Bounty(BaseModel):
    """A posted bounty derived from a Dead-End Flare."""

    bounty_id: str
    escrow_amount: float = 0.0
    status: str = Field(
        default="open",
        description="Lifecycle state: open|submitted|verified|settled|expired|cancelled",
    )
    created_at: str = ""
    deadline: str = ""
    verification_budget: int = 5
