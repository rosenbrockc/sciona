"""Pydantic models for the clearinghouse (verification, sandbox, settlement)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pre-screen
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreScreenResult:
    """Outcome of the LLM pre-screen gate."""

    passed: bool
    rejection_reasons: list[str] = field(default_factory=list)
    estimated_tier: str = "standard"
    estimated_memory_gb: float = 0.0
    estimated_runtime_minutes: float = 0.0


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


class SandboxPayload(BaseModel):
    """Everything needed to execute a CDG in the sandbox."""

    bounty_id: str
    submission_id: str
    cdg_source: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of atom FQDN to source code, pinned by content hash.",
    )
    ageom_yml: dict = Field(default_factory=dict)
    dataset_split_ref: str = ""
    config_yml: dict = Field(default_factory=dict)
    lockfile_hash: str = ""
    atom_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of atom FQDN to content hash.",
    )


class SandboxResult(BaseModel):
    """Result returned by sandbox execution."""

    metric_values: dict[str, float] = Field(default_factory=dict)
    output_hash: str = ""
    execution_time_s: float = 0.0
    peak_memory_bytes: int = 0
    determinism_check: bool = False
    trace: dict = Field(default_factory=dict)
    error: str = ""


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class VerificationRun(BaseModel):
    """Tracks a single verification run."""

    run_id: str = ""
    bounty_id: str = ""
    submission_id: str = ""
    split_type: str = "blind"
    status: str = "pending"
    metric_values: dict[str, float] = Field(default_factory=dict)
    output_hash: str = ""
    execution_time_s: float = 0.0
    peak_memory_bytes: int = 0
    is_deterministic: bool = False
    sandbox_job_id: str = ""
    slot_consumed: bool = False


class VerificationBudget(BaseModel):
    """Budget tracking for a bounty's verification slots."""

    bounty_id: str
    tier: str = "standard"
    total_slots: int = 5
    used_slots: int = 0
    cost_per_extra: Decimal = Decimal("10.00")
    overhead_deposit: Decimal = Decimal("0")
    overhead_used: Decimal = Decimal("0")


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------


class WinningCDG(BaseModel):
    """A verified winning CDG submission."""

    submission_id: str
    architect_id: str
    cdg_hash: str
    atom_versions: dict[str, str] = Field(default_factory=dict)
    metric_values: dict[str, float] = Field(default_factory=dict)
    weight: float = 1.0


class PayoutRecipient(BaseModel):
    """A single payout recipient."""

    recipient_id: str
    role: str  # "architect" | "originator" | "platform"
    amount: Decimal
    stripe_account_id: str = ""
    atom_fqdn: str = ""
    cdg_hash: str = ""


class PayoutPlan(BaseModel):
    """Full payout breakdown for a settled bounty."""

    bounty_id: str
    escrow_amount: Decimal
    recipients: list[PayoutRecipient] = Field(default_factory=list)
    shapley_allocations: dict[str, float] = Field(default_factory=dict)
    winners: list[WinningCDG] = Field(default_factory=list)


class PayoutResult(BaseModel):
    """Result of executing payouts via Stripe."""

    bounty_id: str
    transfer_ids: list[str] = Field(default_factory=list)
    success: bool = True
    error: str = ""


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


class LeaderboardEntry(BaseModel):
    """A single entry on the bounty leaderboard."""

    submission_id: str
    architect_id: str
    metric_values: dict[str, float] = Field(default_factory=dict)
    verified_at: datetime | None = None
    rank: int = 0


class BestScore(BaseModel):
    """Best-to-date score for a metric on a bounty."""

    bounty_id: str
    metric_name: str
    best_value: float
    best_submission_id: str = ""
    is_baseline: bool = False


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------


class SplitAssignment(BaseModel):
    """A single data unit's partition assignment."""

    unit_key: str
    partition: str  # "public" | "blind"


class SplitResult(BaseModel):
    """Result of the data splitting pipeline."""

    bounty_id: str
    split_hash: str = ""
    stratify_by: str = ""
    public_count: int = 0
    blind_count: int = 0
    assignments: list[SplitAssignment] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Pareto
# ---------------------------------------------------------------------------


class ObjectiveSpec(BaseModel):
    """Specification for a single optimization objective."""

    metric: str
    direction: str = "minimize"
    weight: float = 1.0
