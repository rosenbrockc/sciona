"""Pydantic/dataclass models for the ecosystem layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkRecord:
    """A single benchmark measurement for an atom version."""

    atom_fqdn: str
    content_hash: str
    benchmark_id: str
    metric_name: str
    metric_value: float
    dataset_tag: str
    measured_at: str


class BenchmarkSuite(BaseModel):
    """A curated benchmark suite definition."""

    benchmark_id: str
    domain_tags: list[str] = Field(default_factory=list)
    description: str = ""
    dataset_s3_key: str = ""
    metric_names: list[str] = Field(default_factory=list)
    curation_source: str = "foundation"
    proposer_id: str | None = None
    vote_count: int = 0
    status: str = "active"


class BenchmarkProposal(BaseModel):
    """Community-submitted benchmark proposal."""

    benchmark_id: str
    domain_tags: list[str] = Field(default_factory=list)
    description: str = ""
    dataset_s3_key: str = ""
    metric_names: list[str] = Field(default_factory=list)


class BenchmarkVote(BaseModel):
    """A vote on a benchmark proposal."""

    benchmark_id: str
    voter_id: str
    vote: str = "approve"  # "approve" | "reject"


# ---------------------------------------------------------------------------
# Fuzzing
# ---------------------------------------------------------------------------


class FuzzJobMessage(BaseModel):
    """Message schema for the fuzz job queue."""

    atom_fqdn: str
    content_hash: str
    iospec: list[dict] = Field(default_factory=list)
    tunable_params: list[dict] = Field(default_factory=list)
    benchmark_ids: list[str] = Field(default_factory=list)


class FuzzResult(BaseModel):
    """Result of a single fuzz strategy run."""

    atom_fqdn: str
    content_hash: str
    strategy: str  # "property_based" | "boundary_value" | "param_smoothing" | "behavioral_equiv"
    passed: bool
    failures: list[dict] = Field(default_factory=list)
    inputs_tested: int = 0
    runtime_ms: int = 0


class BehavioralEquivalenceFlag(BaseModel):
    """Flag for a pair of atoms with similar behavior."""

    atom_a_fqdn: str
    atom_a_hash: str
    atom_b_fqdn: str
    atom_b_hash: str
    match_ratio: float
    sample_size: int
    reviewed: bool = False
    disposition: str = ""


# ---------------------------------------------------------------------------
# Soft deprecation
# ---------------------------------------------------------------------------


class SupersessionCheck(BaseModel):
    """Input for supersession detection."""

    old_fqdn: str
    old_hash: str
    new_fqdn: str
    new_hash: str
    benchmarks_old: list[BenchmarkRecord] = Field(default_factory=list)
    benchmarks_new: list[BenchmarkRecord] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


class OriginatorImpact(BaseModel):
    """Algorithmic Impact Factor for an originator."""

    originator_id: str
    github_username: str = ""
    affiliation: str = ""
    bounty_count: int = 0
    total_bounty_value: float = 0.0
    atom_count: int = 0
    h_index: int = 0


class ComputePreserved(BaseModel):
    """Aggregate compute-preserved metrics."""

    total_bounties_settled: int = 0
    total_escrow_value: float = 0.0
    estimated_tokens_saved: int = 0
    estimated_cost_saved_usd: float = 0.0


class CrossDisciplineUsage(BaseModel):
    """Cross-disciplinary usage of an atom."""

    atom_fqdn: str
    original_domain: list[str] = Field(default_factory=list)
    bounty_domains: list[str] = Field(default_factory=list)
    cross_uses: int = 0


# ---------------------------------------------------------------------------
# Discipline repo sync
# ---------------------------------------------------------------------------


class DisciplineRepo(BaseModel):
    """A registered discipline repository."""

    repo_url: str
    webhook_secret: str = ""
    domain_tags: list[str] = Field(default_factory=list)
    maintainer_ids: list[str] = Field(default_factory=list)
    last_synced_commit: str = ""
