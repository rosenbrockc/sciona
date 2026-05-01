"""Result types for the high-level sciona API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GroundingReport:
    """Summary of how well a CDG's stages are grounded to atoms."""

    total_stages: int = 0
    bound_active: int = 0
    bound_approximate: int = 0
    orchestration: int = 0
    trivial_inline: int = 0
    external_knowledge: int = 0
    external_tool: int = 0
    unbound: int = 0

    @property
    def grounding_rate(self) -> float:
        resolved = (
            self.bound_active
            + self.bound_approximate
            + self.orchestration
            + self.trivial_inline
            + self.external_knowledge
            + self.external_tool
        )
        return resolved / self.total_stages if self.total_stages else 0.0


@dataclass(frozen=True)
class AtomMatch:
    """A single atom candidate from retrieval."""

    atom_name: str
    atom_fqdn: str
    score: float
    category: str
    description: str


@dataclass(frozen=True)
class AtomSearchResult:
    """Result of a catalog search query."""

    atom_name: str
    atom_fqdn: str
    source: str
    category: str
    description: str
    score: float = 0.0


@dataclass(frozen=True)
class StageMatchResult:
    """Match result for a single CDG stage."""

    stage_id: str
    stage_name: str
    action_class: str
    matched_atom: str | None = None
    match_confidence: float = 0.0
    top_candidates: list[AtomMatch] = field(default_factory=list)
    reasoning: str = ""


@dataclass(frozen=True)
class GapReport:
    """Catalog coverage gaps for a CDG."""

    total_stages: int
    covered: int
    gaps: list[str]
    gap_details: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class CDGInspection:
    """Inspection summary of a CDG."""

    total_stages: int
    total_edges: int
    grounding: GroundingReport
    concept_types: dict[str, int] = field(default_factory=dict)
    max_depth: int = 0


@dataclass(frozen=True)
class GeneratedCode:
    """Result of code generation from a grounded CDG."""

    source: str
    imports: list[str] = field(default_factory=list)
    atom_fqdns_used: list[str] = field(default_factory=list)


@dataclass
class ProposalResult:
    """Result of Sciona.propose() — the main validation entry point."""

    cdg: Any
    template_used: str | None = None
    template_match_score: float = 0.0
    grounding: GroundingReport = field(default_factory=GroundingReport)
    matches: list[StageMatchResult] = field(default_factory=list)
    reasoning: str = ""
    wall_time_seconds: float = 0.0
