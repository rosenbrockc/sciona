"""Hunter agent state passed through the graph."""

from __future__ import annotations

from dataclasses import dataclass, field

from ageom.types import CandidateMatch, PDGNode, VerificationResult


@dataclass
class HunterState:
    """Mutable state threaded through the Hunter graph execution."""

    pdg_node: PDGNode
    max_iterations: int = 5
    top_k_verify: int = 3
    search_k: int = 20
    mode: str = "standard"  # "standard" | "speculative_local"
    use_gbnf: bool = False
    query_batch_size: int = 40
    top_k_per_query: int = 50
    max_candidates_total: int = 3000
    verify_concurrency: int = 1  # >1 enables parallel verification

    # Accumulated across iterations
    candidates_found: list[CandidateMatch] = field(default_factory=list)
    verification_results: list[VerificationResult] = field(default_factory=list)
    queries_tried: list[str] = field(default_factory=list)
    compiler_feedback: list[str] = field(default_factory=list)

    # Current iteration
    iteration: int = 0

    # Final result (set when a verified match is found)
    verified_match: VerificationResult | None = None
