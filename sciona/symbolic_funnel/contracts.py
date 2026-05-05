"""Result models and configuration for the heuristic funnel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FunnelConfig:
    """Tunable parameters for the funnel stages."""

    # Stage 3: invariant variance
    cv_threshold: float = 0.05
    min_rows_for_cv: int = 10

    # Stage 2: exponent extraction
    exponent_snap_tolerance: float = 0.15
    max_exponent_denominator: int = 6

    # Stage 5: RANSAC
    ransac_iterations: int = 20
    ransac_holdout_size: int = 50
    ransac_residual_threshold: float = 0.10

    # Confidence calibration
    cv_high_confidence: float = 0.95
    exponent_match_confidence: float = 0.80
    ransac_confidence: float = 0.75


@dataclass
class StageVerdict:
    """Result of running one funnel stage on one candidate."""

    stage_name: str
    passed: bool
    score: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class FunnelCandidate:
    """A candidate expression that has passed through funnel stages."""

    entry: Any  # FunnelAtomEntry (avoids circular import)
    verdicts: list[StageVerdict] = field(default_factory=list)
    aggregate_score: float = 0.0
    fitted_constants: dict[str, float] = field(default_factory=dict)

    def add_verdict(self, verdict: StageVerdict) -> None:
        self.verdicts.append(verdict)
        if verdict.score is not None:
            # Aggregate is max of individual scores (higher = better match).
            self.aggregate_score = max(self.aggregate_score, verdict.score)


@dataclass
class FunnelResult:
    """Complete result of running the funnel on a dataset."""

    ranked_candidates: list[FunnelCandidate]
    stages_executed: list[str]
    timing: dict[str, float] = field(default_factory=dict)
    equivalence_classes_tested: int = 0
    total_candidates_considered: int = 0
