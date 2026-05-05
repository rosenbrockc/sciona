"""Bridge funnel results into the canonical heuristic system."""

from __future__ import annotations

from typing import Any

from sciona.heuristics import (
    CanonicalHeuristic,
    HeuristicActionClass,
    HeuristicApplicabilityScope,
    HeuristicEvidenceType,
    HeuristicProducerKind,
)
from sciona.principal.runtime_heuristics import (
    RuntimeHeuristicEvidence,
    RuntimeHeuristicObservation,
)
from sciona.symbolic_funnel.contracts import FunnelCandidate, FunnelResult


_FUNNEL_HEURISTICS: dict[str, CanonicalHeuristic] = {
    "boundary_triage_pass": CanonicalHeuristic(
        heuristic_id="boundary_triage_pass",
        display_name="Boundary Triage Pass",
        dejargonized_meaning=(
            "Dataset columns satisfy the validity bounds and dimensional "
            "constraints of the candidate symbolic expression"
        ),
        evidence_type=HeuristicEvidenceType.BOOLEAN_FLAG,
        producer_kind=HeuristicProducerKind.RUNTIME_TRANSFORM,
        applicability_scope=HeuristicApplicabilityScope.CROSS_FAMILY,
        supported_action_classes=[HeuristicActionClass.PRECONDITION],
    ),
    "exponent_signature_match": CanonicalHeuristic(
        heuristic_id="exponent_signature_match",
        display_name="Exponent Signature Match",
        dejargonized_meaning=(
            "The power-law exponent fingerprint extracted from the dataset "
            "via log-space SVD matches the indexed exponent signature"
        ),
        evidence_type=HeuristicEvidenceType.BOOLEAN_FLAG,
        producer_kind=HeuristicProducerKind.RUNTIME_TRANSFORM,
        applicability_scope=HeuristicApplicabilityScope.CROSS_FAMILY,
        supported_action_classes=[HeuristicActionClass.GATE_OR_VALIDATE],
    ),
    "invariant_variance_cv": CanonicalHeuristic(
        heuristic_id="invariant_variance_cv",
        display_name="Invariant Variance CV",
        dejargonized_meaning=(
            "The coefficient of variation of the isolated invariant expression "
            "evaluated over the dataset — lower values indicate the data "
            "follows the symbolic law more closely"
        ),
        evidence_type=HeuristicEvidenceType.SCALAR_SCORE,
        producer_kind=HeuristicProducerKind.RUNTIME_TRANSFORM,
        applicability_scope=HeuristicApplicabilityScope.CROSS_FAMILY,
        supported_action_classes=[HeuristicActionClass.GATE_OR_VALIDATE],
    ),
    "ransac_fit_residual": CanonicalHeuristic(
        heuristic_id="ransac_fit_residual",
        display_name="RANSAC Fit Residual",
        dejargonized_meaning=(
            "Normalized median residual from a minimal-point RANSAC fit "
            "of the symbolic expression to the dataset"
        ),
        evidence_type=HeuristicEvidenceType.SCALAR_SCORE,
        producer_kind=HeuristicProducerKind.RUNTIME_TRANSFORM,
        applicability_scope=HeuristicApplicabilityScope.CROSS_FAMILY,
        supported_action_classes=[HeuristicActionClass.GATE_OR_VALIDATE],
    ),
    "graph_pruning_depth": CanonicalHeuristic(
        heuristic_id="graph_pruning_depth",
        display_name="Graph Pruning Depth",
        dejargonized_meaning=(
            "Number of CDG downstream nodes pruned because a root premise "
            "failed to match the dataset"
        ),
        evidence_type=HeuristicEvidenceType.SCALAR_SCORE,
        producer_kind=HeuristicProducerKind.RUNTIME_TRANSFORM,
        applicability_scope=HeuristicApplicabilityScope.CROSS_FAMILY,
        supported_action_classes=[HeuristicActionClass.GATE_OR_VALIDATE],
    ),
}


def funnel_result_to_observations(
    result: FunnelResult,
) -> list[RuntimeHeuristicObservation]:
    """Convert funnel verdicts into RuntimeHeuristicObservation instances."""
    observations: list[RuntimeHeuristicObservation] = []
    for candidate in result.ranked_candidates:
        observations.extend(_candidate_observations(candidate))
    return observations


def funnel_result_to_evidence(result: FunnelResult) -> RuntimeHeuristicEvidence:
    """Convert a FunnelResult into a RuntimeHeuristicEvidence bundle."""
    observations = funnel_result_to_observations(result)
    summary: dict[str, Any] = {
        "stages_executed": result.stages_executed,
        "timing": result.timing,
        "n_candidates": len(result.ranked_candidates),
        "equivalence_classes_tested": result.equivalence_classes_tested,
    }
    if result.ranked_candidates:
        top = result.ranked_candidates[0]
        summary["top_match"] = {
            "atom_name": top.entry.atom_name,
            "score": top.aggregate_score,
            "fitted_constants": top.fitted_constants,
        }
    return RuntimeHeuristicEvidence(
        observations=observations,
        heuristic_summary=summary,
    )


_STAGE_TO_HEURISTIC: dict[str, str] = {
    "boundary_triage": "boundary_triage_pass",
    "exponent_extraction": "exponent_signature_match",
    "invariant_variance": "invariant_variance_cv",
    "ransac": "ransac_fit_residual",
    "graph_propagation": "graph_pruning_depth",
}


def _candidate_observations(
    candidate: FunnelCandidate,
) -> list[RuntimeHeuristicObservation]:
    """Convert a single candidate's verdicts to observations."""
    observations: list[RuntimeHeuristicObservation] = []
    for verdict in candidate.verdicts:
        heuristic_id = _STAGE_TO_HEURISTIC.get(verdict.stage_name, verdict.stage_name)
        heuristic_def = _FUNNEL_HEURISTICS.get(heuristic_id)
        if heuristic_def is None:
            continue
        observations.append(
            RuntimeHeuristicObservation(
                heuristic=heuristic_def,
                source_section="symbolic_funnel",
                source_key=candidate.entry.atom_name,
                metric_name=verdict.stage_name,
                metric_value=verdict.score,
                confidence=verdict.score or (1.0 if verdict.passed else 0.0),
                supporting_fields={
                    "atom_name": candidate.entry.atom_name,
                    "expression_id": candidate.entry.expression_id,
                    **verdict.evidence,
                },
            )
        )
    return observations
