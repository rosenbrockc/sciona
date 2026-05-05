"""Plan least-intrusive expansion/refinement deltas for a base CDG."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sciona.architect.handoff import CDGExport
from sciona.principal.expansion_retrieval import (
    ExpansionAssetRetriever,
    ExpansionOperationSequence,
    ExpansionRetrievalQuery,
)


class DeltaAdaptationKind(str, Enum):
    """Planner-level decision for how to handle a solution delta."""

    DIRECT_USE = "direct_use"
    REFINEMENT = "refinement"
    EXPANSION = "expansion"
    EXPANSION_PACK = "expansion_pack"
    TRUE_NOVEL = "true_novel"


@dataclass(frozen=True)
class DeltaPlanningQuery:
    """Request for planning a base-CDG-to-adapted-CDG delta."""

    families: tuple[str, ...] = ()
    matched_techniques: tuple[str, ...] = ()
    missing_techniques: tuple[str, ...] = ()
    stage_names: tuple[str, ...] = ()
    input_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()
    runtime_keys: tuple[str, ...] = ()
    intermediate_keys: tuple[str, ...] = ()
    base_coverage: float = 0.0
    direct_use_threshold: float = 0.98
    min_adapted_coverage: float = 0.65
    max_sequences: int = 5
    max_operations_per_sequence: int = 4
    min_operation_score: float = 0.10


@dataclass(frozen=True)
class DeltaPlanCandidate:
    """One possible adaptation path from a base CDG to an adapted CDG."""

    adaptation_kind: DeltaAdaptationKind
    operation_sequence: ExpansionOperationSequence | None
    projected_coverage: float
    intrusion_cost: float
    utility_score: float
    covered_terms: tuple[str, ...] = ()
    missing_terms_after_plan: tuple[str, ...] = ()
    path: tuple[str, ...] = ()
    rationale: str = ""

    @property
    def operation_rule_names(self) -> tuple[str, ...]:
        if self.operation_sequence is None:
            return ()
        return tuple(operation.rule_name for operation in self.operation_sequence.operations)


@dataclass(frozen=True)
class DeltaPlan:
    """Ranked delta-planning result for an architect path."""

    decision: DeltaAdaptationKind
    base_coverage: float
    direct_use_coverage: float
    candidate_count: int
    candidates: tuple[DeltaPlanCandidate, ...]
    selected: DeltaPlanCandidate
    should_compose_novel: bool


_REFINEMENT_OPERATION_TYPES = {
    "diagnostic",
    "precondition",
    "postcondition",
    "validation",
}


def plan_expansion_delta(
    query: DeltaPlanningQuery,
    *,
    cdg: CDGExport | None = None,
    retriever: ExpansionAssetRetriever | None = None,
) -> DeltaPlan:
    """Choose the least-intrusive useful path from base CDG to adapted CDG."""
    if not _unique(query.missing_techniques) or query.base_coverage >= query.direct_use_threshold:
        candidate = DeltaPlanCandidate(
            adaptation_kind=DeltaAdaptationKind.DIRECT_USE,
            operation_sequence=None,
            projected_coverage=round(_clamp(query.base_coverage), 6),
            intrusion_cost=0.0,
            utility_score=float("inf"),
            covered_terms=(),
            missing_terms_after_plan=(),
            path=("base_cdg", "direct_use"),
            rationale="base CDG already covers the requested techniques",
        )
        return DeltaPlan(
            decision=DeltaAdaptationKind.DIRECT_USE,
            base_coverage=round(_clamp(query.base_coverage), 6),
            direct_use_coverage=round(_clamp(query.base_coverage), 6),
            candidate_count=1,
            candidates=(candidate,),
            selected=candidate,
            should_compose_novel=False,
        )

    asset_retriever = retriever or ExpansionAssetRetriever()
    sequences = asset_retriever.retrieve_sequences(
        ExpansionRetrievalQuery(
            families=query.families,
            missing_techniques=query.missing_techniques,
            stage_names=query.stage_names,
            input_names=query.input_names,
            output_names=query.output_names,
            runtime_keys=query.runtime_keys,
            intermediate_keys=query.intermediate_keys,
        ),
        cdg=cdg,
        min_operation_score=query.min_operation_score,
        max_sequences=query.max_sequences,
        max_operations_per_sequence=query.max_operations_per_sequence,
    )
    candidates = tuple(
        _candidate_from_sequence(query, sequence)
        for sequence in sequences
        if sequence.covered_terms
    )

    viable = tuple(
        candidate
        for candidate in candidates
        if candidate.projected_coverage >= query.min_adapted_coverage
    )
    if viable:
        ranked = tuple(sorted(viable, key=_candidate_sort_key))
        selected = ranked[0]
        all_candidates = tuple(sorted(candidates, key=_candidate_sort_key))
        return DeltaPlan(
            decision=selected.adaptation_kind,
            base_coverage=round(_clamp(query.base_coverage), 6),
            direct_use_coverage=round(_clamp(query.base_coverage), 6),
            candidate_count=len(all_candidates),
            candidates=all_candidates,
            selected=selected,
            should_compose_novel=False,
        )

    novel = _true_novel_candidate(query)
    all_candidates = tuple(sorted((*candidates, novel), key=_candidate_sort_key))
    return DeltaPlan(
        decision=DeltaAdaptationKind.TRUE_NOVEL,
        base_coverage=round(_clamp(query.base_coverage), 6),
        direct_use_coverage=round(_clamp(query.base_coverage), 6),
        candidate_count=len(all_candidates),
        candidates=all_candidates,
        selected=novel,
        should_compose_novel=True,
    )


def _candidate_from_sequence(
    query: DeltaPlanningQuery,
    sequence: ExpansionOperationSequence,
) -> DeltaPlanCandidate:
    covered_terms = _covered_missing_terms(query.missing_techniques, sequence.covered_terms)
    missing_after = _missing_after_plan(query.missing_techniques, covered_terms)
    projected = _projected_coverage(query, covered_terms)
    intrusion = max(0.0, sequence.intrusion_cost)
    utility = projected / max(0.10, intrusion)
    kind = _adaptation_kind(sequence)
    return DeltaPlanCandidate(
        adaptation_kind=kind,
        operation_sequence=sequence,
        projected_coverage=round(projected, 6),
        intrusion_cost=round(intrusion, 6),
        utility_score=round(utility, 6),
        covered_terms=covered_terms,
        missing_terms_after_plan=missing_after,
        path=("base_cdg", *[operation.rule_name for operation in sequence.operations], "adapted_cdg"),
        rationale=(
            f"{kind.value} covers {len(covered_terms)} missing technique(s) "
            f"with intrusion cost {intrusion:.2f}"
        ),
    )


def _adaptation_kind(sequence: ExpansionOperationSequence) -> DeltaAdaptationKind:
    operations = sequence.operations
    if len(operations) > 1:
        return DeltaAdaptationKind.EXPANSION_PACK
    operation_type = operations[0].operation_type.lower() if operations else ""
    if operation_type in _REFINEMENT_OPERATION_TYPES:
        return DeltaAdaptationKind.REFINEMENT
    return DeltaAdaptationKind.EXPANSION


def _true_novel_candidate(query: DeltaPlanningQuery) -> DeltaPlanCandidate:
    missing = _unique(query.missing_techniques)
    return DeltaPlanCandidate(
        adaptation_kind=DeltaAdaptationKind.TRUE_NOVEL,
        operation_sequence=None,
        projected_coverage=round(_clamp(query.base_coverage), 6),
        intrusion_cost=float("inf"),
        utility_score=0.0,
        covered_terms=(),
        missing_terms_after_plan=missing,
        path=("base_cdg", "true_novel_composition"),
        rationale="no expansion/refinement sequence covers enough missing techniques",
    )


def _projected_coverage(query: DeltaPlanningQuery, covered_terms: tuple[str, ...]) -> float:
    matched = _normalized_set(query.matched_techniques)
    missing = _normalized_set(query.missing_techniques)
    if not missing:
        return _clamp(query.base_coverage)
    covered = _normalized_set(covered_terms) & missing
    technique_projection = (len(matched) + len(covered)) / max(1, len(matched | missing))
    return _clamp(max(query.base_coverage, technique_projection))


def _covered_missing_terms(
    missing_techniques: tuple[str, ...],
    covered_terms: tuple[str, ...],
) -> tuple[str, ...]:
    covered_normalized = _normalized_set(covered_terms)
    return tuple(
        term for term in _unique(missing_techniques) if _normalize_phrase(term) in covered_normalized
    )


def _missing_after_plan(
    missing_techniques: tuple[str, ...],
    covered_terms: tuple[str, ...],
) -> tuple[str, ...]:
    covered_normalized = _normalized_set(covered_terms)
    return tuple(
        term for term in _unique(missing_techniques) if _normalize_phrase(term) not in covered_normalized
    )


def _candidate_sort_key(candidate: DeltaPlanCandidate) -> tuple[float, float, float, str]:
    sequence_name = ""
    if candidate.operation_sequence is not None:
        sequence_name = candidate.operation_sequence.asset_family
    return (
        -candidate.utility_score,
        -candidate.projected_coverage,
        candidate.intrusion_cost,
        sequence_name,
    )


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _normalized_set(values: tuple[str, ...]) -> set[str]:
    return {_normalize_phrase(value) for value in values if str(value).strip()}


def _normalize_phrase(value: str) -> str:
    return " ".join(str(value).lower().split())


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))

