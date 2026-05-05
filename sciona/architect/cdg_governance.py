"""Governance checks for promoting candidate solution CDGs."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any

from sciona.architect.solution_index import SolutionTemplate, SolutionTemplateIndex
from sciona.principal.expansion_delta_planner import (
    DeltaAdaptationKind,
    DeltaPlan,
    DeltaPlanningQuery,
    plan_expansion_delta,
)


class CDGGovernanceDecision(str, Enum):
    """Promotion decision for a candidate base CDG."""

    ACCEPT_TRUE_NOVEL = "accept_true_novel"
    FLAG_BASE_PLUS_DELTA = "flag_base_plus_delta"
    REJECT_DUPLICATE = "reject_duplicate"


@dataclass(frozen=True)
class CDGSimilarityBreakdown:
    """Structural and contract similarity between two CDG templates."""

    stage_similarity: float
    concept_similarity: float
    edge_similarity: float
    io_contract_similarity: float
    family_similarity: float

    @property
    def structural_similarity(self) -> float:
        return round(
            0.45 * self.stage_similarity
            + 0.25 * self.concept_similarity
            + 0.15 * self.edge_similarity
            + 0.10 * self.io_contract_similarity
            + 0.05 * self.family_similarity,
            6,
        )


@dataclass(frozen=True)
class CDGGovernanceCandidate:
    """One existing base CDG reviewed against the candidate."""

    base_template: str
    decision: CDGGovernanceDecision
    similarity: CDGSimilarityBreakdown
    missing_terms: tuple[str, ...]
    delta_plan: DeltaPlan | None
    rationale: str

    @property
    def operation_rule_names(self) -> tuple[str, ...]:
        if self.delta_plan is None:
            return ()
        return self.delta_plan.selected.operation_rule_names


@dataclass(frozen=True)
class CDGGovernanceReport:
    """Overall promotion review for a candidate base CDG."""

    candidate_name: str
    decision: CDGGovernanceDecision
    reviews: tuple[CDGGovernanceCandidate, ...]
    rationale: str

    @property
    def should_accept_base(self) -> bool:
        return self.decision == CDGGovernanceDecision.ACCEPT_TRUE_NOVEL

    @property
    def best_existing_template(self) -> str:
        return self.reviews[0].base_template if self.reviews else ""


def review_new_base_cdg(
    candidate_cdg: dict[str, Any],
    existing_index: SolutionTemplateIndex,
    *,
    candidate_name: str = "",
    max_reviews: int = 5,
    duplicate_similarity_threshold: float = 0.94,
    base_plus_delta_similarity_threshold: float = 0.80,
    min_output_contract_similarity: float = 0.50,
    max_delta_intrusion_cost: float = 1.25,
    max_delta_operations: int = 3,
) -> CDGGovernanceReport:
    """Review whether a candidate should become a new base CDG.

    The report is intentionally conservative: exact or near-exact templates are
    rejected, mostly-isomorphic templates with a small useful delta are flagged,
    and only distinct topology or output-contract gaps are accepted as novel.
    """
    name = candidate_name or _candidate_name(candidate_cdg)
    reviews = tuple(
        sorted(
            (
                _review_against_template(
                    candidate_cdg,
                    existing,
                    duplicate_similarity_threshold=duplicate_similarity_threshold,
                    base_plus_delta_similarity_threshold=base_plus_delta_similarity_threshold,
                    min_output_contract_similarity=min_output_contract_similarity,
                    max_delta_intrusion_cost=max_delta_intrusion_cost,
                    max_delta_operations=max_delta_operations,
                )
                for existing in existing_index.templates
            ),
            key=_review_sort_key,
        )[:max_reviews]
    )

    exact = next(
        (
            review
            for review in reviews
            if review.decision == CDGGovernanceDecision.REJECT_DUPLICATE
        ),
        None,
    )
    if exact is not None:
        return CDGGovernanceReport(
            candidate_name=name,
            decision=CDGGovernanceDecision.REJECT_DUPLICATE,
            reviews=reviews,
            rationale=f"candidate is already represented by {exact.base_template}",
        )

    base_plus_delta = next(
        (
            review
            for review in reviews
            if review.decision == CDGGovernanceDecision.FLAG_BASE_PLUS_DELTA
        ),
        None,
    )
    if base_plus_delta is not None:
        return CDGGovernanceReport(
            candidate_name=name,
            decision=CDGGovernanceDecision.FLAG_BASE_PLUS_DELTA,
            reviews=reviews,
            rationale=(
                f"candidate is mostly represented by {base_plus_delta.base_template} "
                "plus expansion/refinement operations"
            ),
        )

    return CDGGovernanceReport(
        candidate_name=name,
        decision=CDGGovernanceDecision.ACCEPT_TRUE_NOVEL,
        reviews=reviews,
        rationale="no existing base CDG plus a small delta covers the candidate",
    )


def _review_against_template(
    candidate_cdg: dict[str, Any],
    existing: SolutionTemplate,
    *,
    duplicate_similarity_threshold: float,
    base_plus_delta_similarity_threshold: float,
    min_output_contract_similarity: float,
    max_delta_intrusion_cost: float,
    max_delta_operations: int,
) -> CDGGovernanceCandidate:
    similarity = _similarity(candidate_cdg, existing.raw_cdg)
    missing_terms = _candidate_only_stage_terms(candidate_cdg, existing.raw_cdg)
    delta_plan: DeltaPlan | None = None
    decision = CDGGovernanceDecision.ACCEPT_TRUE_NOVEL
    rationale = "distinct topology or output contract"

    if (
        similarity.structural_similarity >= duplicate_similarity_threshold
        and similarity.io_contract_similarity >= min_output_contract_similarity
        and not missing_terms
    ):
        decision = CDGGovernanceDecision.REJECT_DUPLICATE
        rationale = "near-identical stage topology and output contract"
    elif (
        similarity.structural_similarity >= base_plus_delta_similarity_threshold
        and similarity.io_contract_similarity >= min_output_contract_similarity
        and missing_terms
    ):
        delta_plan = plan_expansion_delta(
            DeltaPlanningQuery(
                families=_families(candidate_cdg, existing.raw_cdg),
                matched_techniques=_base_stage_terms(existing.raw_cdg),
                missing_techniques=missing_terms,
                stage_names=_stage_context(existing.raw_cdg),
                input_names=_port_names(existing.raw_cdg, "inputs"),
                output_names=_port_names(existing.raw_cdg, "outputs"),
                runtime_keys=_port_names(existing.raw_cdg, "inputs"),
                intermediate_keys=_stage_context(existing.raw_cdg),
                base_coverage=_base_coverage(candidate_cdg, missing_terms),
                min_adapted_coverage=0.50,
                max_operations_per_sequence=max_delta_operations,
            )
        )
        selected = delta_plan.selected
        if (
            delta_plan.decision
            not in (DeltaAdaptationKind.DIRECT_USE, DeltaAdaptationKind.TRUE_NOVEL)
            and selected.intrusion_cost <= max_delta_intrusion_cost
            and len(selected.operation_rule_names) <= max_delta_operations
            and not selected.missing_terms_after_plan
        ):
            decision = CDGGovernanceDecision.FLAG_BASE_PLUS_DELTA
            rationale = "mostly-isomorphic base with small expansion/refinement delta"

    return CDGGovernanceCandidate(
        base_template=existing.name,
        decision=decision,
        similarity=similarity,
        missing_terms=missing_terms,
        delta_plan=delta_plan,
        rationale=rationale,
    )


def _similarity(candidate: dict[str, Any], base: dict[str, Any]) -> CDGSimilarityBreakdown:
    return CDGSimilarityBreakdown(
        stage_similarity=_stage_similarity(candidate, base),
        concept_similarity=_counter_jaccard(_concepts(candidate), _concepts(base)),
        edge_similarity=_edge_similarity(candidate, base),
        io_contract_similarity=_io_contract_similarity(candidate, base),
        family_similarity=_family_similarity(candidate, base),
    )


def _stage_similarity(candidate: dict[str, Any], base: dict[str, Any]) -> float:
    candidate_stages = [_stage_tokens(stage) for stage in candidate.get("stages", [])]
    base_stages = [_stage_tokens(stage) for stage in base.get("stages", [])]
    if not candidate_stages and not base_stages:
        return 1.0
    if not candidate_stages or not base_stages:
        return 0.0
    recall = sum(max(_jaccard(stage, other) for other in base_stages) for stage in candidate_stages)
    precision = sum(max(_jaccard(stage, other) for other in candidate_stages) for stage in base_stages)
    return round((recall / len(candidate_stages) + precision / len(base_stages)) / 2, 6)


def _edge_similarity(candidate: dict[str, Any], base: dict[str, Any]) -> float:
    candidate_edges = _edge_tokens(candidate)
    base_edges = _edge_tokens(base)
    if not candidate_edges and not base_edges:
        return 1.0
    return round(_jaccard(candidate_edges, base_edges), 6)


def _io_contract_similarity(candidate: dict[str, Any], base: dict[str, Any]) -> float:
    candidate_outputs = set(_port_names(candidate, "outputs"))
    base_outputs = set(_port_names(base, "outputs"))
    if not candidate_outputs and not base_outputs:
        return 1.0
    output_similarity = _jaccard(candidate_outputs, base_outputs)
    input_similarity = _jaccard(set(_port_names(candidate, "inputs")), set(_port_names(base, "inputs")))
    return round(0.75 * output_similarity + 0.25 * input_similarity, 6)


def _family_similarity(candidate: dict[str, Any], base: dict[str, Any]) -> float:
    candidate_terms = {
        str(candidate.get("family", "")).lower(),
        str(candidate.get("paradigm", "")).lower(),
    } - {""}
    base_terms = {
        str(base.get("family", "")).lower(),
        str(base.get("paradigm", "")).lower(),
    } - {""}
    if not candidate_terms and not base_terms:
        return 1.0
    return round(_jaccard(candidate_terms, base_terms), 6)


def _candidate_only_stage_terms(candidate: dict[str, Any], base: dict[str, Any]) -> tuple[str, ...]:
    base_stage_tokens = [_stage_tokens(stage) for stage in base.get("stages", [])]
    missing: list[str] = []
    for stage in candidate.get("stages", []):
        tokens = _stage_tokens(stage)
        if not tokens:
            continue
        best_overlap = max((_jaccard(tokens, base_tokens) for base_tokens in base_stage_tokens), default=0.0)
        if best_overlap < 0.45:
            missing.append(_stage_name(stage))
    return tuple(dict.fromkeys(term for term in missing if term))


def _base_coverage(candidate: dict[str, Any], missing_terms: tuple[str, ...]) -> float:
    total = max(1, len(candidate.get("stages", [])))
    return round(max(0.0, min(1.0, (total - len(missing_terms)) / total)), 6)


def _base_stage_terms(base: dict[str, Any]) -> tuple[str, ...]:
    return tuple(_stage_name(stage) for stage in base.get("stages", []) if _stage_name(stage))


def _stage_context(cdg: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for stage in cdg.get("stages", []):
        values.extend(
            str(value)
            for value in (
                stage.get("stage_id", ""),
                stage.get("name", ""),
                stage.get("description", ""),
                stage.get("dejargonized_description", ""),
                stage.get("concept_type", ""),
                stage.get("matched_primitive", ""),
            )
            if value
        )
    return tuple(values)


def _families(candidate: dict[str, Any], base: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(value)
            for value in (
                candidate.get("family", ""),
                base.get("family", ""),
                candidate.get("paradigm", ""),
                base.get("paradigm", ""),
            )
            if value
        )
    )


def _concepts(cdg: dict[str, Any]) -> Counter[str]:
    return Counter(
        str(stage.get("concept_type", "")).lower()
        for stage in cdg.get("stages", [])
        if stage.get("concept_type")
    )


def _counter_jaccard(left: Counter[str], right: Counter[str]) -> float:
    if not left and not right:
        return 1.0
    keys = set(left) | set(right)
    intersection = sum(min(left[key], right[key]) for key in keys)
    union = sum(max(left[key], right[key]) for key in keys)
    return round(intersection / union if union else 0.0, 6)


def _edge_tokens(cdg: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for edge in cdg.get("edges", []):
        source = edge.get("source") or edge.get("source_id") or edge.get("from") or ""
        target = edge.get("target") or edge.get("target_id") or edge.get("to") or ""
        if source or target:
            tokens.add(f"{source}->{target}".lower())
    return tokens


def _port_names(cdg: dict[str, Any], key: str) -> tuple[str, ...]:
    names: list[str] = []
    for port in cdg.get(key, []):
        if port.get("name"):
            names.append(str(port["name"]).lower())
    for stage in cdg.get("stages", []):
        for port in stage.get(key, []):
            if port.get("name"):
                names.append(str(port["name"]).lower())
    return tuple(dict.fromkeys(names))


def _stage_name(stage: dict[str, Any]) -> str:
    return str(stage.get("name") or stage.get("stage_id") or stage.get("description") or "").strip()


def _stage_tokens(stage: dict[str, Any]) -> set[str]:
    return _tokens(
        " ".join(
            str(value)
            for value in (
                stage.get("stage_id", ""),
                stage.get("name", ""),
                stage.get("description", ""),
                stage.get("dejargonized_description", ""),
                stage.get("concept_type", ""),
                stage.get("matched_primitive", ""),
            )
            if value
        )
    )


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.split(r"[\s_\-/,;:()]+", text.lower())
        if len(token) >= 3 and token not in _STOP_WORDS
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _candidate_name(cdg: dict[str, Any]) -> str:
    return str(cdg.get("name") or cdg.get("asset_id") or "candidate_cdg")


def _review_sort_key(review: CDGGovernanceCandidate) -> tuple[int, float, float, str]:
    decision_rank = {
        CDGGovernanceDecision.REJECT_DUPLICATE: 0,
        CDGGovernanceDecision.FLAG_BASE_PLUS_DELTA: 1,
        CDGGovernanceDecision.ACCEPT_TRUE_NOVEL: 2,
    }[review.decision]
    delta_score = 0.0
    if review.delta_plan is not None:
        delta_score = review.delta_plan.selected.utility_score
    return (
        decision_rank,
        -review.similarity.structural_similarity,
        -delta_score,
        review.base_template,
    )


_STOP_WORDS = {
    "and",
    "the",
    "for",
    "with",
    "from",
    "into",
    "stage",
    "data",
    "this",
    "that",
    "using",
    "apply",
    "compute",
    "output",
    "input",
    "raw",
}
