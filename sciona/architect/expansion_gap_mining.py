"""Mine reusable expansion/refinement gaps from validation results."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from sciona.principal.expansion_retrieval import (
    ExpansionAssetRetriever,
    ExpansionOperationSequence,
    ExpansionRetrievalQuery,
)


@dataclass(frozen=True)
class ValidationGapOccurrence:
    """One missing technique from one validation case."""

    competition_id: str
    title: str
    assessment: str
    template: str
    family: str
    paradigm: str
    technique: str
    coverage_source: str


@dataclass(frozen=True)
class ExpansionGapCluster:
    """Reusable missing-technique cluster with operation-mining guidance."""

    cluster_id: str
    representative_terms: tuple[str, ...]
    support: int
    competitions: tuple[str, ...]
    families: tuple[str, ...]
    paradigms: tuple[str, ...]
    recommended_action: str
    existing_asset_family: str = ""
    existing_asset_id: str = ""
    existing_operation_rule_names: tuple[str, ...] = ()
    rationale: str = ""


@dataclass(frozen=True)
class ExpansionGapMiningReport:
    """Aggregated reusable-gap mining report."""

    total_results: int
    included_results: int
    occurrence_count: int
    clusters: tuple[ExpansionGapCluster, ...]

    @property
    def reusable_candidate_count(self) -> int:
        return sum(1 for cluster in self.clusters if cluster.recommended_action == "candidate_reusable_operation")

    @property
    def existing_operation_count(self) -> int:
        return sum(1 for cluster in self.clusters if cluster.recommended_action == "covered_by_existing_operation")

    @property
    def one_off_count(self) -> int:
        return sum(1 for cluster in self.clusters if cluster.recommended_action == "defer_one_off")

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_results": self.total_results,
            "included_results": self.included_results,
            "occurrence_count": self.occurrence_count,
            "cluster_count": len(self.clusters),
            "existing_operation_count": self.existing_operation_count,
            "reusable_candidate_count": self.reusable_candidate_count,
            "one_off_count": self.one_off_count,
            "clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "representative_terms": list(cluster.representative_terms),
                    "support": cluster.support,
                    "competitions": list(cluster.competitions),
                    "families": list(cluster.families),
                    "paradigms": list(cluster.paradigms),
                    "recommended_action": cluster.recommended_action,
                    "existing_asset_family": cluster.existing_asset_family,
                    "existing_asset_id": cluster.existing_asset_id,
                    "existing_operation_rule_names": list(cluster.existing_operation_rule_names),
                    "rationale": cluster.rationale,
                }
                for cluster in self.clusters
            ],
        }


_DEFAULT_ASSESSMENTS = frozenset(
    {
        "partial",
        "divergent",
        "inadequate",
        "no_template",
        "no_evaluation",
    }
)


def load_validation_results(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load and concatenate validation result JSON files."""
    results: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        if isinstance(data, list):
            results.extend(item for item in data if isinstance(item, dict))
        elif isinstance(data, dict):
            rows = data.get("results")
            if isinstance(rows, list):
                results.extend(item for item in rows if isinstance(item, dict))
            else:
                results.append(data)
    return results


def mine_expansion_gaps(
    validation_results: Iterable[dict[str, Any]],
    *,
    min_support: int = 2,
    similarity_threshold: float = 0.45,
    include_assessments: Iterable[str] = _DEFAULT_ASSESSMENTS,
    retriever: ExpansionAssetRetriever | None = None,
    max_clusters: int = 50,
) -> ExpansionGapMiningReport:
    """Cluster missing validation techniques into reusable operation gaps."""
    results = tuple(validation_results)
    included_assessments = {assessment.lower() for assessment in include_assessments}
    included_results = tuple(
        result
        for result in results
        if str(result.get("assessment", "")).lower() in included_assessments
    )
    occurrences = _extract_occurrences(included_results)
    raw_clusters = _cluster_occurrences(occurrences, similarity_threshold=similarity_threshold)
    asset_retriever = retriever or ExpansionAssetRetriever()
    clusters = tuple(
        _build_cluster(index, cluster, min_support=min_support, retriever=asset_retriever)
        for index, cluster in enumerate(raw_clusters, start=1)
    )
    ranked = tuple(sorted(clusters, key=_cluster_sort_key)[:max_clusters])
    return ExpansionGapMiningReport(
        total_results=len(results),
        included_results=len(included_results),
        occurrence_count=len(occurrences),
        clusters=ranked,
    )


def _extract_occurrences(results: Iterable[dict[str, Any]]) -> tuple[ValidationGapOccurrence, ...]:
    occurrences: list[ValidationGapOccurrence] = []
    for result in results:
        evaluation = _base_evaluation(result)
        missing = evaluation.get("missing_techniques") or ()
        top_match = _top_match(result)
        for technique in missing:
            technique_text = str(technique).strip()
            if not technique_text:
                continue
            occurrences.append(
                ValidationGapOccurrence(
                    competition_id=str(result.get("competition_id", "")),
                    title=str(result.get("title", "")),
                    assessment=str(result.get("assessment", "")),
                    template=str(top_match.get("template", "")),
                    family=str(top_match.get("family", "")),
                    paradigm=str(top_match.get("paradigm", "")),
                    technique=technique_text,
                    coverage_source=str(evaluation.get("coverage_source", "")),
                )
            )
    return tuple(occurrences)


def _base_evaluation(result: dict[str, Any]) -> dict[str, Any]:
    evaluation = result.get("base_evaluation") or result.get("evaluation") or {}
    return evaluation if isinstance(evaluation, dict) else {}


def _top_match(result: dict[str, Any]) -> dict[str, Any]:
    matches = result.get("template_matches") or ()
    if isinstance(matches, list) and matches and isinstance(matches[0], dict):
        return matches[0]
    return {}


def _cluster_occurrences(
    occurrences: tuple[ValidationGapOccurrence, ...],
    *,
    similarity_threshold: float,
) -> list[list[ValidationGapOccurrence]]:
    clusters: list[list[ValidationGapOccurrence]] = []
    cluster_tokens: list[set[str]] = []
    for occurrence in occurrences:
        tokens = _tokens(occurrence.technique)
        best_index = -1
        best_score = 0.0
        for index, existing_tokens in enumerate(cluster_tokens):
            score = _jaccard(tokens, existing_tokens)
            if score > best_score:
                best_score = score
                best_index = index
        if best_index >= 0 and best_score >= similarity_threshold:
            clusters[best_index].append(occurrence)
            cluster_tokens[best_index] |= tokens
        else:
            clusters.append([occurrence])
            cluster_tokens.append(set(tokens))
    return clusters


def _build_cluster(
    index: int,
    occurrences: list[ValidationGapOccurrence],
    *,
    min_support: int,
    retriever: ExpansionAssetRetriever,
) -> ExpansionGapCluster:
    terms = _representative_terms(occurrences)
    competitions = tuple(sorted({occurrence.competition_id for occurrence in occurrences if occurrence.competition_id}))
    families = _most_common(occurrence.family for occurrence in occurrences if occurrence.family)
    paradigms = _most_common(occurrence.paradigm for occurrence in occurrences if occurrence.paradigm)
    sequence = _existing_sequence(terms, families, paradigms, retriever)
    if sequence is not None and set(sequence.covered_terms) & set(terms):
        action = "covered_by_existing_operation"
        rationale = "existing expansion/refinement asset covers this reusable gap"
        asset_family = sequence.asset_family
        asset_id = sequence.asset_id
        rule_names = tuple(operation.rule_name for operation in sequence.operations)
    elif len(occurrences) >= min_support:
        action = "candidate_reusable_operation"
        rationale = "recurs across validation cases but is not covered by existing operations"
        asset_family = ""
        asset_id = ""
        rule_names = ()
    else:
        action = "defer_one_off"
        rationale = "single validation case; defer until it recurs or exposes a broader pattern"
        asset_family = ""
        asset_id = ""
        rule_names = ()
    return ExpansionGapCluster(
        cluster_id=f"gap_cluster_{index:03d}",
        representative_terms=terms,
        support=len(occurrences),
        competitions=competitions,
        families=families,
        paradigms=paradigms,
        recommended_action=action,
        existing_asset_family=asset_family,
        existing_asset_id=asset_id,
        existing_operation_rule_names=rule_names,
        rationale=rationale,
    )


def _existing_sequence(
    terms: tuple[str, ...],
    families: tuple[str, ...],
    paradigms: tuple[str, ...],
    retriever: ExpansionAssetRetriever,
) -> ExpansionOperationSequence | None:
    sequences = retriever.retrieve_sequences(
        ExpansionRetrievalQuery(
            families=(*families[:3], *paradigms[:3]),
            missing_techniques=terms,
        ),
        max_sequences=5,
        max_operations_per_sequence=3,
    )
    for sequence in sequences:
        if set(sequence.covered_terms) & set(terms):
            return sequence
    return None


def _representative_terms(occurrences: list[ValidationGapOccurrence]) -> tuple[str, ...]:
    counts = Counter(occurrence.technique for occurrence in occurrences)
    return tuple(
        term
        for term, _ in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0].lower()),
        )[:5]
    )


def _most_common(values: Iterable[str]) -> tuple[str, ...]:
    counts = Counter(value for value in values if value)
    return tuple(
        value
        for value, _ in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0].lower()),
        )[:5]
    )


def _cluster_sort_key(cluster: ExpansionGapCluster) -> tuple[int, int, int, str]:
    action_rank = {
        "candidate_reusable_operation": 0,
        "covered_by_existing_operation": 1,
        "defer_one_off": 2,
    }.get(cluster.recommended_action, 3)
    return (action_rank, -cluster.support, -len(cluster.competitions), cluster.cluster_id)


def _tokens(text: str) -> set[str]:
    return {
        _normalize_token(token)
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 3 and token not in _STOP_WORDS
    }


def _normalize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ing"):
        token = token[:-3]
    if len(token) > 4 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s"):
        token = token[:-1]
    return token


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


_STOP_WORDS = {
    "and",
    "for",
    "from",
    "into",
    "with",
    "using",
    "based",
    "before",
    "after",
    "stage",
    "model",
    "models",
    "data",
    "feature",
    "features",
    "training",
    "inference",
}
