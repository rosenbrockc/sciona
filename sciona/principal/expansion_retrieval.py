"""Deterministic retrieval over provider expansion/refinement assets."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from sciona.architect.handoff import CDGExport
from sciona.principal.expansion_assets import (
    ExpansionFamilyAsset,
    ExpansionOperationAsset,
    load_local_expansion_assets,
)


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_OPERATION_TYPE_COST = {
    "diagnostic": 0.10,
    "validation": 0.15,
    "precondition": 0.20,
    "insert": 0.35,
    "replace": 0.60,
    "rewrite": 0.70,
}


@dataclass(frozen=True)
class ExpansionRetrievalQuery:
    """Query for expansion/refinement operations near a base CDG."""

    families: tuple[str, ...] = ()
    missing_techniques: tuple[str, ...] = ()
    stage_names: tuple[str, ...] = ()
    input_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()
    runtime_keys: tuple[str, ...] = ()
    intermediate_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExpansionOperationMatch:
    """One scored operation candidate from an expansion asset."""

    asset_family: str
    asset_id: str
    rule_name: str
    operation_id: str
    operation_type: str
    applies_to: str
    score: float
    intrusion_cost: float
    covered_terms: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    prerequisite_operations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExpansionOperationSequence:
    """Ranked operation sequence for adapting a base CDG."""

    asset_family: str
    asset_id: str
    operations: tuple[ExpansionOperationMatch, ...]
    score: float
    intrusion_cost: float
    covered_terms: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class _IndexedOperation:
    asset: ExpansionFamilyAsset
    operation: ExpansionOperationAsset
    tokens: frozenset[str]
    phrases: tuple[str, ...]


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_normalize_token(token) for token in _TOKEN_RE.findall(str(text or "").lower()))


def _normalize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ing"):
        token = token[:-3]
    if len(token) > 4 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s"):
        token = token[:-1]
    return token


def _token_list(values: Iterable[str]) -> frozenset[str]:
    return frozenset(
        _normalize_token(token)
        for value in values
        for token in _TOKEN_RE.findall(str(value or "").lower())
    )


def _phrase_tokens(value: str) -> frozenset[str]:
    return _tokens(value)


def _operation_text(asset: ExpansionFamilyAsset, op: ExpansionOperationAsset) -> str:
    trigger = op.trigger
    parts = [
        asset.family,
        *asset.family_aliases,
        asset.domain,
        asset.name,
        asset.summary,
        op.rule_name,
        op.operation_id,
        op.operation_type,
        op.applies_to,
        op.name,
        op.intent,
        op.dejargonized_summary,
        op.runtime_diagnostic,
        op.rewrite.before_summary,
        op.rewrite.after_summary,
        op.rewrite.information_flow_effect,
        trigger.metric_name,
        *trigger.required_runtime_keys,
        *trigger.required_any_runtime_keys,
        *trigger.required_runtime_namespaces,
        *trigger.required_any_runtime_namespaces,
        *trigger.required_intermediate_keys,
        *trigger.required_any_intermediate_keys,
        *trigger.required_primitives,
        *trigger.required_any_primitives,
        *op.uncertainty_notes,
        *(action.value for action in op.action_classes),
    ]
    for requirement in trigger.required_boundary_requirements:
        parts.extend(
            [
                requirement.boundary_kind,
                requirement.port_name,
                *requirement.matched_primitives,
                requirement.data_kind,
                requirement.loss_class,
                *requirement.notes,
            ]
        )
    for source, target in trigger.required_adjacencies:
        parts.extend([source, target])
    return " ".join(str(part or "") for part in parts)


def _query_from_cdg(cdg: CDGExport | None, query: ExpansionRetrievalQuery) -> ExpansionRetrievalQuery:
    if cdg is None:
        return query

    stage_names = list(query.stage_names)
    input_names = list(query.input_names)
    output_names = list(query.output_names)
    families = list(query.families)

    for node in cdg.nodes:
        stage_names.extend([node.node_id, node.name, node.description, node.concept_type.value])
        if node.matched_primitive:
            stage_names.append(node.matched_primitive)
        input_names.extend(port.name for port in node.inputs)
        output_names.extend(port.name for port in node.outputs)
        if node.concept_type.value:
            families.append(node.concept_type.value)

    metadata = cdg.metadata or {}
    for key in ("family", "paradigm", "source_family", "skeleton_family"):
        value = metadata.get(key)
        if value:
            families.append(str(value))

    return ExpansionRetrievalQuery(
        families=tuple(families),
        missing_techniques=query.missing_techniques,
        stage_names=tuple(stage_names),
        input_names=tuple(input_names),
        output_names=tuple(output_names),
        runtime_keys=query.runtime_keys,
        intermediate_keys=query.intermediate_keys,
    )


def _jaccard(query_tokens: frozenset[str], candidate_tokens: frozenset[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / max(1, len(query_tokens | candidate_tokens))


def _phrase_coverage(phrases: tuple[str, ...], candidate_tokens: frozenset[str]) -> tuple[str, ...]:
    covered: list[str] = []
    for phrase in phrases:
        phrase_token_set = _phrase_tokens(phrase)
        if not phrase_token_set:
            continue
        overlap = len(phrase_token_set & candidate_tokens) / len(phrase_token_set)
        if overlap >= 0.60:
            covered.append(phrase)
    return tuple(covered)


def _cost(operation_type: str) -> float:
    return _OPERATION_TYPE_COST.get(str(operation_type or "").lower(), 0.40)


class ExpansionAssetRetriever:
    """Rank provider expansion operations and short operation sequences."""

    def __init__(self, assets: Iterable[ExpansionFamilyAsset] | None = None) -> None:
        self._assets = tuple(assets) if assets is not None else load_local_expansion_assets()
        self._indexed = tuple(
            _IndexedOperation(
                asset=asset,
                operation=operation,
                tokens=_tokens(_operation_text(asset, operation)),
                phrases=(
                    operation.name,
                    operation.intent,
                    operation.dejargonized_summary,
                    operation.applies_to,
                    operation.trigger.metric_name,
                ),
            )
            for asset in self._assets
            for operation in asset.operations
        )

    def retrieve_operations(
        self,
        query: ExpansionRetrievalQuery,
        *,
        cdg: CDGExport | None = None,
        min_score: float = 0.10,
        max_results: int = 20,
    ) -> list[ExpansionOperationMatch]:
        """Return ranked individual operation candidates."""
        enriched_query = _query_from_cdg(cdg, query)
        family_terms = _token_list(enriched_query.families)
        technique_phrases = tuple(enriched_query.missing_techniques)
        technique_terms = _token_list(technique_phrases)
        stage_terms = _token_list(enriched_query.stage_names)
        io_terms = _token_list(
            [
                *enriched_query.input_names,
                *enriched_query.output_names,
                *enriched_query.runtime_keys,
                *enriched_query.intermediate_keys,
            ]
        )

        matches: list[ExpansionOperationMatch] = []
        requested_families = {str(f).lower() for f in enriched_query.families}
        for item in self._indexed:
            asset = item.asset
            op = item.operation
            family_names = {asset.family.lower(), *(alias.lower() for alias in asset.family_aliases)}
            family_exact = bool(requested_families & family_names)
            family_score = 1.0 if family_exact else _jaccard(family_terms, item.tokens)
            technique_score = _jaccard(technique_terms, item.tokens)
            stage_score = _jaccard(stage_terms, item.tokens)
            io_score = _jaccard(io_terms, item.tokens)
            covered = _phrase_coverage(technique_phrases, item.tokens)
            if covered:
                technique_score = max(technique_score, min(1.0, len(covered) / max(1, len(technique_phrases))))

            score = (
                0.35 * family_score
                + 0.35 * technique_score
                + 0.18 * stage_score
                + 0.12 * io_score
            )
            if not requested_families and not technique_terms and not stage_terms and not io_terms:
                score = 0.0
            if score < min_score:
                continue

            reasons: list[str] = []
            if family_exact:
                reasons.append(f"family:{asset.family}")
            if covered:
                reasons.append("technique:" + ", ".join(covered))
            if stage_score > 0:
                reasons.append("stage_overlap")
            if io_score > 0:
                reasons.append("io_or_context_overlap")

            matches.append(
                ExpansionOperationMatch(
                    asset_family=asset.family,
                    asset_id=asset.asset_id,
                    rule_name=op.rule_name,
                    operation_id=op.operation_id,
                    operation_type=op.operation_type,
                    applies_to=op.applies_to,
                    score=round(min(1.0, score), 6),
                    intrusion_cost=_cost(op.operation_type),
                    covered_terms=covered,
                    reasons=tuple(reasons),
                    prerequisite_operations=tuple(op.prerequisite_operations),
                )
            )

        matches.sort(
            key=lambda match: (
                -match.score,
                match.intrusion_cost,
                match.asset_family,
                match.rule_name,
            )
        )
        return matches[:max_results]

    def retrieve_sequences(
        self,
        query: ExpansionRetrievalQuery,
        *,
        cdg: CDGExport | None = None,
        min_operation_score: float = 0.10,
        max_sequences: int = 5,
        max_operations_per_sequence: int = 4,
    ) -> list[ExpansionOperationSequence]:
        """Return ranked operation sequences grouped by expansion family."""
        pre_group_limit = max(
            len(self._indexed),
            max_sequences * max(4, max_operations_per_sequence * 4),
        )
        operations = self.retrieve_operations(
            query,
            cdg=cdg,
            min_score=min_operation_score,
            max_results=pre_group_limit,
        )
        by_family: dict[str, list[ExpansionOperationMatch]] = {}
        for operation in operations:
            by_family.setdefault(operation.asset_family, []).append(operation)

        sequences: list[ExpansionOperationSequence] = []
        for family, candidates in by_family.items():
            selected = self._select_sequence(candidates, max_operations_per_sequence)
            if not selected:
                continue
            covered_terms = tuple(
                dict.fromkeys(term for operation in selected for term in operation.covered_terms)
            )
            intrusion = round(sum(operation.intrusion_cost for operation in selected), 6)
            average_score = sum(operation.score for operation in selected) / len(selected)
            coverage_bonus = min(0.20, 0.05 * len(covered_terms))
            score = round(min(1.0, average_score + coverage_bonus - 0.08 * intrusion), 6)
            asset_id = selected[0].asset_id
            reasons = tuple(
                dict.fromkeys(reason for operation in selected for reason in operation.reasons)
            )
            sequences.append(
                ExpansionOperationSequence(
                    asset_family=family,
                    asset_id=asset_id,
                    operations=tuple(selected),
                    score=score,
                    intrusion_cost=intrusion,
                    covered_terms=covered_terms,
                    reasons=reasons,
                )
            )

        sequences.sort(
            key=lambda sequence: (
                -sequence.score,
                sequence.intrusion_cost,
                sequence.asset_family,
                tuple(operation.rule_name for operation in sequence.operations),
            )
        )
        return sequences[:max_sequences]

    @staticmethod
    def _select_sequence(
        candidates: list[ExpansionOperationMatch],
        max_operations: int,
    ) -> list[ExpansionOperationMatch]:
        by_rule = {candidate.rule_name: candidate for candidate in candidates}
        selected: list[ExpansionOperationMatch] = []
        covered_terms: set[str] = set()

        for candidate in sorted(
            candidates,
            key=lambda item: (-item.score, item.intrusion_cost, item.rule_name),
        ):
            if len(selected) >= max_operations:
                break
            new_terms = set(candidate.covered_terms) - covered_terms
            if selected and not new_terms and candidate.score < selected[0].score * 0.85:
                continue
            for prerequisite in candidate.prerequisite_operations:
                prereq = by_rule.get(prerequisite)
                if prereq is not None and prereq not in selected and len(selected) < max_operations:
                    selected.append(prereq)
                    covered_terms.update(prereq.covered_terms)
            if candidate not in selected and len(selected) < max_operations:
                selected.append(candidate)
                covered_terms.update(candidate.covered_terms)

        return selected


def retrieve_expansion_sequences(
    query: ExpansionRetrievalQuery,
    *,
    cdg: CDGExport | None = None,
    max_sequences: int = 5,
    max_operations_per_sequence: int = 4,
) -> list[ExpansionOperationSequence]:
    """Convenience wrapper using the locally configured expansion assets."""
    return ExpansionAssetRetriever().retrieve_sequences(
        query,
        cdg=cdg,
        max_sequences=max_sequences,
        max_operations_per_sequence=max_operations_per_sequence,
    )
