"""Side-effect-free symbolic physics retrieval and ranking.

The ranker consumes rows that have already been fetched from catalog views or
artifact-document RPCs. It never opens a database connection; callers own IO and
pass ordinary dictionaries into this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal


RawTrustPolicy = Literal["prefer_reviewed", "reviewed_only", "allow_raw"]

_BLOCKED_STATUSES = {"blocked", "failed", "parse_failed"}
_REVIEWED_STATUSES = {"human_reviewed", "automated_pass", "source_verified"}
_HUMAN_REVIEWED_STATUSES = {"human_reviewed", "published", "approved"}
_NEEDS_HUMAN_STATUSES = {"needs_human"}
_PUBLISHED_STATUSES = {"published", "approved"}
_RAW_STATUSES = {"", "raw_imported", "unreviewed", "parsed"}
_VERIFIED_RELATIONSHIP_KINDS = {
    "same_math_topology_as",
    "physical_grounding_of",
    "derives_from",
    "limit_case_of",
    "approximation_of",
    "uses_constant",
    "uses_data_artifact",
    "mechanism_analogue_of",
    "algebraic_rearrangement_of",
}
_DATA_ARTIFACT_REFERENCE_KEYS = (
    "data_artifact_dependencies",
    "data_artifacts",
    "data_artifact_ids",
    "artifact_dependencies",
    "future_data_artifact",
    "future_data_artifacts",
    "data_artifact_seed",
    "data_artifact_seeds",
)
_NESTED_REFERENCE_PAYLOAD_KEYS = ("source_payload",)


@dataclass(frozen=True)
class SymbolicValidityBound:
    """A local model for one validity/regime row."""

    variable_name: str = ""
    regime_label: str = ""
    validity_statement: str = ""
    lower_value: float | None = None
    upper_value: float | None = None
    review_status: str = ""

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "SymbolicValidityBound":
        return cls(
            variable_name=_text(row, "variable_name", "symbol_name", "name"),
            regime_label=_text(row, "regime_label", "label"),
            validity_statement=_text(row, "validity_statement", "statement"),
            lower_value=_optional_float(row.get("lower_value")),
            upper_value=_optional_float(row.get("upper_value")),
            review_status=_text(row, "review_status", "status"),
        )

    @property
    def constrained(self) -> bool:
        return bool(
            self.validity_statement
            or self.regime_label
            or self.lower_value is not None
            or self.upper_value is not None
        )

    @property
    def reviewed(self) -> bool:
        return self.review_status in _REVIEWED_STATUSES


@dataclass(frozen=True)
class SymbolicRelationship:
    """A local model for one artifact/expression relationship row."""

    relationship_kind: str
    relationship_label: str = ""
    confidence: float = 0.0
    verified: bool = False
    source_kind: str = ""

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "SymbolicRelationship":
        return cls(
            relationship_kind=_text(row, "relationship_kind", "kind"),
            relationship_label=_text(row, "relationship_label", "label"),
            confidence=_bounded_float(row.get("confidence"), default=0.0),
            verified=_bool(row.get("verified")),
            source_kind=_text(row, "source_kind", "relationship_source_kind"),
        )


@dataclass(frozen=True)
class RawCandidateExternalKnowledgeSuggestion:
    """Side-effect-free review suggestion for an untrusted symbolic candidate."""

    candidate_key: str
    trust_status: str
    reason: str
    raw_formula: str = ""
    suggested_relationship_kinds: tuple[str, ...] = ()
    suggested_reference_queries: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_key": self.candidate_key,
            "trust_status": self.trust_status,
            "reason": self.reason,
            "raw_formula": self.raw_formula,
            "suggested_relationship_kinds": list(self.suggested_relationship_kinds),
            "suggested_reference_queries": list(self.suggested_reference_queries),
        }


@dataclass(frozen=True)
class SymbolicArtifactCandidate:
    """Candidate expression/artifact used by the Phase 6 symbolic ranker."""

    artifact_id: str = ""
    version_id: str = ""
    expression_id: str = ""
    fqdn: str = ""
    artifact_kind: str = ""
    expression_kind: str = ""
    raw_formula: str = ""
    topology_hash: str = ""
    dimensional_hash: str = ""
    dim_signatures: tuple[str, ...] = ()
    mechanism_tags: tuple[str, ...] = ()
    behavioral_archetypes: tuple[str, ...] = ()
    validity_bounds: tuple[SymbolicValidityBound, ...] = ()
    relationships: tuple[SymbolicRelationship, ...] = ()
    source_system: str = ""
    source_kind: str = ""
    source_domains: tuple[str, ...] = ()
    known_analogues: tuple[str, ...] = ()
    data_artifact_dependencies: tuple[str, ...] = ()
    review_status: str = ""
    validation_status: str = ""
    publish_status: str = ""
    candidate_status: str = ""
    trust_readiness: str = ""
    is_publishable: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_catalog_row(
        cls,
        row: Mapping[str, Any],
    ) -> "SymbolicArtifactCandidate":
        """Build a candidate from one flattened ``catalog_symbolic_artifacts`` row."""

        return cls._from_parts(
            row=row,
            expression=row,
            bounds=_sequence_of_mappings(
                row.get("validity_bounds") or row.get("artifact_validity_bounds")
            ),
            relationships=_sequence_of_mappings(
                row.get("relationships") or row.get("artifact_relationships")
            ),
            variables=_sequence_of_mappings(
                row.get("symbolic_variables")
                or row.get("variables")
                or row.get("artifact_symbolic_variables")
            ),
        )

    @classmethod
    def from_artifact_document(
        cls,
        document: Mapping[str, Any],
    ) -> tuple["SymbolicArtifactCandidate", ...]:
        """Build candidates from a nested ``get_artifact_document`` result."""

        artifact = _mapping(document.get("artifact"))
        expressions = _sequence_of_mappings(
            document.get("symbolic_expressions")
            or document.get("expressions")
            or document.get("artifact_symbolic_expressions")
        )
        if not expressions:
            expression = _mapping(document.get("symbolic_expression"))
            if expression:
                expressions = (expression,)
        bounds = _sequence_of_mappings(
            document.get("validity_bounds")
            or document.get("artifact_validity_bounds")
        )
        relationships = _sequence_of_mappings(
            document.get("relationships")
            or document.get("artifact_relationships")
        )
        variables = _sequence_of_mappings(
            document.get("symbolic_variables")
            or document.get("variables")
            or document.get("artifact_symbolic_variables")
        )
        if not expressions and artifact:
            expressions = (artifact,)
        return tuple(
            cls._from_parts(
                row=artifact,
                expression=expression,
                bounds=bounds,
                relationships=relationships,
                variables=variables,
                document=document,
            )
            for expression in expressions
        )

    @classmethod
    def _from_parts(
        cls,
        *,
        row: Mapping[str, Any],
        expression: Mapping[str, Any],
        bounds: Sequence[Mapping[str, Any]] = (),
        relationships: Sequence[Mapping[str, Any]] = (),
        variables: Sequence[Mapping[str, Any]] = (),
        document: Mapping[str, Any] | None = None,
    ) -> "SymbolicArtifactCandidate":
        merged = {**dict(row), **dict(expression)}
        expression_id = _text(merged, "expression_id")
        scoped_bounds = _scoped_rows(bounds, expression_id)
        scoped_relationships = _scoped_rows(relationships, expression_id, relationship=True)
        dim_signatures = _dim_signatures(merged, variables)
        return cls(
            artifact_id=_text(merged, "artifact_id"),
            version_id=_text(merged, "version_id"),
            expression_id=expression_id,
            fqdn=_text(merged, "fqdn", "artifact_fqdn"),
            artifact_kind=_text(merged, "artifact_kind"),
            expression_kind=_text(merged, "expression_kind"),
            raw_formula=_text(merged, "raw_formula", "formula"),
            topology_hash=_text(merged, "topology_hash"),
            dimensional_hash=_text(merged, "dimensional_hash", "dim_hash"),
            dim_signatures=dim_signatures,
            mechanism_tags=_strings(
                merged.get("mechanism_tags")
                or merged.get("mechanisms")
                or merged.get("domain_tags")
            ),
            behavioral_archetypes=_strings(
                merged.get("behavioral_archetypes")
                or merged.get("archetypes")
            ),
            validity_bounds=tuple(
                SymbolicValidityBound.from_mapping(bound) for bound in scoped_bounds
            ),
            relationships=tuple(
                SymbolicRelationship.from_mapping(relationship)
                for relationship in scoped_relationships
            ),
            source_system=_text(merged, "source_system"),
            source_kind=_text(merged, "source_kind"),
            source_domains=_strings(
                merged.get("source_domains")
                or merged.get("source_domain")
                or (document or {}).get("source_domains")
                or (document or {}).get("source_domain")
            ),
            known_analogues=_known_analogues(merged, scoped_relationships, document),
            data_artifact_dependencies=_row_references(
                merged,
                document,
                keys=_DATA_ARTIFACT_REFERENCE_KEYS,
            ),
            review_status=_text(merged, "review_status"),
            validation_status=_text(merged, "validation_status"),
            publish_status=_text(merged, "publish_status", "status"),
            candidate_status=_text(merged, "candidate_status"),
            trust_readiness=_text(merged, "trust_readiness", "overall_verdict"),
            is_publishable=_bool(merged.get("is_publishable")),
            extra={"row": dict(merged), "document": dict(document or {})},
        )

    @property
    def published(self) -> bool:
        return (
            self.is_publishable
            or self.publish_status in _PUBLISHED_STATUSES
            or self.candidate_status == "published"
        )

    @property
    def reviewed(self) -> bool:
        return (
            self.published
            or self.review_status in _REVIEWED_STATUSES
            or self.candidate_status in _REVIEWED_STATUSES
            or self.trust_readiness in {"trusted", "reviewed", "production"}
        )

    @property
    def human_reviewed(self) -> bool:
        return (
            self.published
            or self.review_status in _HUMAN_REVIEWED_STATUSES
            or self.candidate_status in _HUMAN_REVIEWED_STATUSES
        )

    @property
    def needs_human(self) -> bool:
        return (
            self.review_status in _NEEDS_HUMAN_STATUSES
            or self.candidate_status in _NEEDS_HUMAN_STATUSES
            or self.trust_readiness in _NEEDS_HUMAN_STATUSES
        )

    @property
    def blocked(self) -> bool:
        statuses = {
            self.review_status,
            self.validation_status,
            self.publish_status,
            self.candidate_status,
        }
        return bool(statuses & _BLOCKED_STATUSES)

    @property
    def trust_status(self) -> str:
        if self.blocked:
            return "blocked"
        if self.human_reviewed:
            return "human_reviewed"
        if self.needs_human or self.raw_like:
            return "needs_human"
        if self.reviewed:
            return "automated_pass"
        return "unreviewed"

    @property
    def raw_like(self) -> bool:
        return not self.reviewed and (
            self.review_status in _RAW_STATUSES
            or self.candidate_status in _RAW_STATUSES
            or not (self.review_status or self.candidate_status)
        )


@dataclass(frozen=True)
class SymbolicRetrievalQuery:
    """Feature query for symbolic retrieval ranking."""

    topology_hashes: tuple[str, ...] = ()
    dimensional_hashes: tuple[str, ...] = ()
    dim_signatures: tuple[str, ...] = ()
    mechanism_tags: tuple[str, ...] = ()
    behavioral_archetypes: tuple[str, ...] = ()
    relationship_kinds: tuple[str, ...] = ()
    source_systems: tuple[str, ...] = ()
    source_kinds: tuple[str, ...] = ()
    source_domains: tuple[str, ...] = ()
    known_analogues: tuple[str, ...] = ()
    data_artifact_dependencies: tuple[str, ...] = ()
    require_validity_bounds: bool = False
    require_reviewed_bounds: bool = False
    require_data_artifact_dependencies: bool = False
    raw_trust_policy: RawTrustPolicy = "prefer_reviewed"

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "SymbolicRetrievalQuery":
        return cls(
            topology_hashes=_strings(row.get("topology_hashes") or row.get("topology_hash")),
            dimensional_hashes=_strings(
                row.get("dimensional_hashes")
                or row.get("dimensional_hash")
                or row.get("dim_hash")
            ),
            dim_signatures=_strings(row.get("dim_signatures") or row.get("dim_signature")),
            mechanism_tags=_strings(row.get("mechanism_tags")),
            behavioral_archetypes=_strings(row.get("behavioral_archetypes")),
            relationship_kinds=_strings(row.get("relationship_kinds")),
            source_systems=_strings(row.get("source_systems") or row.get("source_system")),
            source_kinds=_strings(row.get("source_kinds") or row.get("source_kind")),
            source_domains=_strings(row.get("source_domains") or row.get("source_domain")),
            known_analogues=_strings(
                row.get("known_analogues")
                or row.get("known_analogue")
                or row.get("analogue_artifact_fqdns")
            ),
            data_artifact_dependencies=_row_references(
                row,
                keys=_DATA_ARTIFACT_REFERENCE_KEYS,
            ),
            require_validity_bounds=_bool(row.get("require_validity_bounds")),
            require_reviewed_bounds=_bool(row.get("require_reviewed_bounds")),
            require_data_artifact_dependencies=_bool(
                row.get("require_data_artifact_dependencies")
            ),
            raw_trust_policy=_raw_trust_policy(
                row.get("raw_trust_policy") or "prefer_reviewed"
            ),
        )


@dataclass(frozen=True)
class SymbolicRankingResult:
    """Ranker output with score components for explainability."""

    candidate: SymbolicArtifactCandidate
    score: float
    eligible: bool
    reasons: tuple[str, ...]
    components: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_key": _candidate_key(self.candidate),
            "score": self.score,
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "components": dict(self.components),
            "trust_status": self.candidate.trust_status,
        }


def rank_symbolic_candidates(
    query: SymbolicRetrievalQuery | Mapping[str, Any],
    candidates: Iterable[SymbolicArtifactCandidate | Mapping[str, Any]],
    *,
    limit: int | None = None,
) -> tuple[SymbolicRankingResult, ...]:
    """Rank local symbolic candidates without IO or global state."""

    request = (
        query
        if isinstance(query, SymbolicRetrievalQuery)
        else SymbolicRetrievalQuery.from_mapping(query)
    )
    scored = [
        score_symbolic_candidate(request, _candidate(candidate))
        for candidate in candidates
    ]
    scored.sort(
        key=lambda result: (
            result.eligible,
            result.score,
            result.candidate.published,
            result.candidate.reviewed,
            result.candidate.fqdn,
            result.candidate.expression_id,
        ),
        reverse=True,
    )
    if limit is not None:
        return tuple(scored[:limit])
    return tuple(scored)


def score_symbolic_candidate(
    query: SymbolicRetrievalQuery,
    candidate: SymbolicArtifactCandidate,
) -> SymbolicRankingResult:
    """Score one candidate and return component-level rationale."""

    components: dict[str, float] = {}
    reasons: list[str] = []
    eligible = True

    if candidate.blocked:
        eligible = False
        reasons.append("blocked_status")
    if candidate.needs_human:
        reasons.append("needs_human_review")
    if query.raw_trust_policy == "reviewed_only" and not candidate.reviewed:
        eligible = False
        reasons.append("raw_excluded_by_policy")

    if candidate.topology_hash and candidate.topology_hash in query.topology_hashes:
        components["topology_hash"] = 4.0
        reasons.append("topology_hash_match")

    if (
        candidate.dimensional_hash
        and candidate.dimensional_hash in query.dimensional_hashes
    ):
        components["dimensional_hash"] = 2.0
        reasons.append("dimensional_hash_match")

    dim_score = _overlap_score(
        query.dim_signatures,
        candidate.dim_signatures,
        weight=1.5,
    )
    if dim_score:
        components["dim_signatures"] = dim_score
        reasons.append("dim_signature_overlap")

    mechanism_score = _overlap_score(
        query.mechanism_tags,
        candidate.mechanism_tags,
        weight=1.2,
    )
    if mechanism_score:
        components["mechanism_tags"] = mechanism_score
        reasons.append("mechanism_overlap")

    archetype_score = _overlap_score(
        query.behavioral_archetypes,
        candidate.behavioral_archetypes,
        weight=1.0,
    )
    if archetype_score:
        components["behavioral_archetypes"] = archetype_score
        reasons.append("behavioral_archetype_overlap")

    relationship_kinds = tuple(
        relationship.relationship_kind for relationship in candidate.relationships
    )
    relationship_score = _overlap_score(
        query.relationship_kinds,
        relationship_kinds,
        weight=0.8,
    )
    verified_requested = {
        relationship.relationship_kind
        for relationship in candidate.relationships
        if relationship.verified
    }
    if relationship_score:
        components["relationship_kinds"] = relationship_score
        reasons.append("relationship_kind_overlap")
    if query.relationship_kinds and set(query.relationship_kinds) <= verified_requested:
        components["verified_relationships"] = 0.4
        reasons.append("requested_relationships_verified")

    analogue_score = _overlap_score(
        query.known_analogues,
        candidate.known_analogues,
        weight=0.7,
    )
    if analogue_score:
        components["known_analogues"] = analogue_score
        reasons.append("known_analogue_overlap")

    artifact_score = _overlap_score(
        query.data_artifact_dependencies,
        candidate.data_artifact_dependencies,
        weight=0.7,
    )
    if artifact_score:
        components["data_artifact_dependencies"] = artifact_score
        reasons.append("data_artifact_dependency_overlap")
    elif query.require_data_artifact_dependencies and (
        not candidate.data_artifact_dependencies
        or bool(query.data_artifact_dependencies)
    ):
        eligible = False
        reasons.append("missing_required_data_artifact_dependencies")

    provenance_component, provenance_reasons = _provenance_score(query, candidate)
    components.update(provenance_component)
    reasons.extend(provenance_reasons)

    validity_component, validity_reasons, validity_eligible = _validity_score(
        query,
        candidate,
    )
    components.update(validity_component)
    reasons.extend(validity_reasons)
    eligible = eligible and validity_eligible

    trust_component, trust_reasons = _trust_score(query, candidate)
    components.update(trust_component)
    reasons.extend(trust_reasons)

    score = round(sum(components.values()), 6)
    if not eligible:
        score = min(score, 0.0)
    return SymbolicRankingResult(
        candidate=candidate,
        score=score,
        eligible=eligible,
        reasons=tuple(reasons),
        components=components,
    )


def candidates_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[SymbolicArtifactCandidate, ...]:
    """Normalize flattened catalog rows and nested document rows."""

    candidates: list[SymbolicArtifactCandidate] = []
    for row in rows:
        if any(
            key in row
            for key in (
                "artifact",
                "symbolic_expressions",
                "artifact_symbolic_expressions",
                "symbolic_expression",
            )
        ):
            candidates.extend(SymbolicArtifactCandidate.from_artifact_document(row))
        else:
            candidates.append(SymbolicArtifactCandidate.from_catalog_row(row))
    return tuple(candidates)


def suggest_raw_candidate_external_knowledge(
    candidates: Iterable[SymbolicArtifactCandidate | Mapping[str, Any]],
) -> tuple[RawCandidateExternalKnowledgeSuggestion, ...]:
    """Suggest external-knowledge checks for raw or needs-human candidates.

    This helper only inspects already-fetched rows. Callers can render the
    suggestions in CLI reports or decide which source adapters to run.
    """

    suggestions: list[RawCandidateExternalKnowledgeSuggestion] = []
    for value in candidates:
        candidate = _candidate(value)
        if candidate.blocked or candidate.reviewed:
            continue
        if not (candidate.raw_like or candidate.needs_human):
            continue

        verified_kinds = {
            relationship.relationship_kind
            for relationship in candidate.relationships
            if relationship.verified
        }
        missing_kinds = tuple(
            kind
            for kind in ("physical_grounding_of", "derives_from", "uses_constant")
            if kind not in verified_kinds
        )
        reason = (
            "raw_candidate_needs_external_knowledge"
            if candidate.raw_like
            else "candidate_needs_human_external_knowledge"
        )
        suggestions.append(
            RawCandidateExternalKnowledgeSuggestion(
                candidate_key=_candidate_key(candidate),
                trust_status=candidate.trust_status,
                reason=reason,
                raw_formula=candidate.raw_formula,
                suggested_relationship_kinds=missing_kinds,
                suggested_reference_queries=_reference_queries(candidate),
            )
        )
    return tuple(suggestions)


def build_symbolic_retrieval_report(
    query: SymbolicRetrievalQuery | Mapping[str, Any],
    candidates: Iterable[SymbolicArtifactCandidate | Mapping[str, Any]],
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Return ranking and raw-candidate review suggestions without IO."""

    candidate_tuple = tuple(_candidate(candidate) for candidate in candidates)
    results = rank_symbolic_candidates(query, candidate_tuple, limit=limit)
    suggestions = suggest_raw_candidate_external_knowledge(candidate_tuple)
    return {
        "result_count": len(results),
        "results": [result.to_dict() for result in results],
        "raw_candidate_external_knowledge_suggestions": [
            suggestion.to_dict() for suggestion in suggestions
        ],
    }


def build_symbolic_synthesis_retrieval_report(
    query: SymbolicRetrievalQuery | Mapping[str, Any],
    candidates: Iterable[SymbolicArtifactCandidate | Mapping[str, Any]],
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe synthesis report from already-fetched candidate rows.

    The report is intentionally side-effect-free: it accepts catalog/document
    rows and performs no DB, file, or network IO. Executable candidates are
    published/reviewed, eligible, and have dimensional metadata that a compiler
    can check. Raw or needs-human rows are kept as external-knowledge
    suggestions with their scoring context.
    """

    request = (
        query
        if isinstance(query, SymbolicRetrievalQuery)
        else SymbolicRetrievalQuery.from_mapping(query)
    )
    candidate_tuple = tuple(_candidate(candidate) for candidate in candidates)
    results = rank_symbolic_candidates(request, candidate_tuple, limit=limit)

    executable: list[dict[str, Any]] = []
    external_suggestions: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for result in results:
        contract = _compiler_contract_guidance(request, result)
        payload = _synthesis_candidate_payload(result, contract)
        if _is_executable_candidate(result, contract):
            executable.append(payload)
        elif result.candidate.raw_like or result.candidate.needs_human:
            external_suggestions.append(
                {
                    **payload,
                    "suggestion": _external_knowledge_suggestion_payload(
                        result.candidate
                    ),
                }
            )
        else:
            blocked.append(payload)

    return {
        "report_kind": "symbolic_synthesis_retrieval",
        "result_count": len(results),
        "executable_candidate_count": len(executable),
        "external_knowledge_suggestion_count": len(external_suggestions),
        "executable_candidates": executable,
        "external_knowledge_suggestions": external_suggestions,
        "blocked_candidates": blocked,
        "compiler_contract": {
            "required_dimensional_checks": _query_dimensional_checks(request),
            "blocker_kinds": [
                "blocked_status",
                "not_published_or_reviewed",
                "missing_dimensional_metadata",
                "missing_required_validity_bounds",
                "missing_reviewed_validity_bounds",
                "missing_required_data_artifact_dependencies",
                "raw_excluded_by_policy",
            ],
        },
    }


def _validity_score(
    query: SymbolicRetrievalQuery,
    candidate: SymbolicArtifactCandidate,
) -> tuple[dict[str, float], list[str], bool]:
    components: dict[str, float] = {}
    reasons: list[str] = []
    bounds = [bound for bound in candidate.validity_bounds if bound.constrained]
    eligible = True
    if bounds:
        components["validity_bounds"] = 0.6
        reasons.append("has_validity_bounds")
    elif query.require_validity_bounds:
        eligible = False
        reasons.append("missing_required_validity_bounds")
    if bounds and all(bound.reviewed for bound in bounds):
        components["reviewed_validity_bounds"] = 0.4
        reasons.append("validity_bounds_reviewed")
    elif query.require_reviewed_bounds:
        eligible = False
        reasons.append("missing_reviewed_validity_bounds")
    return components, reasons, eligible


def _trust_score(
    query: SymbolicRetrievalQuery,
    candidate: SymbolicArtifactCandidate,
) -> tuple[dict[str, float], list[str]]:
    components: dict[str, float] = {}
    reasons: list[str] = []
    if candidate.published:
        components["published"] = 2.0
        reasons.append("published_or_publishable")
    elif candidate.human_reviewed:
        components["human_reviewed"] = 1.6
        reasons.append("human_reviewed")
    elif candidate.reviewed:
        components["reviewed"] = 1.2
        reasons.append("reviewed")
    if candidate.validation_status == "passed":
        components["validation_passed"] = 0.8
        reasons.append("validation_passed")
    if any(
        relationship.verified
        and relationship.relationship_kind in _VERIFIED_RELATIONSHIP_KINDS
        for relationship in candidate.relationships
    ):
        components["source_relationship_trust"] = 0.4
        reasons.append("verified_source_relationship")
    if query.raw_trust_policy == "prefer_reviewed" and candidate.raw_like:
        components["raw_penalty"] = -1.5
        reasons.append("raw_penalty")
    return components, reasons


def _provenance_score(
    query: SymbolicRetrievalQuery,
    candidate: SymbolicArtifactCandidate,
) -> tuple[dict[str, float], list[str]]:
    components: dict[str, float] = {}
    reasons: list[str] = []
    if candidate.source_system and candidate.source_system in query.source_systems:
        components["source_system"] = 0.5
        reasons.append("source_system_match")
    if candidate.source_kind and candidate.source_kind in query.source_kinds:
        components["source_kind"] = 0.3
        reasons.append("source_kind_match")
    source_domain_score = _overlap_score(
        query.source_domains,
        candidate.source_domains,
        weight=0.4,
    )
    if source_domain_score:
        components["source_domains"] = source_domain_score
        reasons.append("source_domain_overlap")
    if candidate.source_system and candidate.source_kind:
        components["provenance_present"] = 0.2
        reasons.append("provenance_present")
    return components, reasons


def _is_executable_candidate(
    result: SymbolicRankingResult,
    contract: Mapping[str, Any],
) -> bool:
    candidate = result.candidate
    return bool(
        result.eligible
        and (candidate.published or candidate.reviewed)
        and _dimensionally_usable(candidate)
        and not contract["blockers"]
    )


def _dimensionally_usable(candidate: SymbolicArtifactCandidate) -> bool:
    return bool(candidate.dimensional_hash or candidate.dim_signatures)


def _synthesis_candidate_payload(
    result: SymbolicRankingResult,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = result.candidate
    return {
        "candidate_key": _candidate_key(candidate),
        "artifact_id": candidate.artifact_id,
        "version_id": candidate.version_id,
        "expression_id": candidate.expression_id,
        "fqdn": candidate.fqdn,
        "raw_formula": candidate.raw_formula,
        "score": result.score,
        "eligible": result.eligible,
        "trust_status": candidate.trust_status,
        "score_components": dict(result.components),
        "score_reasons": list(result.reasons),
        "topology": {
            "topology_hash": candidate.topology_hash,
            "behavioral_archetypes": list(candidate.behavioral_archetypes),
        },
        "mechanism": {"mechanism_tags": list(candidate.mechanism_tags)},
        "known_analogues": list(candidate.known_analogues),
        "data_artifact_dependencies": list(candidate.data_artifact_dependencies),
        "dimensions": {
            "dimensional_hash": candidate.dimensional_hash,
            "dim_signatures": list(candidate.dim_signatures),
            "dimensionally_usable": _dimensionally_usable(candidate),
        },
        "validity_bounds": [
            {
                "variable_name": bound.variable_name,
                "regime_label": bound.regime_label,
                "validity_statement": bound.validity_statement,
                "lower_value": bound.lower_value,
                "upper_value": bound.upper_value,
                "review_status": bound.review_status,
            }
            for bound in candidate.validity_bounds
        ],
        "provenance": {
            "source_system": candidate.source_system,
            "source_kind": candidate.source_kind,
            "source_domains": list(candidate.source_domains),
            "review_status": candidate.review_status,
            "validation_status": candidate.validation_status,
            "publish_status": candidate.publish_status,
        },
        "relationship_edges": [
            {
                "relationship_kind": relationship.relationship_kind,
                "relationship_label": relationship.relationship_label,
                "confidence": relationship.confidence,
                "verified": relationship.verified,
                "source_kind": relationship.source_kind,
            }
            for relationship in candidate.relationships
        ],
        "compiler_contract": dict(contract),
    }


def _external_knowledge_suggestion_payload(
    candidate: SymbolicArtifactCandidate,
) -> dict[str, Any]:
    suggestions = suggest_raw_candidate_external_knowledge((candidate,))
    if suggestions:
        return suggestions[0].to_dict()
    return {
        "candidate_key": _candidate_key(candidate),
        "trust_status": candidate.trust_status,
        "reason": "candidate_needs_human_external_knowledge",
        "raw_formula": candidate.raw_formula,
        "suggested_relationship_kinds": [],
        "suggested_reference_queries": list(_reference_queries(candidate)),
    }


def _compiler_contract_guidance(
    query: SymbolicRetrievalQuery,
    result: SymbolicRankingResult,
) -> dict[str, Any]:
    candidate = result.candidate
    blockers = list(_compiler_blockers(query, result))
    return {
        "required_dimensional_checks": _candidate_dimensional_checks(query, candidate),
        "blockers": blockers,
        "can_compile": not blockers,
        "requires_human_review": candidate.needs_human or candidate.raw_like,
    }


def _compiler_blockers(
    query: SymbolicRetrievalQuery,
    result: SymbolicRankingResult,
) -> tuple[str, ...]:
    candidate = result.candidate
    blockers = []
    if candidate.blocked:
        blockers.append("blocked_status")
    if not (candidate.published or candidate.reviewed):
        blockers.append("not_published_or_reviewed")
    if not _dimensionally_usable(candidate):
        blockers.append("missing_dimensional_metadata")
    for reason in result.reasons:
        if reason in {
            "missing_required_validity_bounds",
            "missing_reviewed_validity_bounds",
            "missing_required_data_artifact_dependencies",
            "raw_excluded_by_policy",
        }:
            blockers.append(reason)
    if query.raw_trust_policy == "reviewed_only" and candidate.raw_like:
        blockers.append("raw_excluded_by_policy")
    return _unique(blockers)


def _query_dimensional_checks(query: SymbolicRetrievalQuery) -> list[str]:
    checks = ["verify_candidate_formula_dimensions_against_problem"]
    if query.dimensional_hashes:
        checks.append("match_requested_dimensional_hash")
    if query.dim_signatures:
        checks.append("match_requested_input_output_dim_signatures")
    return checks


def _candidate_dimensional_checks(
    query: SymbolicRetrievalQuery,
    candidate: SymbolicArtifactCandidate,
) -> list[str]:
    checks = _query_dimensional_checks(query)
    if candidate.dimensional_hash:
        checks.append("verify_candidate_dimensional_hash")
    if candidate.dim_signatures:
        checks.append("verify_candidate_symbol_dim_signatures")
    return checks


def _candidate(
    value: SymbolicArtifactCandidate | Mapping[str, Any],
) -> SymbolicArtifactCandidate:
    if isinstance(value, SymbolicArtifactCandidate):
        return value
    return SymbolicArtifactCandidate.from_catalog_row(value)


def _candidate_key(candidate: SymbolicArtifactCandidate) -> str:
    return (
        candidate.expression_id
        or candidate.version_id
        or candidate.artifact_id
        or candidate.fqdn
        or "<candidate>"
    )


def _reference_queries(candidate: SymbolicArtifactCandidate) -> tuple[str, ...]:
    seeds = (
        candidate.raw_formula,
        candidate.fqdn.replace(".", " "),
        *candidate.mechanism_tags,
        *candidate.behavioral_archetypes,
        *candidate.source_domains,
        *candidate.known_analogues,
        *candidate.data_artifact_dependencies,
    )
    queries = []
    for seed in seeds:
        text = seed.strip()
        if text:
            queries.append(text)
    return _unique(queries)


def _overlap_score(
    requested: Sequence[str],
    available: Sequence[str],
    *,
    weight: float,
) -> float:
    requested_set = {item for item in requested if item}
    available_set = {item for item in available if item}
    if not requested_set or not available_set:
        return 0.0
    overlap = requested_set & available_set
    if not overlap:
        return 0.0
    return weight * (len(overlap) / len(requested_set))


def _dim_signatures(
    row: Mapping[str, Any],
    variables: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    values = list(
        _strings(
            row.get("dim_signatures")
            or row.get("dim_signature")
            or row.get("io_dim_signatures")
        )
    )
    for variable in variables:
        expression_id = _text(row, "expression_id")
        if expression_id and _text(variable, "expression_id") not in {"", expression_id}:
            continue
        values.extend(_strings(variable.get("dim_signature")))
    return _unique(values)


def _scoped_rows(
    rows: Sequence[Mapping[str, Any]],
    expression_id: str,
    *,
    relationship: bool = False,
) -> tuple[Mapping[str, Any], ...]:
    if not expression_id:
        return tuple(rows)
    scoped = []
    for row in rows:
        ids = {_text(row, "expression_id")}
        if relationship:
            ids.add(_text(row, "source_expression_id"))
            ids.add(_text(row, "target_expression_id"))
        if not any(ids) or expression_id in ids:
            scoped.append(row)
    return tuple(scoped)


def _known_analogues(
    row: Mapping[str, Any],
    relationships: Sequence[Mapping[str, Any]],
    document: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    values = list(
        _row_references(
            row,
            document,
            keys=(
                "known_analogues",
                "known_analogue",
                "analogue_artifact_fqdns",
                "analogues",
            ),
        )
    )
    for relationship in relationships:
        if _text(relationship, "relationship_kind", "kind") != "mechanism_analogue_of":
            continue
        values.extend(
            _row_reference_values(
                relationship,
                keys=(
                    "target_artifact_fqdn",
                    "target_fqdn",
                    "target_artifact_id",
                    "relationship_label",
                    "label",
                ),
            )
        )
    return _unique(values)


def _row_references(
    *rows: Mapping[str, Any] | None,
    keys: Sequence[str],
) -> tuple[str, ...]:
    values: list[str] = []
    for row in rows:
        if not row:
            continue
        values.extend(_row_reference_values(row, keys=keys))
    return _unique(values)


def _row_reference_values(
    row: Mapping[str, Any],
    *,
    keys: Sequence[str],
) -> tuple[str, ...]:
    values: list[str] = []
    for key in keys:
        values.extend(_reference_values(row.get(key)))
    for payload_key in _NESTED_REFERENCE_PAYLOAD_KEYS:
        payload = row.get(payload_key)
        if isinstance(payload, Mapping):
            for key in keys:
                values.extend(_reference_values(payload.get(key)))
    return _unique(values)


def _reference_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        text = _text(
            value,
            "artifact_id",
            "artifact_key",
            "artifact_fqdn",
            "fqdn",
            "source_candidate_id",
            "reference_id",
            "id",
            "key",
            "name",
        )
        return (text,) if text else ()
    if isinstance(value, str):
        return _strings(value)
    if isinstance(value, Iterable):
        references: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                references.extend(_reference_values(item))
            elif item is not None:
                text = str(item).strip()
                if text:
                    references.append(text)
        return _unique(references)
    if value is None:
        return ()
    return (str(value).strip(),) if str(value).strip() else ()


def _sequence_of_mappings(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return tuple(part for part in parts if part)
    if isinstance(value, Iterable):
        return _unique(str(item).strip() for item in value if str(item).strip())
    return ()


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


def _text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str):
            return value.strip()
        if value is not None and not isinstance(value, (Mapping, Sequence)):
            return str(value).strip()
    return ""


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}
    return bool(value)


def _bounded_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _raw_trust_policy(value: Any) -> RawTrustPolicy:
    text = str(value or "prefer_reviewed")
    if text not in {"prefer_reviewed", "reviewed_only", "allow_raw"}:
        raise ValueError("raw_trust_policy must be prefer_reviewed, reviewed_only, or allow_raw")
    return text  # type: ignore[return-value]
