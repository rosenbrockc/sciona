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
    def blocked(self) -> bool:
        statuses = {
            self.review_status,
            self.validation_status,
            self.publish_status,
            self.candidate_status,
        }
        return bool(statuses & _BLOCKED_STATUSES)

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
    require_validity_bounds: bool = False
    require_reviewed_bounds: bool = False
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
            require_validity_bounds=_bool(row.get("require_validity_bounds")),
            require_reviewed_bounds=_bool(row.get("require_reviewed_bounds")),
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


def _candidate(
    value: SymbolicArtifactCandidate | Mapping[str, Any],
) -> SymbolicArtifactCandidate:
    if isinstance(value, SymbolicArtifactCandidate):
        return value
    return SymbolicArtifactCandidate.from_catalog_row(value)


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
