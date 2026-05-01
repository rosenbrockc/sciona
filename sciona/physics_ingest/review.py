"""Phase 5 audit and review contracts for physics ingestion.

The review layer consumes already-staged Wave 0 rows or plain dictionaries and
computes publishability gates without reaching into Supabase. It is deliberately
side-effect free so loaders, CLIs, and CI checks can share the same contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


REVIEW_STATUSES: tuple[str, ...] = (
    "unreviewed",
    "automated_pass",
    "needs_human",
    "human_reviewed",
    "blocked",
)

WORKFLOW_STATUSES: tuple[str, ...] = (
    "raw_imported",
    "parsed",
    "dimension_resolved",
    "symbolically_validated",
    "source_verified",
    "human_reviewed",
    "published",
)

_STATUS_RANK = {status: index for index, status in enumerate(WORKFLOW_STATUSES)}
_PARSED_PARSE_STATUSES = {"parsed", "normalized"}
_PASS_VALUES = {"pass", "passed", "ok", "success", "succeeded", "true", True}
_REVIEWED_BOUND_STATUSES = {"automated_pass", "human_reviewed"}
_DEPENDENCY_KINDS = {"uses_constant", "uses_data_artifact"}
_BLOCKED_STATUSES = {"blocked", "failed", "parse_failed"}
_UNKNOWN_DIM_SIGNATURES = {"", "?", "unknown", "unresolved", "tbd"}
_UNKNOWN_DIMENSION_SOURCES = {"", "unknown", "unresolved", "tbd"}


@dataclass(frozen=True)
class ReviewGateResult:
    """Result for one publishability gate."""

    status: str
    passed: bool
    blockers: tuple[str, ...] = ()
    evidence: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ReviewAssessment:
    """Deterministic Phase 5 review result."""

    achieved_status: str
    publishable: bool
    gates: tuple[ReviewGateResult, ...]
    trust_status: str = "needs_human"

    @property
    def blockers(self) -> tuple[str, ...]:
        """All blockers from failed concrete gates, preserving gate order."""

        blockers: list[str] = []
        for gate in self.gates:
            if gate.status == "published":
                continue
            blockers.extend(gate.blockers)
        return tuple(blockers)

    def gate(self, status: str) -> ReviewGateResult:
        """Return the gate result for ``status``."""

        for gate in self.gates:
            if gate.status == status:
                return gate
        raise KeyError(status)

    @property
    def blocked(self) -> bool:
        return self.trust_status == "blocked"

    @property
    def needs_human(self) -> bool:
        return self.trust_status == "needs_human"

    @property
    def human_reviewed(self) -> bool:
        return self.trust_status == "human_reviewed"

    def to_report(self) -> "ReviewTrustReport":
        """Return a JSON-friendly, side-effect-free review report."""

        return ReviewTrustReport.from_assessment(self)


@dataclass(frozen=True)
class ReviewTrustReport:
    """Compact trust report for CLI and pipeline callers."""

    achieved_status: str
    trust_status: str
    publishable: bool
    blocked: bool
    needs_human: bool
    human_reviewed: bool
    blockers: tuple[str, ...]
    gates: tuple[Mapping[str, Any], ...]

    @classmethod
    def from_assessment(cls, assessment: ReviewAssessment) -> "ReviewTrustReport":
        return cls(
            achieved_status=assessment.achieved_status,
            trust_status=assessment.trust_status,
            publishable=assessment.publishable,
            blocked=assessment.blocked,
            needs_human=assessment.needs_human,
            human_reviewed=assessment.human_reviewed,
            blockers=assessment.blockers,
            gates=tuple(
                {
                    "status": gate.status,
                    "passed": gate.passed,
                    "blockers": list(gate.blockers),
                    "evidence": dict(gate.evidence or {}),
                }
                for gate in assessment.gates
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "achieved_status": self.achieved_status,
            "trust_status": self.trust_status,
            "publishable": self.publishable,
            "blocked": self.blocked,
            "needs_human": self.needs_human,
            "human_reviewed": self.human_reviewed,
            "blockers": list(self.blockers),
            "gates": [dict(gate) for gate in self.gates],
        }


def assess_publishability(
    *,
    candidate: Mapping[str, Any] | Any | None = None,
    expression: Mapping[str, Any] | Any | None = None,
    variables: Iterable[Mapping[str, Any] | Any] = (),
    references: Iterable[Mapping[str, Any] | Any] = (),
    validity_bounds: Iterable[Mapping[str, Any] | Any] = (),
    relationships: Iterable[Mapping[str, Any] | Any] = (),
    io_specs: Iterable[Mapping[str, Any] | Any] = (),
    min_parse_confidence: float = 0.8,
) -> ReviewAssessment:
    """Assess Phase 5 physics-ingest publishability from local row data.

    Inputs may be Pydantic rows from :mod:`sciona.physics_ingest.staging`, plain
    dictionaries, or objects with attributes. No database client is accepted or
    consulted.
    """

    candidate_row = _row(candidate)
    expression_row = _row(expression)
    variable_rows = tuple(_row(row) for row in variables)
    reference_rows = tuple(_row(row) for row in references)
    bound_rows = tuple(_row(row) for row in validity_bounds)
    relationship_rows = tuple(_row(row) for row in relationships)
    io_spec_rows = tuple(_row(row) for row in io_specs)

    gates = (
        _raw_imported_gate(candidate_row, expression_row),
        _parsed_gate(candidate_row, expression_row, min_parse_confidence),
        _dimension_resolved_gate(expression_row, variable_rows, io_spec_rows),
        _symbolically_validated_gate(expression_row, bound_rows),
        _source_verified_gate(expression_row, reference_rows, relationship_rows),
        _human_reviewed_gate(expression_row, bound_rows),
    )
    published_gate = _published_gate(gates)
    all_gates = (*gates, published_gate)

    achieved_status = "raw_imported"
    for gate in all_gates:
        if gate.passed:
            achieved_status = gate.status
        else:
            break

    return ReviewAssessment(
        achieved_status=achieved_status,
        publishable=published_gate.passed,
        gates=all_gates,
        trust_status=_trust_status(candidate_row, expression_row, bound_rows, all_gates),
    )


def build_review_trust_report(**kwargs: Any) -> dict[str, Any]:
    """Build a JSON-serializable Phase 5 review report without database access."""

    return assess_publishability(**kwargs).to_report().to_dict()


def require_publishable(**kwargs: Any) -> ReviewAssessment:
    """Return an assessment or raise ``ValueError`` with gate blockers."""

    assessment = assess_publishability(**kwargs)
    if not assessment.publishable:
        raise ValueError("; ".join(assessment.blockers))
    return assessment


def _raw_imported_gate(
    candidate: Mapping[str, Any],
    expression: Mapping[str, Any],
) -> ReviewGateResult:
    blockers = []
    blockers.extend(_blocked_status_blockers(candidate, expression))
    if not (
        _text(candidate, "raw_formula")
        or _text(expression, "raw_formula")
        or _text(expression, "sympy_srepr")
        or _mapping(candidate.get("source_payload"))
    ):
        blockers.append("raw import payload is missing")
    return _gate("raw_imported", blockers)


def _parsed_gate(
    candidate: Mapping[str, Any],
    expression: Mapping[str, Any],
    min_parse_confidence: float,
) -> ReviewGateResult:
    blockers = []
    parse_status = _text(expression, "parse_status")
    parse_confidence = max(
        _float(candidate.get("parse_confidence")),
        _float(expression.get("parse_confidence")),
    )
    evidence = _evidence(expression)
    roundtrip_status = _nested_text(
        evidence,
        ("parse_roundtrip", "status"),
        "parse_roundtrip_status",
        ("roundtrip", "status"),
    )

    if parse_status not in _PARSED_PARSE_STATUSES:
        blockers.append("expression parse_status must be parsed or normalized")
    if parse_status in _BLOCKED_STATUSES:
        blockers.append(f"expression parse_status is {parse_status}")
    if parse_confidence < min_parse_confidence:
        blockers.append(
            f"parse_confidence must be >= {min_parse_confidence:g}"
        )
    if not (_text(expression, "sympy_srepr") or _text(expression, "canonical_expr_hash")):
        blockers.append("canonical symbolic payload is missing")
    if not _is_pass(roundtrip_status):
        blockers.append("parse roundtrip evidence must pass")
    return _gate("parsed", blockers, {"parse_roundtrip_status": roundtrip_status})


def _dimension_resolved_gate(
    expression: Mapping[str, Any],
    variables: Sequence[Mapping[str, Any]],
    io_specs: Sequence[Mapping[str, Any]],
) -> ReviewGateResult:
    blockers = []
    evidence = _evidence(expression)
    dimensional_status = _nested_text(
        evidence,
        ("dimensional_analysis", "status"),
        "dimensional_consistency_status",
        ("dimension_check", "status"),
    )

    if not _text(expression, "dimensional_hash"):
        blockers.append("dimensional_hash is missing")
    if variables:
        missing_dims = [
            _text(variable, "symbol_name") or "<unnamed>"
            for variable in variables
            if _text(variable, "variable_role") != "intermediate"
            and _unknown_dimension_signature(_text(variable, "dim_signature"))
        ]
        unknown_sources = [
            _text(variable, "symbol_name") or "<unnamed>"
            for variable in variables
            if _text(variable, "dim_signature")
            and _text(variable, "dimension_source") in _UNKNOWN_DIMENSION_SOURCES
        ]
        if missing_dims:
            blockers.append(
                "variables missing dim_signature: " + ", ".join(missing_dims)
            )
        if unknown_sources:
            blockers.append(
                "variables missing dimension_source: " + ", ".join(unknown_sources)
            )
    else:
        blockers.append("symbolic variables are required for dimension review")

    io_missing = [
        _text(row, "name") or _text(row, "symbol_name") or _text(row, "label") or "<io>"
        for row in io_specs
        if _unknown_dimension_signature(_text(row, "dim_signature"))
    ]
    if io_missing:
        blockers.append("io_specs missing dim_signature: " + ", ".join(io_missing))
    if not _is_pass(dimensional_status):
        blockers.append("dimensional consistency evidence must pass")
    return _gate(
        "dimension_resolved",
        blockers,
        {"dimensional_consistency_status": dimensional_status},
    )


def _symbolically_validated_gate(
    expression: Mapping[str, Any],
    bounds: Sequence[Mapping[str, Any]],
) -> ReviewGateResult:
    blockers = []
    evidence = _evidence(expression)
    numpy_evidence = _mapping(
        evidence.get("numpy_runtime")
        or evidence.get("generated_numpy")
        or evidence.get("numpy_codegen")
    )

    if _text(expression, "validation_status") != "passed":
        blockers.append("validation_status must be passed")
    if _text(expression, "validation_status") in _BLOCKED_STATUSES:
        blockers.append(f"validation_status is {_text(expression, 'validation_status')}")
    if not _text(expression, "canonical_expr_hash"):
        blockers.append("canonical_expr_hash is missing")
    if not _text(expression, "topology_hash"):
        blockers.append("topology_hash is missing")
    if not bounds and not _bool(evidence.get("validity_bounds_not_required")):
        blockers.append("validity bounds are required or must be explicitly waived")
    for index, bound in enumerate(bounds):
        label = _text(bound, "variable_name") or _text(bound, "regime_label") or str(index)
        if _text(bound, "review_status") == "blocked":
            blockers.append(f"validity bound {label} is blocked")
        if not (
            _text(bound, "validity_statement")
            or _text(bound, "regime_label")
            or bound.get("lower_value") is not None
            or bound.get("upper_value") is not None
        ):
            blockers.append(f"validity bound {label} has no constraint")

    if numpy_evidence:
        imports = tuple(str(item) for item in numpy_evidence.get("runtime_imports", ()))
        source = str(numpy_evidence.get("source", ""))
        if not _bool(numpy_evidence.get("no_sympy_runtime")):
            blockers.append("generated NumPy evidence must assert no_sympy_runtime")
        if "sympy" in imports or "import sympy" in source or "from sympy" in source:
            blockers.append("generated NumPy runtime evidence imports SymPy")
        if numpy_evidence.get("tests_passed") is not None and not _bool(
            numpy_evidence.get("tests_passed")
        ):
            blockers.append("generated NumPy runtime tests did not pass")

    return _gate(
        "symbolically_validated",
        blockers,
        {"numpy_runtime_checked": bool(numpy_evidence)},
    )


def _source_verified_gate(
    expression: Mapping[str, Any],
    references: Sequence[Mapping[str, Any]],
    relationships: Sequence[Mapping[str, Any]],
) -> ReviewGateResult:
    blockers = []
    evidence = _evidence(expression)
    verified_refs = [ref for ref in references if _reference_verified(ref)]
    if not verified_refs:
        blockers.append("at least one verified reference is required")

    dependencies = _declared_dependencies(evidence)
    dependency_relationships = [
        row for row in relationships if _text(row, "relationship_kind") in _DEPENDENCY_KINDS
    ]
    if dependencies and not dependency_relationships:
        blockers.append("declared constants/data dependencies need relationships")
    for relationship in dependency_relationships:
        kind = _text(relationship, "relationship_kind")
        label = _text(relationship, "relationship_label") or kind
        if not _bool(relationship.get("verified")):
            blockers.append(f"{label} relationship must be verified")
        if not (
            relationship.get("target_artifact_id")
            or relationship.get("target_version_id")
            or relationship.get("target_expression_id")
            or _text(relationship, "target_node_id")
        ):
            blockers.append(f"{label} relationship is missing a target")

    return _gate(
        "source_verified",
        blockers,
        {
            "verified_reference_count": len(verified_refs),
            "dependency_relationship_count": len(dependency_relationships),
        },
    )


def _human_reviewed_gate(
    expression: Mapping[str, Any],
    bounds: Sequence[Mapping[str, Any]],
) -> ReviewGateResult:
    blockers = []
    evidence = _evidence(expression)
    human_review = _mapping(evidence.get("human_review"))
    review_status = _text(expression, "review_status") or "unreviewed"
    if review_status == "blocked":
        blockers.append("expression review_status is blocked")
    elif review_status == "needs_human":
        blockers.append("expression review_status needs human review")
    elif review_status != "human_reviewed":
        blockers.append("expression review_status must be human_reviewed")
    if not (human_review.get("reviewer_id") or human_review.get("reviewed_by")):
        blockers.append("human review evidence must identify a reviewer")
    if not (human_review.get("reviewed_at") or human_review.get("timestamp")):
        blockers.append("human review evidence must include a review timestamp")
    for index, bound in enumerate(bounds):
        if _text(bound, "review_status") not in _REVIEWED_BOUND_STATUSES:
            label = _text(bound, "variable_name") or _text(bound, "regime_label") or str(index)
            blockers.append(f"validity bound {label} must be reviewed")
    return _gate("human_reviewed", blockers)


def _trust_status(
    candidate: Mapping[str, Any],
    expression: Mapping[str, Any],
    bounds: Sequence[Mapping[str, Any]],
    gates: Sequence[ReviewGateResult],
) -> str:
    statuses = {
        _text(candidate, "candidate_status"),
        _text(expression, "parse_status"),
        _text(expression, "review_status"),
        _text(expression, "validation_status"),
        *(_text(bound, "review_status") for bound in bounds),
    }
    if statuses & _BLOCKED_STATUSES:
        return "blocked"
    if all(gate.passed for gate in gates) and _text(expression, "review_status") == "human_reviewed":
        return "human_reviewed"
    return "needs_human"


def _blocked_status_blockers(
    candidate: Mapping[str, Any],
    expression: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    candidate_status = _text(candidate, "candidate_status")
    expression_status = _text(expression, "review_status")
    if candidate_status in _BLOCKED_STATUSES:
        blockers.append(f"candidate_status is {candidate_status}")
    if expression_status == "blocked":
        blockers.append("expression review_status is blocked")
    return blockers


def _published_gate(gates: Sequence[ReviewGateResult]) -> ReviewGateResult:
    blockers = []
    for gate in gates:
        blockers.extend(gate.blockers)
    return _gate("published", blockers)


def _gate(
    status: str,
    blockers: Sequence[str],
    evidence: Mapping[str, Any] | None = None,
) -> ReviewGateResult:
    return ReviewGateResult(
        status=status,
        passed=not blockers,
        blockers=tuple(blockers),
        evidence=evidence,
    )


def _row(value: Mapping[str, Any] | Any | None) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"cannot treat {type(value)!r} as a review row")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _evidence(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(row.get("evidence_json"))


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return value if isinstance(value, str) else ""


def _unknown_dimension_signature(value: str) -> bool:
    return value.strip().lower() in _UNKNOWN_DIM_SIGNATURES


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _PASS_VALUES
    return bool(value)


def _is_pass(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in _PASS_VALUES
    return value in _PASS_VALUES


def _nested_text(row: Mapping[str, Any], *paths: str | tuple[str, ...]) -> str:
    for path in paths:
        parts = (path,) if isinstance(path, str) else path
        current: Any = row
        for part in parts:
            if not isinstance(current, Mapping) or part not in current:
                current = None
                break
            current = current[part]
        if isinstance(current, str):
            return current
        if isinstance(current, bool):
            return str(current).lower()
    return ""


def _reference_verified(reference: Mapping[str, Any]) -> bool:
    has_locator = any(
        _text(reference, key)
        for key in (
            "doi",
            "url",
            "source_uri",
            "reference_uri",
            "isbn",
            "arxiv_id",
            "title",
        )
    )
    if not has_locator:
        return False
    if reference.get("verified") is not None:
        return _bool(reference.get("verified"))
    status = _text(reference, "review_status") or _text(reference, "verification_status")
    return status in {"verified", "source_verified", "human_reviewed", "automated_pass"}


def _declared_dependencies(evidence: Mapping[str, Any]) -> tuple[Any, ...]:
    constants = evidence.get("required_constants") or evidence.get("constants") or ()
    data = evidence.get("required_data_artifacts") or evidence.get("data_dependencies") or ()
    if isinstance(constants, str):
        constants = (constants,)
    if isinstance(data, str):
        data = (data,)
    if not isinstance(constants, Sequence):
        constants = ()
    if not isinstance(data, Sequence):
        data = ()
    return (*constants, *data)
