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


@dataclass(frozen=True)
class ReviewPublicationDiagnostic:
    """One non-fatal status-row publication diagnostic."""

    table: str
    reason: str
    row_id: str = ""
    severity: str = "skipped"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "reason": self.reason,
            "row_id": self.row_id,
            "severity": self.severity,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ReviewPublicationRows:
    """Side-effect-free status row patches derived from a review decision."""

    artifact_symbolic_expressions: tuple[dict[str, Any], ...] = ()
    physics_equation_candidates: tuple[dict[str, Any], ...] = ()
    diagnostics: tuple[ReviewPublicationDiagnostic, ...] = ()

    def to_upsert_rows(self) -> dict[str, list[dict[str, Any]]]:
        rows: dict[str, list[dict[str, Any]]] = {}
        if self.physics_equation_candidates:
            rows["physics_equation_candidates"] = [
                dict(row) for row in self.physics_equation_candidates
            ]
        if self.artifact_symbolic_expressions:
            rows["artifact_symbolic_expressions"] = [
                dict(row) for row in self.artifact_symbolic_expressions
            ]
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_symbolic_expressions": [
                dict(row) for row in self.artifact_symbolic_expressions
            ],
            "physics_equation_candidates": [
                dict(row) for row in self.physics_equation_candidates
            ],
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
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


def build_review_publication_status_rows(
    review: ReviewAssessment | ReviewTrustReport | Mapping[str, Any],
    *,
    candidate: Mapping[str, Any] | Any | None = None,
    expression: Mapping[str, Any] | Any | None = None,
) -> ReviewPublicationRows:
    """Build deterministic publication status patches from a review decision.

    The returned rows are intentionally minimal and inert. They include only the
    table conflict key plus status columns that callers may choose to upsert:
    ``artifact_symbolic_expressions.review_status`` and
    ``physics_equation_candidates.candidate_status``. Rows are omitted when the
    target ID is unavailable or the review decision is not actionable.
    """

    review_row = _review_row(review)
    review_status = _publication_review_status(review, review_row)
    diagnostics: list[ReviewPublicationDiagnostic] = []
    expression_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    if review_status == "unreviewed":
        diagnostics.append(
            ReviewPublicationDiagnostic(
                table="review",
                reason="non_actionable_assessment",
                severity="info",
                detail="review decision did not progress beyond unreviewed",
            )
        )
        return ReviewPublicationRows(diagnostics=tuple(diagnostics))

    expression_row = _row(expression)
    expression_id = _text(expression_row, "expression_id")
    if expression_row:
        if expression_id:
            expression_rows.append(
                {
                    "expression_id": expression_id,
                    "review_status": review_status,
                }
            )
        else:
            diagnostics.append(
                ReviewPublicationDiagnostic(
                    table="artifact_symbolic_expressions",
                    reason="missing_expression_id",
                    detail="review_status patch requires expression_id",
                )
            )

    candidate_row = _row(candidate)
    candidate_id = _text(candidate_row, "candidate_id")
    if candidate_row:
        if candidate_id:
            candidate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_status": _candidate_status_for_review(
                        review_row, review_status
                    ),
                }
            )
        else:
            diagnostics.append(
                ReviewPublicationDiagnostic(
                    table="physics_equation_candidates",
                    reason="missing_candidate_id",
                    detail="candidate_status patch requires candidate_id",
                )
            )

    if not expression_row and not candidate_row:
        diagnostics.append(
            ReviewPublicationDiagnostic(
                table="review",
                reason="missing_target_rows",
                detail="provide candidate and/or expression rows to build status patches",
            )
        )

    return ReviewPublicationRows(
        artifact_symbolic_expressions=tuple(expression_rows),
        physics_equation_candidates=tuple(candidate_rows),
        diagnostics=tuple(diagnostics),
    )


def require_publishable(**kwargs: Any) -> ReviewAssessment:
    """Return an assessment or raise ``ValueError`` with gate blockers."""

    assessment = assess_publishability(**kwargs)
    if not assessment.publishable:
        raise ValueError("; ".join(assessment.blockers))
    return assessment


def _review_row(
    review: ReviewAssessment | ReviewTrustReport | Mapping[str, Any],
) -> Mapping[str, Any]:
    if isinstance(review, ReviewAssessment):
        return review.to_report().to_dict()
    if isinstance(review, ReviewTrustReport):
        return review.to_dict()
    if isinstance(review, Mapping):
        return review
    raise TypeError(f"cannot treat {type(review)!r} as a review decision")


def _publication_review_status(
    review: ReviewAssessment | ReviewTrustReport | Mapping[str, Any],
    review_row: Mapping[str, Any],
) -> str:
    trust_status = _text(review_row, "trust_status")
    if trust_status in REVIEW_STATUSES:
        if trust_status == "needs_human" and _automated_gates_passed(review_row):
            return "automated_pass"
        return trust_status
    if _bool(review_row.get("blocked")):
        return "blocked"
    if _bool(review_row.get("human_reviewed")):
        return "human_reviewed"
    if _bool(review_row.get("needs_human")):
        return "automated_pass" if _automated_gates_passed(review_row) else "needs_human"
    if _bool(review_row.get("publishable")):
        return "human_reviewed"
    if isinstance(review, ReviewAssessment):
        return "needs_human"
    return "unreviewed"


def _automated_gates_passed(review_row: Mapping[str, Any]) -> bool:
    gates = review_row.get("gates")
    if not isinstance(gates, Sequence) or isinstance(gates, (str, bytes)):
        return False
    passed_by_status: dict[str, bool] = {}
    for gate in gates:
        if isinstance(gate, Mapping):
            status = _text(gate, "status")
            if status:
                passed_by_status[status] = _bool(gate.get("passed"))
    automated_statuses = WORKFLOW_STATUSES[:5]
    return all(passed_by_status.get(status) is True for status in automated_statuses)


def _candidate_status_for_review(
    review_row: Mapping[str, Any],
    review_status: str,
) -> str:
    if review_status == "blocked":
        return "blocked"
    if review_status == "human_reviewed":
        return "human_reviewed"
    achieved_status = _text(review_row, "achieved_status")
    if achieved_status in _STATUS_RANK and achieved_status != "published":
        return achieved_status
    if review_status == "automated_pass":
        return "source_verified"
    return "raw_imported"


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
    mechanism_evidence = _mechanism_classification_evidence(expression, evidence)
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
        {
            "numpy_runtime_checked": bool(numpy_evidence),
            **mechanism_evidence,
        },
    )


def _source_verified_gate(
    expression: Mapping[str, Any],
    references: Sequence[Mapping[str, Any]],
    relationships: Sequence[Mapping[str, Any]],
) -> ReviewGateResult:
    blockers = []
    evidence = _evidence(expression)
    mechanism_evidence = _mechanism_classification_evidence(expression, evidence)
    verified_refs = [ref for ref in references if _reference_verified(ref)]
    if not verified_refs:
        blockers.append("at least one verified reference is required")
    if not mechanism_evidence["has_mechanism_classification"]:
        blockers.append("mechanism classification evidence is required")

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
            **mechanism_evidence,
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


def _mechanism_classification_evidence(
    expression: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    tags: list[str] = []
    archetypes: list[str] = []
    details: list[str] = []
    sources: list[str] = []

    def add_strings(target: list[str], values: Any, source: str) -> None:
        items = _string_sequence(values)
        if not items:
            return
        target.extend(item for item in items if item not in target)
        if source not in sources:
            sources.append(source)

    def add_detail(value: Any, source: str) -> None:
        items = _string_sequence(value)
        if not items:
            return
        details.extend(item for item in items if item not in details)
        if source not in sources:
            sources.append(source)

    add_strings(tags, expression.get("mechanism_tags"), "expression.mechanism_tags")
    add_strings(
        archetypes,
        expression.get("behavioral_archetypes"),
        "expression.behavioral_archetypes",
    )
    for key in (
        "mechanism_classification",
        "mechanism_label",
        "classification_label",
    ):
        add_detail(expression.get(key), f"expression.{key}")

    add_strings(tags, evidence.get("mechanism_tags"), "evidence_json.mechanism_tags")
    add_strings(
        archetypes,
        evidence.get("behavioral_archetypes"),
        "evidence_json.behavioral_archetypes",
    )
    for key in (
        "mechanism_classification",
        "mechanism_label",
        "classification_label",
    ):
        add_detail(evidence.get(key), f"evidence_json.{key}")

    for container_key in ("mechanism", "classification", "mechanism_classification"):
        value = evidence.get(container_key)
        if isinstance(value, Mapping):
            source = f"evidence_json.{container_key}"
            add_strings(
                tags,
                value.get("mechanism_tags")
                or value.get("mechanisms")
                or value.get("physics_mechanisms"),
                source,
            )
            add_strings(
                archetypes,
                value.get("behavioral_archetypes") or value.get("archetypes"),
                source,
            )
            for detail_key in (
                "mechanism",
                "mechanism_label",
                "mechanism_class",
                "classification",
                "classification_label",
                "class",
                "category",
                "rationale",
                "basis",
                "justification",
                "notes",
            ):
                add_detail(value.get(detail_key), source)
        else:
            add_detail(value, f"evidence_json.{container_key}")

    return {
        "has_mechanism_classification": bool(tags or archetypes or details),
        "mechanism_tag_count": len(tags),
        "behavioral_archetype_count": len(archetypes),
        "classification_detail_count": len(details),
        "mechanism_evidence_sources": list(sources),
    }


def _string_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, Mapping) or not isinstance(value, Iterable):
        return ()
    strings = []
    for item in value:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                strings.append(stripped)
    return tuple(strings)
