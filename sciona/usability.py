"""Cross-family heuristic-derived usability assessments."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_TEXT_TOKEN_RE = re.compile(r"[a-z0-9]+")
_BANNED_SHARED_TOKENS = {
    "beat",
    "bpm",
    "ecg",
    "eeg",
    "emg",
    "heart",
    "ppg",
    "qrs",
    "rr",
}


class UsabilityScope(str, Enum):
    """Assessment scopes used across guidance, scoring, and benchmark policy."""

    GUIDANCE = "guidance"
    SCORING = "scoring"
    FINAL_BENCHMARK = "final_benchmark"


class UsabilityReasonKind(str, Enum):
    """Canonical severity classes for usability explanations."""

    BLOCKING = "blocking"
    WARNING = "warning"


class UsabilityProvenanceKind(str, Enum):
    """Family-neutral provenance sources for usability decisions."""

    HEURISTIC_OBSERVATION = "heuristic_observation"
    RUNTIME_ASSESSOR = "runtime_assessor"
    FAMILY_RULE_REGISTRY = "family_rule_registry"
    BENCHMARK_POLICY = "benchmark_policy"
    OUTCOME_MEMORY = "outcome_memory"
    MANUAL_REVIEW = "manual_review"


class UsabilityBlockingReasonCode(str, Enum):
    """Cross-family blocking reasons for data usability."""

    REQUIRED_INPUT_MISSING = "required_input_missing"
    REQUIRED_REFERENCE_MISSING = "required_reference_missing"
    COVERAGE_INSUFFICIENT = "coverage_insufficient"
    TIMING_CONTEXT_INCOHERENT = "timing_context_incoherent"
    ALIGNMENT_ERROR = "alignment_error"
    PLAUSIBILITY_FRAGMENTATION = "plausibility_fragmentation"
    BENCHMARK_CONTRACT_MISMATCH = "benchmark_contract_mismatch"


class UsabilityWarningReasonCode(str, Enum):
    """Cross-family warning reasons for degraded but potentially useful data."""

    QUALITY_INSTABILITY = "quality_instability"
    OUTPUT_DENSITY_COLLAPSE = "output_density_collapse"
    CONFIDENCE_FRAGMENTATION = "confidence_fragmentation"
    PARTIAL_COVERAGE = "partial_coverage"
    REVIEW_RECOMMENDED = "review_recommended"
    NARROW_SUPPORT = "narrow_support"


_BLOCKING_REASON_CODES = {item.value for item in UsabilityBlockingReasonCode}
_WARNING_REASON_CODES = {item.value for item in UsabilityWarningReasonCode}
_ALL_REASON_CODES = _BLOCKING_REASON_CODES | _WARNING_REASON_CODES


def _validate_shared_identifier(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not _IDENTIFIER_RE.fullmatch(text):
        raise ValueError(f"{label} must be lowercase snake_case")
    tokens = [token for token in text.split("_") if token]
    banned = sorted(token for token in tokens if token in _BANNED_SHARED_TOKENS)
    if banned:
        raise ValueError(
            f"{label} must stay de-jargonized; found domain-specific tokens: "
            + ", ".join(banned)
        )
    return text


def _normalize_text_tokens(value: str) -> list[str]:
    return _TEXT_TOKEN_RE.findall(str(value or "").lower())


def _validate_shared_text(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must not be empty")
    banned = sorted(
        token for token in _normalize_text_tokens(text) if token in _BANNED_SHARED_TOKENS
    )
    if banned:
        raise ValueError(
            f"{label} must stay de-jargonized; found domain-specific tokens: "
            + ", ".join(banned)
        )
    return text


class UsabilityProvenance(BaseModel):
    """Provenance attached to an assessment or reason."""

    kind: UsabilityProvenanceKind
    source_id: str = ""
    reference: str = ""
    note: str = ""

    @model_validator(mode="after")
    def _validate_provenance(self) -> "UsabilityProvenance":
        if not (self.source_id.strip() or self.reference.strip() or self.note.strip()):
            raise ValueError(
                "Usability provenance must include at least one of source_id, reference, or note"
            )
        if self.kind in {
            UsabilityProvenanceKind.HEURISTIC_OBSERVATION,
            UsabilityProvenanceKind.RUNTIME_ASSESSOR,
            UsabilityProvenanceKind.FAMILY_RULE_REGISTRY,
            UsabilityProvenanceKind.BENCHMARK_POLICY,
            UsabilityProvenanceKind.OUTCOME_MEMORY,
        } and not self.source_id.strip():
            raise ValueError(
                f"Usability provenance kind '{self.kind.value}' requires a structured source_id"
            )
        return self


class CanonicalUsabilityReasonDefinition(BaseModel):
    """Governed shared definition for one canonical usability reason code."""

    code: str
    kind: UsabilityReasonKind
    display_name: str
    shared_meaning: str
    governance_rationale: str
    required_provenance_kinds: list[UsabilityProvenanceKind] = Field(default_factory=list)
    uncertainty_guidance: str = ""

    @model_validator(mode="after")
    def _validate_definition(self) -> "CanonicalUsabilityReasonDefinition":
        self.code = _validate_shared_identifier(
            self.code,
            label="Canonical usability reason code",
        )
        self.display_name = _validate_shared_text(
            self.display_name,
            label="Canonical usability display name",
        )
        self.shared_meaning = _validate_shared_text(
            self.shared_meaning,
            label="Canonical usability shared meaning",
        )
        self.governance_rationale = _validate_shared_text(
            self.governance_rationale,
            label="Canonical usability governance rationale",
        )
        if self.code not in _ALL_REASON_CODES:
            raise ValueError(f"Unknown canonical usability reason code: {self.code}")
        if self.kind == UsabilityReasonKind.BLOCKING and self.code not in _BLOCKING_REASON_CODES:
            raise ValueError(f"Blocking reason kind cannot use warning code: {self.code}")
        if self.kind == UsabilityReasonKind.WARNING and self.code not in _WARNING_REASON_CODES:
            raise ValueError(f"Warning reason kind cannot use blocking code: {self.code}")
        return self


def _canonical_reason_definition_data() -> tuple[dict[str, Any], ...]:
    return (
        {
            "code": "required_input_missing",
            "kind": UsabilityReasonKind.BLOCKING,
            "display_name": "Required Input Missing",
            "shared_meaning": "A required input needed to evaluate the artifact is missing.",
            "governance_rationale": "This remains cross-family because it describes absent required context rather than one domain artifact.",
            "required_provenance_kinds": [
                UsabilityProvenanceKind.RUNTIME_ASSESSOR,
                UsabilityProvenanceKind.FAMILY_RULE_REGISTRY,
            ],
            "uncertainty_guidance": "Use uncertainty notes when the requirement depends on incomplete runtime evidence rather than a confirmed absence.",
        },
        {
            "code": "required_reference_missing",
            "kind": UsabilityReasonKind.BLOCKING,
            "display_name": "Required Reference Missing",
            "shared_meaning": "A required reference basis for scoring or review is missing.",
            "governance_rationale": "The shared meaning is about missing support for evaluation, not a family-specific source type.",
            "required_provenance_kinds": [
                UsabilityProvenanceKind.BENCHMARK_POLICY,
                UsabilityProvenanceKind.MANUAL_REVIEW,
            ],
            "uncertainty_guidance": "Use uncertainty notes when the available references are partial rather than fully absent.",
        },
        {
            "code": "coverage_insufficient",
            "kind": UsabilityReasonKind.BLOCKING,
            "display_name": "Coverage Insufficient",
            "shared_meaning": "Observed coverage is too limited to support the intended use.",
            "governance_rationale": "Coverage is expressed as a portable completeness concept rather than any family-local notion of support.",
            "required_provenance_kinds": [
                UsabilityProvenanceKind.HEURISTIC_OBSERVATION,
                UsabilityProvenanceKind.RUNTIME_ASSESSOR,
            ],
        },
        {
            "code": "timing_context_incoherent",
            "kind": UsabilityReasonKind.BLOCKING,
            "display_name": "Timing Context Incoherent",
            "shared_meaning": "The artifact does not preserve a coherent timing or ordering context needed for evaluation.",
            "governance_rationale": "The shared field stays about general timing coherence instead of any discipline-specific clock or stream jargon.",
            "required_provenance_kinds": [
                UsabilityProvenanceKind.RUNTIME_ASSESSOR,
                UsabilityProvenanceKind.BENCHMARK_POLICY,
            ],
        },
        {
            "code": "alignment_error",
            "kind": UsabilityReasonKind.BLOCKING,
            "display_name": "Alignment Error",
            "shared_meaning": "Required structures are misaligned enough that the artifact cannot be trusted for the requested use.",
            "governance_rationale": "Alignment stays generic so families can describe their local mismatch without redefining the shared issue.",
            "required_provenance_kinds": [
                UsabilityProvenanceKind.RUNTIME_ASSESSOR,
                UsabilityProvenanceKind.FAMILY_RULE_REGISTRY,
            ],
        },
        {
            "code": "plausibility_fragmentation",
            "kind": UsabilityReasonKind.BLOCKING,
            "display_name": "Plausibility Fragmentation",
            "shared_meaning": "The available evidence fragments plausibility so strongly that the artifact is not dependable.",
            "governance_rationale": "This preserves a cross-family notion of fractured plausibility rather than domain-specific failure signatures.",
            "required_provenance_kinds": [
                UsabilityProvenanceKind.HEURISTIC_OBSERVATION,
                UsabilityProvenanceKind.RUNTIME_ASSESSOR,
            ],
        },
        {
            "code": "benchmark_contract_mismatch",
            "kind": UsabilityReasonKind.BLOCKING,
            "display_name": "Benchmark Contract Mismatch",
            "shared_meaning": "The artifact does not satisfy the declared benchmark contract for final use.",
            "governance_rationale": "The shared meaning is the benchmark contract itself, not any family-local metric rule.",
            "required_provenance_kinds": [
                UsabilityProvenanceKind.RUNTIME_ASSESSOR,
                UsabilityProvenanceKind.BENCHMARK_POLICY,
            ],
        },
        {
            "code": "quality_instability",
            "kind": UsabilityReasonKind.WARNING,
            "display_name": "Quality Instability",
            "shared_meaning": "Quality varies enough that one fixed interpretation may be brittle.",
            "governance_rationale": "The canonical field captures instability generically so families can add local interpretation without changing meaning.",
            "required_provenance_kinds": [
                UsabilityProvenanceKind.HEURISTIC_OBSERVATION,
                UsabilityProvenanceKind.RUNTIME_ASSESSOR,
            ],
            "uncertainty_guidance": "Warnings about instability should normally explain how strong the evidence is and why the artifact may still be usable.",
        },
        {
            "code": "output_density_collapse",
            "kind": UsabilityReasonKind.WARNING,
            "display_name": "Output Density Collapse",
            "shared_meaning": "Observed output structure is sparser than expected for confident downstream use.",
            "governance_rationale": "The shared field refers to general sparsity or collapse rather than any one output representation.",
            "required_provenance_kinds": [
                UsabilityProvenanceKind.HEURISTIC_OBSERVATION,
            ],
        },
        {
            "code": "confidence_fragmentation",
            "kind": UsabilityReasonKind.WARNING,
            "display_name": "Confidence Fragmentation",
            "shared_meaning": "Confidence is uneven enough that the artifact may require caution or review.",
            "governance_rationale": "Confidence fragmentation is kept as a portable interpretation problem rather than a domain-specific score threshold.",
            "required_provenance_kinds": [
                UsabilityProvenanceKind.RUNTIME_ASSESSOR,
                UsabilityProvenanceKind.MANUAL_REVIEW,
            ],
        },
        {
            "code": "partial_coverage",
            "kind": UsabilityReasonKind.WARNING,
            "display_name": "Partial Coverage",
            "shared_meaning": "Coverage is incomplete but may still support limited use.",
            "governance_rationale": "This stays distinct from blocking coverage insufficiency so families do not collapse different portability levels into one meaning.",
            "required_provenance_kinds": [
                UsabilityProvenanceKind.RUNTIME_ASSESSOR,
                UsabilityProvenanceKind.FAMILY_RULE_REGISTRY,
            ],
        },
        {
            "code": "review_recommended",
            "kind": UsabilityReasonKind.WARNING,
            "display_name": "Review Recommended",
            "shared_meaning": "The artifact remains usable in principle but should be reviewed before higher-stakes use.",
            "governance_rationale": "The shared meaning protects review as a generic governance action rather than a family-specific escalation trick.",
            "required_provenance_kinds": [
                UsabilityProvenanceKind.RUNTIME_ASSESSOR,
                UsabilityProvenanceKind.MANUAL_REVIEW,
                UsabilityProvenanceKind.FAMILY_RULE_REGISTRY,
            ],
        },
        {
            "code": "narrow_support",
            "kind": UsabilityReasonKind.WARNING,
            "display_name": "Narrow Support",
            "shared_meaning": "The supporting evidence is too narrow to justify broad confidence.",
            "governance_rationale": "The shared meaning is about limited support breadth, which keeps the field portable across families.",
            "required_provenance_kinds": [
                UsabilityProvenanceKind.HEURISTIC_OBSERVATION,
                UsabilityProvenanceKind.RUNTIME_ASSESSOR,
                UsabilityProvenanceKind.OUTCOME_MEMORY,
            ],
        },
    )


_CANONICAL_REASON_DEFINITIONS = {
    item["code"]: CanonicalUsabilityReasonDefinition.model_validate(item)
    for item in _canonical_reason_definition_data()
}


def canonical_usability_reason_definition(
    code: str,
) -> CanonicalUsabilityReasonDefinition:
    """Return the governed shared definition for one reason code."""
    normalized = _validate_shared_identifier(
        code,
        label="Canonical usability reason code",
    )
    if normalized not in _CANONICAL_REASON_DEFINITIONS:
        raise ValueError(f"Unknown canonical usability reason code: {normalized}")
    return _CANONICAL_REASON_DEFINITIONS[normalized]


def canonical_usability_reason_definitions() -> tuple[CanonicalUsabilityReasonDefinition, ...]:
    """Return all governed shared reason definitions in stable order."""
    return tuple(_CANONICAL_REASON_DEFINITIONS[key] for key in sorted(_CANONICAL_REASON_DEFINITIONS))


class UsabilityReason(BaseModel):
    """One explicit explanation for a usability decision."""

    kind: UsabilityReasonKind
    code: str
    summary: str = ""
    related_heuristic_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    uncertainty_notes: list[str] = Field(default_factory=list)
    provenance: list[UsabilityProvenance] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_reason(self) -> "UsabilityReason":
        self.code = _validate_shared_identifier(self.code, label="Usability reason code")
        if self.code not in _ALL_REASON_CODES:
            raise ValueError(f"Unknown canonical usability reason code: {self.code}")
        if self.kind == UsabilityReasonKind.BLOCKING and self.code not in _BLOCKING_REASON_CODES:
            raise ValueError(f"Blocking reason kind cannot use warning code: {self.code}")
        if self.kind == UsabilityReasonKind.WARNING and self.code not in _WARNING_REASON_CODES:
            raise ValueError(f"Warning reason kind cannot use blocking code: {self.code}")
        normalized: list[str] = []
        for item in self.related_heuristic_ids:
            normalized.append(_validate_shared_identifier(item, label="Heuristic identifier"))
        self.related_heuristic_ids = normalized
        self.summary = _validate_shared_text(
            self.summary,
            label="Usability reason summary",
        )
        if self.confidence < 1.0 and not self.uncertainty_notes:
            raise ValueError(
                "Usability reasons with confidence below 1.0 must include uncertainty notes"
            )
        if not self.provenance:
            raise ValueError("Usability reasons must include structured provenance")
        definition = canonical_usability_reason_definition(self.code)
        if definition.kind != self.kind:
            raise ValueError(
                f"Usability reason '{self.code}' must use canonical kind {definition.kind.value}"
            )
        provenance_kinds = {item.kind for item in self.provenance}
        if (
            definition.required_provenance_kinds
            and not provenance_kinds.intersection(definition.required_provenance_kinds)
        ):
            required = ", ".join(
                item.value for item in definition.required_provenance_kinds
            )
            raise ValueError(
                f"Usability reason '{self.code}' must include provenance compatible with: {required}"
            )
        return self


class UsabilityScopeAssessment(BaseModel):
    """Scope-specific usability decision and explanation payload."""

    scope: UsabilityScope
    usable: bool
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    uncertainty_notes: list[str] = Field(default_factory=list)
    provenance: list[UsabilityProvenance] = Field(default_factory=list)
    blocking_reasons: list[UsabilityReason] = Field(default_factory=list)
    warning_reasons: list[UsabilityReason] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_reason_kinds(self) -> "UsabilityScopeAssessment":
        for reason in self.blocking_reasons:
            if reason.kind != UsabilityReasonKind.BLOCKING:
                raise ValueError(
                    f"blocking_reasons may only contain blocking reasons, got {reason.code}"
                )
        for reason in self.warning_reasons:
            if reason.kind != UsabilityReasonKind.WARNING:
                raise ValueError(
                    f"warning_reasons may only contain warning reasons, got {reason.code}"
                )
        if self.usable and self.blocking_reasons:
            raise ValueError(
                f"Usability scope '{self.scope.value}' cannot be usable when blocking reasons are present"
            )
        return self


class UsabilityAssessment(BaseModel):
    """First-class usability record for one evaluated member or artifact context."""

    assessment_id: str
    family: str = ""
    task_intent: str = ""
    heuristic_signature: list[str] = Field(default_factory=list)
    required_contracts_checked: list[str] = Field(default_factory=list)
    usable_for_guidance: bool
    usable_for_scoring: bool
    usable_for_final_benchmark: bool
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    uncertainty_notes: list[str] = Field(default_factory=list)
    provenance: list[UsabilityProvenance] = Field(default_factory=list)
    guidance: UsabilityScopeAssessment
    scoring: UsabilityScopeAssessment
    final_benchmark: UsabilityScopeAssessment

    @model_validator(mode="after")
    def _validate_assessment(self) -> "UsabilityAssessment":
        self.assessment_id = _validate_shared_identifier(
            self.assessment_id,
            label="Assessment identifier",
        )
        normalized_signature: list[str] = []
        for item in self.heuristic_signature:
            normalized_signature.append(
                _validate_shared_identifier(item, label="Heuristic identifier")
            )
        self.heuristic_signature = normalized_signature
        normalized_contracts: list[str] = []
        for item in self.required_contracts_checked:
            normalized_contracts.append(
                _validate_shared_identifier(item, label="Required contract identifier")
            )
        self.required_contracts_checked = normalized_contracts

        expected_scopes = {
            "guidance": UsabilityScope.GUIDANCE,
            "scoring": UsabilityScope.SCORING,
            "final_benchmark": UsabilityScope.FINAL_BENCHMARK,
        }
        for field_name, expected_scope in expected_scopes.items():
            scope_assessment = getattr(self, field_name)
            if scope_assessment.scope != expected_scope:
                raise ValueError(
                    f"{field_name} must use scope {expected_scope.value}, got {scope_assessment.scope.value}"
                )

        if self.usable_for_guidance != self.guidance.usable:
            raise ValueError("usable_for_guidance must match guidance.usable")
        if self.usable_for_scoring != self.scoring.usable:
            raise ValueError("usable_for_scoring must match scoring.usable")
        if self.usable_for_final_benchmark != self.final_benchmark.usable:
            raise ValueError(
                "usable_for_final_benchmark must match final_benchmark.usable"
            )
        if self.usable_for_final_benchmark and not self.usable_for_scoring:
            raise ValueError(
                "final benchmark usability cannot be true when scoring usability is false"
            )
        if self.confidence < 1.0 and not self.uncertainty_notes:
            raise ValueError(
                "Usability assessments with confidence below 1.0 must include uncertainty notes"
            )
        if any(
            (
                self.guidance.blocking_reasons
                or self.guidance.warning_reasons
                or self.scoring.blocking_reasons
                or self.scoring.warning_reasons
                or self.final_benchmark.blocking_reasons
                or self.final_benchmark.warning_reasons
            )
        ) and not self.provenance:
            raise ValueError(
                "Usability assessments with explicit reasons must include top-level provenance"
            )
        return self


def known_usability_blocking_reason_codes() -> tuple[str, ...]:
    """Return canonical blocking reason codes."""

    return tuple(item.value for item in UsabilityBlockingReasonCode)


def known_usability_warning_reason_codes() -> tuple[str, ...]:
    """Return canonical warning reason codes."""

    return tuple(item.value for item in UsabilityWarningReasonCode)


def known_usability_reason_codes() -> tuple[str, ...]:
    """Return all canonical usability reason codes."""

    return tuple(sorted(_ALL_REASON_CODES))


def known_canonical_usability_reason_definitions() -> tuple[CanonicalUsabilityReasonDefinition, ...]:
    """Return the governed shared usability reason catalog."""

    return canonical_usability_reason_definitions()


__all__ = [
    "CanonicalUsabilityReasonDefinition",
    "UsabilityAssessment",
    "UsabilityBlockingReasonCode",
    "UsabilityProvenance",
    "UsabilityProvenanceKind",
    "UsabilityReason",
    "UsabilityReasonKind",
    "UsabilityScope",
    "UsabilityScopeAssessment",
    "UsabilityWarningReasonCode",
    "canonical_usability_reason_definition",
    "canonical_usability_reason_definitions",
    "known_usability_blocking_reason_codes",
    "known_canonical_usability_reason_definitions",
    "known_usability_reason_codes",
    "known_usability_warning_reason_codes",
]
