"""Canonical cross-family heuristic contracts and compatibility helpers."""

from __future__ import annotations

import json
import re
from enum import Enum
from functools import lru_cache

from pydantic import BaseModel, Field, model_validator
from sciona.atom_identity import candidate_atom_provider_roots


_HEURISTIC_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_BANNED_SHARED_TOKENS = {
    "rr",
    "ecg",
    "ppg",
    "eeg",
    "emg",
    "pcg",
    "bpm",
    "sqi",
    "baseline",
    "wander",
    "qrs",
    "beat",
    "heart",
}

_EXPLICIT_CANONICAL_HEURISTIC_IDS = {
    "alignment_error",
    "boundary_discontinuity",
    "confidence_instability",
    "constraint_violation_risk",
    "convergence_instability",
    "coverage_fragmentation",
    "density_collapse",
    "dominant_nuisance_structure",
    "interval_instability",
    "numerical_condition_instability",
    "oscillation_instability",
    "plausibility_fragmentation",
    "quality_instability",
    "residual_structure_after_transform",
    "resource_growth_instability",
}
EXTERNAL_CANONICAL_ASSET_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("data", "heuristics", "canonical_registry.json"),
)


class HeuristicEvidenceType(str, Enum):
    """Family-neutral shapes for first-class heuristic values."""

    SCALAR_SCORE = "scalar_score"
    BOOLEAN_FLAG = "boolean_flag"
    DISTRIBUTION_SUMMARY = "distribution_summary"
    CATEGORICAL_LABEL = "categorical_label"
    STRUCTURED_SUMMARY = "structured_summary"


class HeuristicProducerKind(str, Enum):
    """How a heuristic value was produced."""

    ATOM_OUTPUT = "atom_output"
    DIAGNOSTIC_ATOM = "diagnostic_atom"
    RUNTIME_TRANSFORM = "runtime_transform"
    COMPATIBILITY_MAPPING = "compatibility_mapping"


class HeuristicApplicabilityScope(str, Enum):
    """Scope over which a heuristic identifier remains semantically stable."""

    CROSS_FAMILY = "cross_family"
    FAMILY_LOCAL = "family_local"
    SKELETON_LOCAL = "skeleton_local"


class HeuristicActionClass(str, Enum):
    """Generic structural responses supported by the shared heuristic layer."""

    PRECONDITION = "precondition"
    REPLACE_STAGE = "replace_stage"
    SPLIT_STAGE = "split_stage"
    INSERT_CORRECTION = "insert_correction"
    GATE_OR_VALIDATE = "gate_or_validate"
    SMOOTH_OR_AGGREGATE = "smooth_or_aggregate"
    BRANCH_AND_COMPARE = "branch_and_compare"


class HeuristicCompatibilityHint(BaseModel):
    """Compatibility bridge from legacy diagnostics into canonical heuristics."""

    legacy_metric_name: str
    heuristic_id: str
    supported_action_classes: list[HeuristicActionClass] = Field(default_factory=list)
    rationale: str = ""
    source_domain: str = ""
    notes: list[str] = Field(default_factory=list)


class CanonicalHeuristic(BaseModel):
    """First-class audited heuristic contract used across families."""

    heuristic_id: str
    display_name: str
    dejargonized_meaning: str
    evidence_type: HeuristicEvidenceType
    value_kind: str = ""
    value_shape: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    uncertainty_notes: list[str] = Field(default_factory=list)
    producer_kind: HeuristicProducerKind = HeuristicProducerKind.RUNTIME_TRANSFORM
    applicability_scope: HeuristicApplicabilityScope = (
        HeuristicApplicabilityScope.CROSS_FAMILY
    )
    supported_action_classes: list[HeuristicActionClass] = Field(default_factory=list)
    provenance_requirements: list[str] = Field(default_factory=list)
    compatibility_aliases: list[str] = Field(default_factory=list)
    family_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_shared_identifier(self) -> "CanonicalHeuristic":
        heuristic_id = self.heuristic_id.strip()
        if not _HEURISTIC_ID_RE.fullmatch(heuristic_id):
            raise ValueError(
                "heuristic_id must be snake_case with lowercase alphanumeric tokens"
            )
        tokens = [token for token in heuristic_id.split("_") if token]
        banned = sorted(token for token in tokens if token in _BANNED_SHARED_TOKENS)
        if banned and self.applicability_scope == HeuristicApplicabilityScope.CROSS_FAMILY:
            raise ValueError(
                "Cross-family heuristic identifiers must stay de-jargonized; "
                f"found domain-specific tokens: {', '.join(banned)}"
            )
        return self

@lru_cache(maxsize=1)
def _external_canonical_heuristics() -> tuple[CanonicalHeuristic, ...]:
    for root in candidate_atom_provider_roots():
        for rel in EXTERNAL_CANONICAL_ASSET_CANDIDATES:
            path = root.joinpath(*rel)
            if not path.exists():
                continue
            raw = json.loads(path.read_text())
            heuristics = raw.get("heuristics", []) or []
            loaded: list[CanonicalHeuristic] = []
            for item in heuristics:
                if not isinstance(item, dict):
                    continue
                payload = dict(item)
                payload.setdefault("producer_kind", HeuristicProducerKind.RUNTIME_TRANSFORM)
                payload.setdefault(
                    "applicability_scope",
                    HeuristicApplicabilityScope.CROSS_FAMILY,
                )
                loaded.append(CanonicalHeuristic.model_validate(payload))
            if loaded:
                return tuple(loaded)
    return tuple()


_LEGACY_COMPATIBILITY_HINTS: dict[str, HeuristicCompatibilityHint] = {
    "jump_discontinuities": HeuristicCompatibilityHint(
        legacy_metric_name="jump_discontinuities",
        heuristic_id="boundary_discontinuity",
        supported_action_classes=[
            HeuristicActionClass.PRECONDITION,
            HeuristicActionClass.INSERT_CORRECTION,
        ],
        rationale=(
            "Large discontinuities suggest that the boundary entering a downstream "
            "stage is unstable and may need cleanup before the main transformation."
        ),
    ),
    "jump_discontinuity_count": HeuristicCompatibilityHint(
        legacy_metric_name="jump_discontinuity_count",
        heuristic_id="boundary_discontinuity",
        supported_action_classes=[
            HeuristicActionClass.PRECONDITION,
            HeuristicActionClass.INSERT_CORRECTION,
        ],
        rationale="Counts abrupt boundary changes that can confound downstream stages.",
    ),
    "signal_quality_variance": HeuristicCompatibilityHint(
        legacy_metric_name="signal_quality_variance",
        heuristic_id="quality_instability",
        supported_action_classes=[
            HeuristicActionClass.GATE_OR_VALIDATE,
            HeuristicActionClass.BRANCH_AND_COMPARE,
        ],
        rationale=(
            "Large quality variation suggests that one fixed path may be brittle "
            "and a quality-aware gate or alternate branch may be needed."
        ),
    ),
    "signal_quality_kurtosis_cv": HeuristicCompatibilityHint(
        legacy_metric_name="signal_quality_kurtosis_cv",
        heuristic_id="quality_instability",
        supported_action_classes=[
            HeuristicActionClass.GATE_OR_VALIDATE,
            HeuristicActionClass.BRANCH_AND_COMPARE,
        ],
        rationale="Variation in local quality supports explicit quality gating.",
    ),
    "interval_outlier_fraction": HeuristicCompatibilityHint(
        legacy_metric_name="interval_outlier_fraction",
        heuristic_id="interval_instability",
        supported_action_classes=[
            HeuristicActionClass.INSERT_CORRECTION,
            HeuristicActionClass.SMOOTH_OR_AGGREGATE,
        ],
        rationale=(
            "A high fraction of implausible intervals suggests downstream correction "
            "or smoothing is warranted."
        ),
    ),
    "rate_cv": HeuristicCompatibilityHint(
        legacy_metric_name="rate_cv",
        heuristic_id="interval_instability",
        supported_action_classes=[
            HeuristicActionClass.SMOOTH_OR_AGGREGATE,
            HeuristicActionClass.GATE_OR_VALIDATE,
        ],
        rationale="Rate instability often supports smoothing or plausibility gating.",
    ),
    "snr_db": HeuristicCompatibilityHint(
        legacy_metric_name="snr_db",
        heuristic_id="dominant_nuisance_structure",
        supported_action_classes=[
            HeuristicActionClass.PRECONDITION,
            HeuristicActionClass.REPLACE_STAGE,
        ],
        rationale=(
            "Poor separation between relevant and nuisance structure often motivates "
            "a different conditioning stage."
        ),
    ),
    "false_positive_rate": HeuristicCompatibilityHint(
        legacy_metric_name="false_positive_rate",
        heuristic_id="density_collapse",
        supported_action_classes=[
            HeuristicActionClass.GATE_OR_VALIDATE,
            HeuristicActionClass.REPLACE_STAGE,
        ],
        rationale="Spurious output density often motivates validation or stage replacement.",
    ),
    "innovation_whiteness_pvalue": HeuristicCompatibilityHint(
        legacy_metric_name="innovation_whiteness_pvalue",
        heuristic_id="residual_structure_after_transform",
        supported_action_classes=[
            HeuristicActionClass.GATE_OR_VALIDATE,
            HeuristicActionClass.REPLACE_STAGE,
        ],
        rationale="Residual structure implies the current transform is not fully explaining the signal.",
    ),
    "nis_ratio": HeuristicCompatibilityHint(
        legacy_metric_name="nis_ratio",
        heuristic_id="alignment_error",
        supported_action_classes=[
            HeuristicActionClass.GATE_OR_VALIDATE,
            HeuristicActionClass.REPLACE_STAGE,
        ],
        rationale="Mismatch between predicted and observed behavior suggests alignment error.",
    ),
    "accept_rate": HeuristicCompatibilityHint(
        legacy_metric_name="accept_rate",
        heuristic_id="convergence_instability",
        supported_action_classes=[
            HeuristicActionClass.REPLACE_STAGE,
            HeuristicActionClass.INSERT_CORRECTION,
        ],
        rationale="Unstable acceptance behavior often motivates corrective adaptation.",
    ),
    "rhat": HeuristicCompatibilityHint(
        legacy_metric_name="rhat",
        heuristic_id="convergence_instability",
        supported_action_classes=[
            HeuristicActionClass.GATE_OR_VALIDATE,
            HeuristicActionClass.BRANCH_AND_COMPARE,
        ],
        rationale="Convergence instability supports validation and alternate search paths.",
    ),
    "condition_number": HeuristicCompatibilityHint(
        legacy_metric_name="condition_number",
        heuristic_id="numerical_condition_instability",
        supported_action_classes=[
            HeuristicActionClass.PRECONDITION,
            HeuristicActionClass.REPLACE_STAGE,
        ],
        rationale="Numerical instability often motivates preconditioning or stage replacement.",
    ),
    "constraint_violations": HeuristicCompatibilityHint(
        legacy_metric_name="constraint_violations",
        heuristic_id="constraint_violation_risk",
        supported_action_classes=[
            HeuristicActionClass.GATE_OR_VALIDATE,
            HeuristicActionClass.PRECONDITION,
        ],
        rationale="Observed violations support validation or explicit constraint handling.",
    ),
}


def compatibility_hint_for_metric(
    metric_name: str,
    *,
    source_domain: str = "",
) -> HeuristicCompatibilityHint | None:
    """Map one legacy metric/diagnostic name into a canonical heuristic hint."""
    metric_key = str(metric_name or "").strip().lower()
    if not metric_key:
        return None
    hint = _LEGACY_COMPATIBILITY_HINTS.get(metric_key)
    if hint is None:
        return None
    if not source_domain:
        return hint
    return hint.model_copy(update={"source_domain": source_domain})


def canonical_heuristic_from_metric(
    metric_name: str,
    *,
    source_domain: str = "",
    confidence: float = 1.0,
) -> CanonicalHeuristic | None:
    """Build a canonical heuristic skeleton from one legacy metric name."""
    hint = compatibility_hint_for_metric(metric_name, source_domain=source_domain)
    if hint is None:
        return None
    external = {
        item.heuristic_id: item for item in _external_canonical_heuristics()
    }.get(hint.heuristic_id)
    display_name = (
        external.display_name
        if external is not None
        else hint.heuristic_id.replace("_", " ").title()
    )
    meaning = external.dejargonized_meaning if external is not None else hint.rationale
    evidence_type = (
        external.evidence_type
        if external is not None
        else HeuristicEvidenceType.SCALAR_SCORE
    )
    supported_action_classes = (
        list(external.supported_action_classes)
        if external is not None and external.supported_action_classes
        else list(hint.supported_action_classes)
    )
    provenance_requirements = (
        list(external.provenance_requirements)
        if external is not None and external.provenance_requirements
        else ["metric_name", "metric_value", "threshold"]
    )
    compatibility_aliases = list(
        dict.fromkeys(
            [
                *(list(external.compatibility_aliases) if external is not None else []),
                hint.legacy_metric_name,
            ]
        )
    )
    return CanonicalHeuristic(
        heuristic_id=hint.heuristic_id,
        display_name=display_name,
        dejargonized_meaning=meaning,
        evidence_type=evidence_type,
        value_kind="score",
        value_shape="scalar",
        confidence=confidence,
        producer_kind=HeuristicProducerKind.COMPATIBILITY_MAPPING,
        applicability_scope=HeuristicApplicabilityScope.CROSS_FAMILY,
        supported_action_classes=supported_action_classes,
        provenance_requirements=provenance_requirements,
        compatibility_aliases=compatibility_aliases,
        family_notes=[f"source_domain:{source_domain}"] if source_domain else [],
    )


def known_heuristic_compatibility_hints() -> tuple[HeuristicCompatibilityHint, ...]:
    """Return the built-in compatibility hints in a stable order."""
    return tuple(
        _LEGACY_COMPATIBILITY_HINTS[key]
        for key in sorted(_LEGACY_COMPATIBILITY_HINTS.keys())
    )


def known_heuristic_ids() -> tuple[str, ...]:
    """Return the canonical heuristic identifiers supported by the shared layer."""
    ids = {item.heuristic_id for item in _external_canonical_heuristics()}
    ids.update(_EXPLICIT_CANONICAL_HEURISTIC_IDS)
    ids.update(hint.heuristic_id for hint in _LEGACY_COMPATIBILITY_HINTS.values())
    return tuple(sorted(ids))
