"""Wave 0 staging contracts for physics ingestion rows.

Source adapters produce ordinary dictionaries so they can stay side-effect
free. This module validates those dictionaries against the Wave 0 SQL shape
before a loader inserts them into Supabase.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SOURCE_SYSTEMS = frozenset(
    {
        "wikidata",
        "qudt",
        "physics_derivation_graph",
        "nist_codata",
        "nist_dlmf",
        "hitran",
        "materials_project",
        "opb",
        "theoria",
        "phy_srbench",
        "manual",
    }
)

RAW_FORMULA_FORMATS = frozenset(
    {
        "",
        "latex",
        "mathml",
        "content_mathml",
        "wikidata_math",
        "asciimath",
        "sympy",
        "plain_text",
    }
)

CANDIDATE_STATUSES = frozenset(
    {
        "raw_imported",
        "parse_failed",
        "parsed",
        "dimension_resolved",
        "symbolically_validated",
        "source_verified",
        "human_reviewed",
        "published",
        "blocked",
    }
)

EXPRESSION_KINDS = frozenset(
    {
        "equation",
        "identity",
        "inequality",
        "ode",
        "pde",
        "constraint",
        "definition",
    }
)

EXPRESSION_ROLES = frozenset(
    {
        "primary",
        "auxiliary",
        "constraint",
        "assumption",
        "compiled_output",
    }
)

PARSE_STATUSES = frozenset(
    {
        "raw_imported",
        "parse_failed",
        "parsed",
        "normalized",
        "blocked",
    }
)

REVIEW_STATUSES = frozenset(
    {
        "unreviewed",
        "automated_pass",
        "needs_human",
        "human_reviewed",
        "blocked",
    }
)

VALIDATION_STATUSES = frozenset({"unknown", "passed", "failed", "skipped"})

RELATIONSHIP_KINDS = frozenset(
    {
        "same_math_topology_as",
        "physical_grounding_of",
        "derives_from",
        "limit_case_of",
        "approximation_of",
        "uses_constant",
        "uses_data_artifact",
        "has_use",
        "mechanism_analogue_of",
        "algebraic_rearrangement_of",
        "requires_assumption",
        "replaces_outside_regime",
    }
)

RELATIONSHIP_SOURCE_KINDS = frozenset(
    {
        "manual",
        "wikidata",
        "physics_derivation_graph",
        "qudt",
        "nist",
        "llm_assisted",
        "automated",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Wave0ContractError(ValueError):
    """Raised when staged rows cannot satisfy the Wave 0 insert contract."""


class _Wave0Row(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_insert_dict(self) -> dict[str, Any]:
        """Return a JSON-ready DB insert dictionary without unset UUID fields."""

        return self.model_dump(mode="json", exclude_none=True)


class PhysicsIngestSnapshotRow(_Wave0Row):
    """Insert contract for ``physics_ingest_snapshots``."""

    snapshot_id: str | None = None
    source_system: str
    source_version: str = ""
    source_uri: str = ""
    retrieved_at: str | None = None
    adapter_name: str = ""
    adapter_version: str = ""
    license_expression: str = ""
    provenance_summary: str = ""
    payload_sha256: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None

    @field_validator("snapshot_id")
    @classmethod
    def _optional_uuid(cls, value: str | None) -> str | None:
        return _validate_optional_uuid(value)

    @field_validator("source_system")
    @classmethod
    def _source_system(cls, value: str) -> str:
        return _validate_member(value, SOURCE_SYSTEMS, "source_system")

    @field_validator("payload_sha256")
    @classmethod
    def _payload_sha256(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("payload_sha256 must be a lowercase SHA-256 hex digest")
        return value


class PhysicsEquationCandidateRow(_Wave0Row):
    """Insert contract for ``physics_equation_candidates``."""

    candidate_id: str | None = None
    snapshot_id: str
    source_candidate_id: str = ""
    source_entity_uri: str = ""
    source_label: str = ""
    source_description: str = ""
    raw_formula: str = ""
    raw_formula_format: str = ""
    candidate_status: str = "raw_imported"
    parse_confidence: float = 0.0
    priority_score: float = 0.0
    mechanism_tags: list[str] = Field(default_factory=list)
    behavioral_archetypes: list[str] = Field(default_factory=list)
    source_payload: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    created_at: str | None = None
    updated_at: str | None = None

    @field_validator("candidate_id")
    @classmethod
    def _optional_candidate_uuid(cls, value: str | None) -> str | None:
        return _validate_optional_uuid(value)

    @field_validator("snapshot_id")
    @classmethod
    def _snapshot_uuid(cls, value: str) -> str:
        return _validate_uuid(value, "snapshot_id")

    @field_validator("raw_formula_format")
    @classmethod
    def _formula_format(cls, value: str) -> str:
        return _validate_member(value, RAW_FORMULA_FORMATS, "raw_formula_format")

    @field_validator("candidate_status")
    @classmethod
    def _candidate_status(cls, value: str) -> str:
        return _validate_member(value, CANDIDATE_STATUSES, "candidate_status")

    @field_validator("parse_confidence")
    @classmethod
    def _parse_confidence(cls, value: float) -> float:
        return _validate_unit_interval(value, "parse_confidence")

    @field_validator("mechanism_tags", "behavioral_archetypes")
    @classmethod
    def _string_list(cls, value: list[str]) -> list[str]:
        return _validate_string_list(value)


class SymbolicExpressionRow(_Wave0Row):
    """Insert contract for ``artifact_symbolic_expressions``."""

    expression_id: str | None = None
    artifact_id: str
    version_id: str
    candidate_id: str | None = None
    expression_kind: str
    expression_role: str = "primary"
    sympy_srepr: str = ""
    canonical_expr_hash: str = ""
    topology_hash: str = ""
    dimensional_hash: str = ""
    raw_formula: str = ""
    raw_formula_format: str = ""
    source_expression_id: str = ""
    parse_status: str = "raw_imported"
    parse_confidence: float = 0.0
    review_status: str = "unreviewed"
    validation_status: str = "unknown"
    mechanism_tags: list[str] = Field(default_factory=list)
    behavioral_archetypes: list[str] = Field(default_factory=list)
    assumptions_json: dict[str, Any] = Field(default_factory=dict)
    evidence_json: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None

    @field_validator("expression_id", "candidate_id")
    @classmethod
    def _optional_id(cls, value: str | None) -> str | None:
        return _validate_optional_uuid(value)

    @field_validator("artifact_id", "version_id")
    @classmethod
    def _required_id(cls, value: str, info: Any) -> str:
        return _validate_uuid(value, info.field_name)

    @field_validator("expression_kind")
    @classmethod
    def _expression_kind(cls, value: str) -> str:
        return _validate_member(value, EXPRESSION_KINDS, "expression_kind")

    @field_validator("expression_role")
    @classmethod
    def _expression_role(cls, value: str) -> str:
        return _validate_member(value, EXPRESSION_ROLES, "expression_role")

    @field_validator("raw_formula_format")
    @classmethod
    def _formula_format(cls, value: str) -> str:
        return _validate_member(value, RAW_FORMULA_FORMATS, "raw_formula_format")

    @field_validator("parse_status")
    @classmethod
    def _parse_status(cls, value: str) -> str:
        return _validate_member(value, PARSE_STATUSES, "parse_status")

    @field_validator("review_status")
    @classmethod
    def _review_status(cls, value: str) -> str:
        return _validate_member(value, REVIEW_STATUSES, "review_status")

    @field_validator("validation_status")
    @classmethod
    def _validation_status(cls, value: str) -> str:
        return _validate_member(value, VALIDATION_STATUSES, "validation_status")

    @field_validator("parse_confidence")
    @classmethod
    def _parse_confidence(cls, value: float) -> float:
        return _validate_unit_interval(value, "parse_confidence")

    @field_validator("canonical_expr_hash", "topology_hash", "dimensional_hash")
    @classmethod
    def _optional_hash(cls, value: str) -> str:
        return _validate_optional_sha256(value)

    @field_validator("mechanism_tags", "behavioral_archetypes")
    @classmethod
    def _string_list(cls, value: list[str]) -> list[str]:
        return _validate_string_list(value)

    @model_validator(mode="after")
    def _has_expression_payload(self) -> "SymbolicExpressionRow":
        if not self.sympy_srepr and not self.raw_formula:
            raise ValueError("sympy_srepr or raw_formula is required")
        return self


class ArtifactRelationshipRow(_Wave0Row):
    """Insert contract for ``artifact_relationships``."""

    relationship_id: str | None = None
    source_artifact_id: str | None = None
    source_version_id: str | None = None
    source_expression_id: str | None = None
    target_artifact_id: str | None = None
    target_version_id: str | None = None
    target_expression_id: str | None = None
    relationship_kind: str
    relationship_label: str = ""
    source_node_id: str = ""
    target_node_id: str = ""
    inference_rule_id: str = ""
    binding_metadata: dict[str, Any] = Field(default_factory=dict)
    assumptions_json: dict[str, Any] = Field(default_factory=dict)
    evidence_json: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    source_kind: str = "manual"
    verified: bool = False
    created_at: str | None = None

    @field_validator(
        "relationship_id",
        "source_artifact_id",
        "source_version_id",
        "source_expression_id",
        "target_artifact_id",
        "target_version_id",
        "target_expression_id",
    )
    @classmethod
    def _optional_uuid_id(cls, value: str | None) -> str | None:
        return _validate_optional_uuid(value)

    @field_validator("relationship_kind")
    @classmethod
    def _relationship_kind(cls, value: str) -> str:
        return _validate_member(value, RELATIONSHIP_KINDS, "relationship_kind")

    @field_validator("source_kind")
    @classmethod
    def _source_kind(cls, value: str) -> str:
        return _validate_member(value, RELATIONSHIP_SOURCE_KINDS, "source_kind")

    @field_validator("confidence")
    @classmethod
    def _confidence(cls, value: float) -> float:
        return _validate_unit_interval(value, "confidence")

    @model_validator(mode="after")
    def _has_endpoints(self) -> "ArtifactRelationshipRow":
        if not (
            self.source_artifact_id
            or self.source_version_id
            or self.source_expression_id
        ):
            raise ValueError("a source artifact, version, or expression endpoint is required")
        if not (
            self.target_artifact_id
            or self.target_version_id
            or self.target_expression_id
        ):
            raise ValueError("a target artifact, version, or expression endpoint is required")
        return self


def attach_snapshot_id(
    candidate_rows: Iterable[Mapping[str, Any]],
    snapshot_id: str,
    *,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Attach a DB-generated snapshot id to candidate row dictionaries.

    Adapters may omit ``snapshot_id`` until the loader inserts the corresponding
    snapshot. This helper returns copies and fails if an existing id disagrees
    unless ``overwrite`` is explicit.
    """

    valid_snapshot_id = _validate_uuid(snapshot_id, "snapshot_id")
    attached: list[dict[str, Any]] = []
    for row in candidate_rows:
        copied = dict(row)
        existing = copied.get("snapshot_id")
        if existing not in (None, "") and existing != valid_snapshot_id and not overwrite:
            raise Wave0ContractError(
                "candidate row already has a different snapshot_id; pass overwrite=True"
            )
        copied["snapshot_id"] = valid_snapshot_id
        attached.append(copied)
    return attached


def validate_snapshot_row(row: Mapping[str, Any]) -> PhysicsIngestSnapshotRow:
    """Validate one ``physics_ingest_snapshots`` row."""

    return PhysicsIngestSnapshotRow.model_validate(dict(row))


def validate_candidate_row(row: Mapping[str, Any]) -> PhysicsEquationCandidateRow:
    """Validate one ``physics_equation_candidates`` row."""

    return PhysicsEquationCandidateRow.model_validate(dict(row))


def validate_symbolic_expression_row(row: Mapping[str, Any]) -> SymbolicExpressionRow:
    """Validate one ``artifact_symbolic_expressions`` row."""

    return SymbolicExpressionRow.model_validate(dict(row))


def validate_artifact_relationship_row(row: Mapping[str, Any]) -> ArtifactRelationshipRow:
    """Validate one ``artifact_relationships`` row."""

    return ArtifactRelationshipRow.model_validate(dict(row))


def validate_candidate_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[PhysicsEquationCandidateRow]:
    return [validate_candidate_row(row) for row in rows]


def validate_artifact_relationship_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[ArtifactRelationshipRow]:
    return [validate_artifact_relationship_row(row) for row in rows]


def stage_source_rows(
    *,
    snapshot_row: Mapping[str, Any],
    candidate_rows: Iterable[Mapping[str, Any]] = (),
    snapshot_id: str | None = None,
) -> tuple[PhysicsIngestSnapshotRow, list[PhysicsEquationCandidateRow]]:
    """Validate a source snapshot and its candidates for insertion.

    Pass ``snapshot_id`` after inserting the snapshot to validate DB-ready
    candidates. If omitted, candidate rows must already contain ``snapshot_id``.
    """

    snapshot = validate_snapshot_row(snapshot_row)
    rows = list(candidate_rows)
    if snapshot_id is not None:
        rows = attach_snapshot_id(rows, snapshot_id)
    candidates = validate_candidate_rows(rows)
    return snapshot, candidates


def _validate_uuid(value: str, field_name: str) -> str:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a UUID") from exc


def _validate_optional_uuid(value: str | None) -> str | None:
    if value in (None, ""):
        return None
    return _validate_uuid(str(value), "id")


def _validate_member(value: str, allowed: frozenset[str], field_name: str) -> str:
    if value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(f"{field_name} must be one of: {allowed_values}")
    return value


def _validate_unit_interval(value: float, field_name: str) -> float:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return value


def _validate_optional_sha256(value: str) -> str:
    if value and not _SHA256_RE.fullmatch(value):
        raise ValueError("hash must be empty or a lowercase SHA-256 hex digest")
    return value


def _validate_string_list(value: list[str]) -> list[str]:
    if any(not isinstance(item, str) for item in value):
        raise ValueError("array values must be strings")
    return value
