"""Side-effect-free publication manifest loader for symbolic physics atoms."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sciona.physics_ingest.normalization import NormalizedExpressionDraft
from sciona.physics_ingest.staging import (
    SymbolicExpressionRow,
    validate_symbolic_expression_row,
)


_EXPRESSION_NAMESPACE = uuid5(NAMESPACE_URL, "sciona.physics_ingest.publication")
_VARIABLE_NAMESPACE = uuid5(NAMESPACE_URL, "sciona.physics_ingest.publication.variable")
_BOUND_NAMESPACE = uuid5(NAMESPACE_URL, "sciona.physics_ingest.publication.bound")
_VARIABLE_ROLES = frozenset(
    {"input", "output", "parameter", "constant", "state", "intermediate"}
)
_VARIABLE_ROLE_ALIASES = {
    "coordinate": "state",
    "observable": "output",
    "target": "output",
}
_DIMENSION_SOURCES = frozenset({"unknown", "qudt", "source", "manual", "inferred"})
_BOUND_KINDS = frozenset(
    {"domain", "regime", "approximation", "replacement", "assumption"}
)
_BOUND_SCOPES = frozenset({"artifact", "version", "expression", "variable", "edge"})
_BOUND_CONFIDENCES = frozenset({"", "low", "medium", "high"})
_REVIEW_STATUSES = frozenset(
    {"unreviewed", "automated_pass", "needs_human", "human_reviewed", "blocked"}
)


class ArtifactBinding(BaseModel):
    """Resolved artifact/version IDs for one local publication manifest key."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    version_id: str

    @field_validator("artifact_id", "version_id")
    @classmethod
    def _uuid(cls, value: str, info: Any) -> str:
        return _validate_uuid(value, info.field_name)


class SymbolicVariableRow(BaseModel):
    """Insert contract for ``artifact_symbolic_variables`` rows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variable_id: str | None = None
    expression_id: str
    symbol_name: str
    source_symbol: str = ""
    aliases: list[str] = Field(default_factory=list)
    variable_role: str
    quantity_kind_uri: str = ""
    quantity_kind_label: str = ""
    unit_uri: str = ""
    unit_label: str = ""
    dim_signature: str = ""
    dimension_source: str = "source"
    assumptions_json: dict[str, Any] = Field(default_factory=dict)
    evidence_json: dict[str, Any] = Field(default_factory=dict)
    ordinal: int = 0

    @field_validator("variable_id")
    @classmethod
    def _optional_uuid(cls, value: str | None) -> str | None:
        if value in (None, ""):
            return None
        return _validate_uuid(value, "variable_id")

    @field_validator("expression_id")
    @classmethod
    def _required_uuid(cls, value: str, info: Any) -> str:
        return _validate_uuid(value, info.field_name)

    @field_validator("symbol_name")
    @classmethod
    def _symbol_name(cls, value: str) -> str:
        if not value:
            raise ValueError("symbol_name is required")
        return value

    @field_validator("variable_role")
    @classmethod
    def _variable_role(cls, value: str) -> str:
        value = _VARIABLE_ROLE_ALIASES.get(value, value)
        if value not in _VARIABLE_ROLES:
            allowed = ", ".join(sorted(_VARIABLE_ROLES))
            raise ValueError(f"variable_role must be one of: {allowed}")
        return value

    @field_validator("dimension_source")
    @classmethod
    def _dimension_source(cls, value: str) -> str:
        if value not in _DIMENSION_SOURCES:
            allowed = ", ".join(sorted(_DIMENSION_SOURCES))
            raise ValueError(f"dimension_source must be one of: {allowed}")
        return value

    @field_validator("aliases")
    @classmethod
    def _aliases(cls, value: list[str]) -> list[str]:
        return [str(item) for item in value if str(item)]

    def to_insert_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class ValidityBoundRow(BaseModel):
    """Insert contract for ``artifact_validity_bounds`` rows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bound_id: str | None = None
    artifact_id: str
    version_id: str
    expression_id: str
    variable_id: str | None = None
    scope: str = "expression"
    variable_name: str
    bound_kind: str = "domain"
    lower_value: float | None = None
    upper_value: float | None = None
    lower_inclusive: bool = True
    upper_inclusive: bool = True
    unit_uri: str = ""
    dim_signature: str = ""
    regime_label: str = ""
    validity_statement: str = ""
    replacement_artifact_fqdn: str = ""
    evidence_ref_key: str = ""
    confidence: str = ""
    review_status: str = "automated_pass"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bound_id", "variable_id")
    @classmethod
    def _optional_uuid(cls, value: str | None) -> str | None:
        if value in (None, ""):
            return None
        return _validate_uuid(value, "uuid")

    @field_validator("expression_id", "artifact_id", "version_id")
    @classmethod
    def _required_uuid(cls, value: str, info: Any) -> str:
        return _validate_uuid(value, info.field_name)

    @field_validator("variable_name")
    @classmethod
    def _variable_name(cls, value: str) -> str:
        if not value:
            raise ValueError("variable_name is required")
        return value

    @field_validator("bound_kind")
    @classmethod
    def _bound_kind(cls, value: str) -> str:
        if value not in _BOUND_KINDS:
            allowed = ", ".join(sorted(_BOUND_KINDS))
            raise ValueError(f"bound_kind must be one of: {allowed}")
        return value

    @field_validator("scope")
    @classmethod
    def _scope(cls, value: str) -> str:
        if value not in _BOUND_SCOPES:
            allowed = ", ".join(sorted(_BOUND_SCOPES))
            raise ValueError(f"scope must be one of: {allowed}")
        return value

    @field_validator("confidence")
    @classmethod
    def _confidence(cls, value: str) -> str:
        if value not in _BOUND_CONFIDENCES:
            allowed = ", ".join(sorted(_BOUND_CONFIDENCES))
            raise ValueError(f"confidence must be one of: {allowed}")
        return value

    @field_validator("review_status")
    @classmethod
    def _review_status(cls, value: str) -> str:
        if value not in _REVIEW_STATUSES:
            allowed = ", ".join(sorted(_REVIEW_STATUSES))
            raise ValueError(f"review_status must be one of: {allowed}")
        return value

    @model_validator(mode="after")
    def _valid_range(self) -> "ValidityBoundRow":
        if (
            self.lower_value is not None
            and self.upper_value is not None
            and self.lower_value > self.upper_value
        ):
            raise ValueError("lower_value must be <= upper_value")
        return self

    def to_insert_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


@dataclass(frozen=True)
class PublicationDiagnostic:
    """One skipped manifest row or row-level validation error."""

    table: str
    reason: str
    artifact_key: str = ""
    atom_name: str = ""
    severity: str = "skipped"
    detail: str = ""


@dataclass(frozen=True)
class PublicationLoadResult:
    """Validated publication insert rows plus non-fatal diagnostics."""

    artifact_symbolic_expressions: tuple[SymbolicExpressionRow, ...] = ()
    artifact_symbolic_variables: tuple[SymbolicVariableRow, ...] = ()
    artifact_validity_bounds: tuple[ValidityBoundRow, ...] = ()
    diagnostics: tuple[PublicationDiagnostic, ...] = ()

    @property
    def skipped_rows(self) -> tuple[PublicationDiagnostic, ...]:
        return tuple(row for row in self.diagnostics if row.severity == "skipped")

    @property
    def error_rows(self) -> tuple[PublicationDiagnostic, ...]:
        return tuple(row for row in self.diagnostics if row.severity == "error")

    def to_insert_rows(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "artifact_symbolic_expressions": [
                row.to_insert_dict() for row in self.artifact_symbolic_expressions
            ],
            "artifact_symbolic_variables": [
                row.to_insert_dict() for row in self.artifact_symbolic_variables
            ],
            "artifact_validity_bounds": [
                row.to_insert_dict() for row in self.artifact_validity_bounds
            ],
        }


def load_symbolic_publication_manifest(
    manifest: Mapping[str, Any],
    artifact_bindings: Mapping[str, Mapping[str, Any] | ArtifactBinding],
) -> PublicationLoadResult:
    """Resolve and validate symbolic publication rows without database IO.

    ``artifact_bindings`` may be keyed by a manifest row's ``local_artifact_key``,
    ``artifact_key``, ``atom_name``, or ``registry_name``. Each binding must
    provide ``artifact_id`` and ``version_id`` UUIDs. Rows whose artifact cannot
    be bound are skipped with diagnostics; malformed bound rows are reported as
    errors and excluded from insert output.
    """

    bindings, binding_errors = _normalize_bindings(artifact_bindings)
    diagnostics = list(binding_errors)
    expression_by_key: dict[str, SymbolicExpressionRow] = {}
    binding_by_key: dict[str, ArtifactBinding] = {}

    for row in _rows(manifest, "artifact_symbolic_expressions"):
        artifact_key = _text(row, "artifact_key", "local_artifact_key")
        atom_name = _text(row, "atom_name")
        binding = _resolve_binding(row, bindings)
        if binding is None:
            diagnostics.append(
                PublicationDiagnostic(
                    table="artifact_symbolic_expressions",
                    reason="missing_artifact_binding",
                    artifact_key=artifact_key,
                    atom_name=atom_name,
                )
            )
            continue

        expression_id = _text(row, "expression_id") or _stable_expression_id(row, binding)
        insert_row = {
            "expression_id": expression_id,
            "artifact_id": binding.artifact_id,
            "version_id": binding.version_id,
            "expression_kind": _text(row, "expression_kind") or "equation",
            "expression_role": _text(row, "expression_role") or "primary",
            "sympy_srepr": _text(row, "sympy_srepr", "expression_srepr"),
            "canonical_expr_hash": _text(row, "canonical_expr_hash"),
            "topology_hash": _text(row, "topology_hash"),
            "dimensional_hash": _text(row, "dimensional_hash"),
            "raw_formula": _text(row, "raw_formula", "expression_text"),
            "raw_formula_format": _text(row, "raw_formula_format") or "plain_text",
            "source_expression_id": _text(row, "source_expression_id") or artifact_key,
            "parse_status": _text(row, "parse_status") or "normalized",
            "parse_confidence": _number(row.get("parse_confidence"), 1.0),
            "review_status": _text(row, "review_status") or "automated_pass",
            "validation_status": _text(row, "validation_status") or "passed",
            "mechanism_tags": _string_list(row.get("mechanism_tags")),
            "behavioral_archetypes": _string_list(row.get("behavioral_archetypes")),
            "assumptions_json": _dict(row.get("assumptions_json")),
            "evidence_json": {
                **_dict(row.get("evidence_json")),
                "publication_manifest": {
                    "provider": _text(row, "provider") or _text(manifest, "provider"),
                    "atom_name": atom_name,
                    "atom_module": _text(row, "atom_module"),
                    "registry_name": _text(row, "registry_name"),
                    "local_artifact_key": _text(row, "local_artifact_key"),
                    "artifact_uuid": row.get("artifact_uuid"),
                    "constants": _dict(row.get("constants")),
                    "bibliography": list(row.get("bibliography") or []),
                },
            },
        }
        try:
            expression = validate_symbolic_expression_row(insert_row)
        except ValueError as exc:
            diagnostics.append(
                PublicationDiagnostic(
                    table="artifact_symbolic_expressions",
                    reason="validation_error",
                    artifact_key=artifact_key,
                    atom_name=atom_name,
                    severity="error",
                    detail=str(exc),
                )
            )
            continue

        for key in _candidate_keys(row):
            expression_by_key[key] = expression
            binding_by_key[key] = binding

    variables = _load_variables(manifest, expression_by_key, binding_by_key, diagnostics)
    bounds = _load_bounds(manifest, expression_by_key, binding_by_key, diagnostics)
    expressions = _unique_expressions(expression_by_key.values())
    return PublicationLoadResult(
        artifact_symbolic_expressions=expressions,
        artifact_symbolic_variables=tuple(variables),
        artifact_validity_bounds=tuple(bounds),
        diagnostics=tuple(diagnostics),
    )


def build_publication_load_result_from_normalized_drafts(
    drafts: Iterable[NormalizedExpressionDraft | Mapping[str, Any] | Any],
) -> PublicationLoadResult:
    """Convert normalized expression drafts into validated publication rows.

    The helper performs no database IO and intentionally preserves failed or
    ``needs_human`` expression drafts in ``artifact_symbolic_expressions``.
    Variable and bound rows are best-effort: malformed partial metadata is
    reported through diagnostics without dropping the parent expression row.
    """

    expressions: list[SymbolicExpressionRow] = []
    variables: list[SymbolicVariableRow] = []
    bounds: list[ValidityBoundRow] = []
    diagnostics: list[PublicationDiagnostic] = []
    variable_ids_by_expression_symbol: dict[tuple[str, str], str] = {}

    for draft_index, draft in enumerate(drafts):
        try:
            row = _draft_expression_row(draft)
        except (TypeError, ValueError) as exc:
            diagnostics.append(
                PublicationDiagnostic(
                    table="artifact_symbolic_expressions",
                    reason="validation_error",
                    severity="error",
                    detail=str(exc),
                )
            )
            continue

        expression_id = row.expression_id or _stable_draft_expression_id(row, draft_index)
        expression = row.model_copy(update={"expression_id": expression_id})
        try:
            expression = validate_symbolic_expression_row(expression.to_insert_dict())
        except ValueError as exc:
            diagnostics.append(
                PublicationDiagnostic(
                    table="artifact_symbolic_expressions",
                    reason="validation_error",
                    artifact_key=row.source_expression_id,
                    severity="error",
                    detail=str(exc),
                )
            )
            continue
        expressions.append(expression)

        normalized_candidate = _draft_normalized_candidate(draft)
        for ordinal, variable in enumerate(
            _normalized_candidate_variables(normalized_candidate)
        ):
            symbol = _field_text(variable, "symbol", "symbol_name", "name")
            variable_id = _field_text(variable, "variable_id") or str(
                uuid5(_VARIABLE_NAMESPACE, f"{expression_id}:{symbol}")
            )
            try:
                variable_row = SymbolicVariableRow.model_validate(
                    {
                        "variable_id": variable_id,
                        "expression_id": expression_id,
                        "symbol_name": symbol,
                        "source_symbol": _field_text(variable, "source_symbol")
                        or symbol,
                        "aliases": _field_string_list(variable, "aliases", "alias"),
                        "variable_role": _field_text(variable, "role", "variable_role")
                        or "input",
                        "quantity_kind_uri": _field_text(
                            variable,
                            "quantity_kind_uri",
                            "qudt_uri",
                        ),
                        "quantity_kind_label": _field_text(
                            variable,
                            "quantity_kind_label",
                            "quantity_kind",
                        ),
                        "unit_uri": _field_text(variable, "unit_uri"),
                        "unit_label": _field_text(variable, "unit_label", "unit"),
                        "dim_signature": _compact_dim_signature(
                            _field_value(
                                variable,
                                "dim_signature",
                                "dimension",
                                "dimensions",
                            )
                        ),
                        "dimension_source": _field_text(variable, "dimension_source")
                        or "inferred",
                        "assumptions_json": _field_dict(
                            variable,
                            "assumptions_json",
                            "assumptions",
                        ),
                        "evidence_json": {
                            **_field_dict(variable, "evidence_json", "evidence"),
                            "normalized_draft": {
                                "source_expression_id": expression.source_expression_id,
                                "parse_status": expression.parse_status,
                                "review_status": expression.review_status,
                            },
                        },
                        "ordinal": _field_int(variable, "ordinal", default=ordinal),
                    }
                )
            except ValueError as exc:
                diagnostics.append(
                    PublicationDiagnostic(
                        table="artifact_symbolic_variables",
                        reason="validation_error",
                        artifact_key=expression.source_expression_id,
                        severity="error",
                        detail=str(exc),
                    )
                )
                continue
            variables.append(variable_row)
            variable_ids_by_expression_symbol[(expression_id, variable_row.symbol_name)] = (
                variable_row.variable_id or variable_id
            )

        for bound in _normalized_candidate_bounds(normalized_candidate):
            symbol = _field_text(bound, "variable_name", "symbol", "name")
            lower = _field_value(bound, "lower_value", "min_value", "lower", "min")
            upper = _field_value(bound, "upper_value", "max_value", "upper", "max")
            bound_kind = _field_text(bound, "bound_kind", "kind") or "domain"
            bound_id = _field_text(bound, "bound_id") or str(
                uuid5(
                    _BOUND_NAMESPACE,
                    f"{expression_id}:{symbol}:{bound_kind}:{lower}:{upper}:"
                    f"{_field_text(bound, 'validity_statement')}",
                )
            )
            try:
                bounds.append(
                    ValidityBoundRow.model_validate(
                        {
                            "bound_id": bound_id,
                            "artifact_id": expression.artifact_id,
                            "version_id": expression.version_id,
                            "expression_id": expression_id,
                            "variable_id": _field_text(bound, "variable_id")
                            or variable_ids_by_expression_symbol.get(
                                (expression_id, symbol)
                            ),
                            "scope": _field_text(bound, "scope") or "expression",
                            "variable_name": symbol,
                            "bound_kind": bound_kind,
                            "lower_value": lower,
                            "upper_value": upper,
                            "lower_inclusive": _field_value(bound, "lower_inclusive")
                            if _field_value(bound, "lower_inclusive") is not None
                            else True,
                            "upper_inclusive": _field_value(bound, "upper_inclusive")
                            if _field_value(bound, "upper_inclusive") is not None
                            else True,
                            "unit_uri": _field_text(bound, "unit_uri"),
                            "dim_signature": _compact_dim_signature(
                                _field_value(
                                    bound,
                                    "dim_signature",
                                    "dimension",
                                    "dimensions",
                                )
                            ),
                            "regime_label": _field_text(bound, "regime_label"),
                            "validity_statement": _field_text(
                                bound,
                                "validity_statement",
                                "statement",
                            )
                            or _validity_statement(symbol, lower, upper),
                            "replacement_artifact_fqdn": _field_text(
                                bound,
                                "replacement_artifact_fqdn",
                            ),
                            "evidence_ref_key": _field_text(bound, "evidence_ref_key"),
                            "confidence": _field_text(bound, "confidence"),
                            "review_status": _field_text(bound, "review_status")
                            or expression.review_status,
                            "metadata": {
                                **_field_dict(bound, "metadata"),
                                "evidence": _field_dict(bound, "evidence_json", "evidence"),
                                "normalized_draft": {
                                    "source_expression_id": expression.source_expression_id,
                                },
                            },
                        }
                    )
                )
            except ValueError as exc:
                diagnostics.append(
                    PublicationDiagnostic(
                        table="artifact_validity_bounds",
                        reason="validation_error",
                        artifact_key=expression.source_expression_id,
                        severity="error",
                        detail=str(exc),
                    )
                )

    return PublicationLoadResult(
        artifact_symbolic_expressions=tuple(expressions),
        artifact_symbolic_variables=tuple(variables),
        artifact_validity_bounds=tuple(bounds),
        diagnostics=tuple(diagnostics),
    )


def _load_variables(
    manifest: Mapping[str, Any],
    expression_by_key: Mapping[str, SymbolicExpressionRow],
    binding_by_key: Mapping[str, ArtifactBinding],
    diagnostics: list[PublicationDiagnostic],
) -> list[SymbolicVariableRow]:
    variables: list[SymbolicVariableRow] = []
    for row in _rows(manifest, "artifact_symbolic_variables"):
        artifact_key = _text(row, "artifact_key")
        atom_name = _text(row, "atom_name")
        expression = _resolve_expression(row, expression_by_key)
        binding = _resolve_expression(row, binding_by_key)
        if expression is None or binding is None:
            diagnostics.append(
                PublicationDiagnostic(
                    table="artifact_symbolic_variables",
                    reason="missing_expression_binding",
                    artifact_key=artifact_key,
                    atom_name=atom_name,
                )
            )
            continue
        try:
            variables.append(
                SymbolicVariableRow.model_validate(
                    {
                        "expression_id": expression.expression_id,
                        "symbol_name": _text(row, "symbol_name", "symbol"),
                        "source_symbol": _text(row, "source_symbol", "symbol"),
                        "aliases": _list(row.get("aliases")),
                        "variable_role": _text(row, "variable_role", "role"),
                        "quantity_kind_uri": _text(row, "quantity_kind_uri"),
                        "quantity_kind_label": _text(row, "quantity_kind_label"),
                        "unit_uri": _text(row, "unit_uri"),
                        "unit_label": _text(row, "unit_label"),
                        "dim_signature": _text(row, "dim_signature"),
                        "dimension_source": _text(row, "dimension_source")
                        or "source",
                        "assumptions_json": _dict(row.get("assumptions_json")),
                        "evidence_json": {
                            **_dict(row.get("evidence_json")),
                            "publication_manifest": {
                                "provider": _text(row, "provider")
                                or _text(manifest, "provider"),
                                "artifact_key": artifact_key,
                                "atom_name": atom_name,
                                "source_variable_id": _text(row, "source_variable_id")
                                or f"{artifact_key}:{_text(row, 'symbol_name', 'symbol')}",
                                "artifact_id": binding.artifact_id,
                                "version_id": binding.version_id,
                            },
                        },
                        "ordinal": _int(row.get("ordinal"), len(variables)),
                    }
                )
            )
        except ValueError as exc:
            diagnostics.append(
                PublicationDiagnostic(
                    table="artifact_symbolic_variables",
                    reason="validation_error",
                    artifact_key=artifact_key,
                    atom_name=atom_name,
                    severity="error",
                    detail=str(exc),
                )
            )
    return variables


def _load_bounds(
    manifest: Mapping[str, Any],
    expression_by_key: Mapping[str, SymbolicExpressionRow],
    binding_by_key: Mapping[str, ArtifactBinding],
    diagnostics: list[PublicationDiagnostic],
) -> list[ValidityBoundRow]:
    bounds: list[ValidityBoundRow] = []
    for row in _rows(manifest, "artifact_validity_bounds"):
        artifact_key = _text(row, "artifact_key")
        atom_name = _text(row, "atom_name")
        expression = _resolve_expression(row, expression_by_key)
        binding = _resolve_expression(row, binding_by_key)
        if expression is None or binding is None:
            diagnostics.append(
                PublicationDiagnostic(
                    table="artifact_validity_bounds",
                    reason="missing_expression_binding",
                    artifact_key=artifact_key,
                    atom_name=atom_name,
                )
            )
            continue
        lower = row.get("lower_value", row.get("min_value"))
        upper = row.get("upper_value", row.get("max_value"))
        try:
            bounds.append(
                ValidityBoundRow.model_validate(
                    {
                        "expression_id": expression.expression_id,
                        "artifact_id": binding.artifact_id,
                        "version_id": binding.version_id,
                        "variable_id": _text(row, "variable_id") or None,
                        "scope": _text(row, "scope") or "expression",
                        "variable_name": _text(row, "variable_name", "symbol"),
                        "bound_kind": _text(row, "bound_kind") or "domain",
                        "lower_value": lower,
                        "upper_value": upper,
                        "lower_inclusive": row.get("lower_inclusive", True),
                        "upper_inclusive": row.get("upper_inclusive", True),
                        "unit_uri": _text(row, "unit_uri"),
                        "dim_signature": _text(row, "dim_signature"),
                        "regime_label": _text(row, "regime_label"),
                        "validity_statement": _text(row, "validity_statement")
                        or _validity_statement(_text(row, "variable_name", "symbol"), lower, upper),
                        "replacement_artifact_fqdn": _text(
                            row, "replacement_artifact_fqdn"
                        ),
                        "evidence_ref_key": _text(row, "evidence_ref_key"),
                        "confidence": _text(row, "confidence"),
                        "review_status": _text(row, "review_status") or "automated_pass",
                        "metadata": {
                            **_dict(row.get("metadata")),
                            "publication_manifest": {
                                "provider": _text(row, "provider")
                                or _text(manifest, "provider"),
                                "artifact_key": artifact_key,
                                "atom_name": atom_name,
                                "source_bound_id": _text(row, "source_bound_id")
                                or f"{artifact_key}:{_text(row, 'variable_name', 'symbol')}",
                            },
                        },
                    }
                )
            )
        except ValueError as exc:
            diagnostics.append(
                PublicationDiagnostic(
                    table="artifact_validity_bounds",
                    reason="validation_error",
                    artifact_key=artifact_key,
                    atom_name=atom_name,
                    severity="error",
                    detail=str(exc),
                )
            )
    return bounds


def _normalize_bindings(
    bindings: Mapping[str, Mapping[str, Any] | ArtifactBinding],
) -> tuple[dict[str, ArtifactBinding], list[PublicationDiagnostic]]:
    normalized: dict[str, ArtifactBinding] = {}
    diagnostics: list[PublicationDiagnostic] = []
    for key, value in bindings.items():
        try:
            binding = (
                value
                if isinstance(value, ArtifactBinding)
                else ArtifactBinding.model_validate(dict(value))
            )
        except (TypeError, ValueError) as exc:
            diagnostics.append(
                PublicationDiagnostic(
                    table="artifact_bindings",
                    reason="validation_error",
                    artifact_key=str(key),
                    severity="error",
                    detail=str(exc),
                )
            )
            continue
        normalized[str(key)] = binding
    return normalized, diagnostics


def _rows(manifest: Mapping[str, Any], table: str) -> tuple[Mapping[str, Any], ...]:
    return tuple(row for row in manifest.get(table, ()) if isinstance(row, Mapping))


def _resolve_binding(
    row: Mapping[str, Any],
    bindings: Mapping[str, ArtifactBinding],
) -> ArtifactBinding | None:
    return _resolve_expression(row, bindings)


def _resolve_expression(
    row: Mapping[str, Any],
    values: Mapping[str, Any],
) -> Any | None:
    for key in _candidate_keys(row):
        if key in values:
            return values[key]
    return None


def _candidate_keys(row: Mapping[str, Any]) -> tuple[str, ...]:
    keys = []
    for key_name in ("local_artifact_key", "artifact_key", "atom_name", "registry_name"):
        value = _text(row, key_name)
        if value and value not in keys:
            keys.append(value)
    return tuple(keys)


def _stable_expression_id(row: Mapping[str, Any], binding: ArtifactBinding) -> str:
    key = _text(row, "local_artifact_key", "artifact_key", "atom_name", "registry_name")
    return str(uuid5(_EXPRESSION_NAMESPACE, f"{binding.artifact_id}:{binding.version_id}:{key}"))


def _stable_draft_expression_id(row: SymbolicExpressionRow, draft_index: int) -> str:
    key = (
        row.source_expression_id
        or row.candidate_id
        or row.canonical_expr_hash
        or row.raw_formula
        or str(draft_index)
    )
    return str(uuid5(_EXPRESSION_NAMESPACE, f"{row.artifact_id}:{row.version_id}:{key}"))


def _unique_expressions(
    rows: Iterable[SymbolicExpressionRow],
) -> tuple[SymbolicExpressionRow, ...]:
    seen: set[str] = set()
    unique: list[SymbolicExpressionRow] = []
    for row in rows:
        expression_id = row.expression_id or ""
        if expression_id in seen:
            continue
        seen.add(expression_id)
        unique.append(row)
    return tuple(unique)


def _validity_statement(symbol: str, lower: Any, upper: Any) -> str:
    if lower is None and upper is None:
        return ""
    if lower is None:
        return f"{symbol} <= {upper}"
    if upper is None:
        return f"{symbol} >= {lower}"
    return f"{lower} <= {symbol} <= {upper}"


def _validate_uuid(value: str, field_name: str) -> str:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a UUID") from exc


def _text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _list(value) if str(item)]


def _int(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    return int(value)


def _number(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _draft_expression_row(
    draft: NormalizedExpressionDraft | Mapping[str, Any] | Any,
) -> SymbolicExpressionRow:
    row = _field_value(draft, "row")
    if isinstance(row, SymbolicExpressionRow):
        return row
    if isinstance(row, Mapping):
        return validate_symbolic_expression_row(row)
    if isinstance(draft, SymbolicExpressionRow):
        return draft
    if isinstance(draft, Mapping):
        source = draft.get("artifact_symbolic_expression") or draft.get("expression") or draft
        if isinstance(source, SymbolicExpressionRow):
            return source
        if isinstance(source, Mapping):
            return validate_symbolic_expression_row(source)
    raise TypeError("normalized draft must provide a SymbolicExpressionRow row")


def _draft_normalized_candidate(
    draft: NormalizedExpressionDraft | Mapping[str, Any] | Any,
) -> Any:
    return _field_value(draft, "normalized_candidate", "candidate") or {}


def _normalized_candidate_variables(normalized_candidate: Any) -> tuple[Any, ...]:
    variables = _field_value(normalized_candidate, "variables")
    if isinstance(variables, Mapping):
        rows = []
        for symbol, variable in variables.items():
            if isinstance(variable, Mapping):
                rows.append({"symbol": symbol, **dict(variable)})
            else:
                rows.append(variable)
        return tuple(rows)
    if isinstance(variables, list | tuple):
        return tuple(variables)
    return ()


def _normalized_candidate_bounds(normalized_candidate: Any) -> tuple[Any, ...]:
    bounds = _field_value(normalized_candidate, "validity_bounds", "bounds")
    if isinstance(bounds, Mapping):
        rows = []
        for symbol, bound in bounds.items():
            if isinstance(bound, Mapping):
                rows.append({"variable_name": symbol, **dict(bound)})
            else:
                try:
                    lower, upper = bound
                except (TypeError, ValueError):
                    rows.append({"variable_name": symbol})
                else:
                    rows.append(
                        {
                            "variable_name": symbol,
                            "lower_value": lower,
                            "upper_value": upper,
                        }
                    )
        return tuple(rows)
    if isinstance(bounds, list | tuple):
        return tuple(bounds)
    return ()


def _field_value(source: Any, *keys: str) -> Any:
    for key in keys:
        value = getattr(source, key, None)
        if value is not None:
            return value
    if isinstance(source, BaseModel):
        data = source.model_dump()
    elif isinstance(source, Mapping):
        data = source
    else:
        data = None
    if data is not None:
        for key in keys:
            if key in data and data[key] is not None:
                return data[key]
    return None


def _field_text(source: Any, *keys: str) -> str:
    value = _field_value(source, *keys)
    return "" if value is None else str(value)


def _field_dict(source: Any, *keys: str) -> dict[str, Any]:
    value = _field_value(source, *keys)
    return dict(value) if isinstance(value, Mapping) else {}


def _field_string_list(source: Any, *keys: str) -> list[str]:
    return _string_list(_field_value(source, *keys))


def _field_int(source: Any, *keys: str, default: int) -> int:
    value = _field_value(source, *keys)
    return default if value in (None, "") else int(value)


def _compact_dim_signature(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "to_compact"):
        return str(value.to_compact())
    return str(value)
