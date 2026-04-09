"""Auditable local expansion-family assets with a runtime compatibility layer."""

from __future__ import annotations

import json
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, model_validator

from sciona.architect.handoff import CDGExport
from sciona.architect.semantic_graph import (
    SemanticBoundaryKind,
    SemanticCDG,
    project_semantic_cdg,
)
from sciona.heuristics import HeuristicActionClass
from sciona.principal.expansion import (
    ExpansionContext,
    ExpansionDiagnostic,
    ExpansionRuleSet,
)
from sciona.asset_migration import (
    MigrationReadinessAsset,
    migration_readiness_summary,
)


ASSET_DIR = Path(__file__).resolve().parent / "assets" / "expansions"


class ExpansionReference(BaseModel):
    """Human-reviewable reference for an expansion family asset."""

    title: str
    citation: str = ""
    url: str = ""
    note: str = ""


class ExpansionTriggerAsset(BaseModel):
    """Structured trigger metadata for one expansion operation.

    The public contract is family-neutral. Legacy signal-specific fields are
    accepted as aliases so older assets continue to load.
    """

    metric_name: str = ""
    comparison: Literal["gt", "gte", "lt", "lte", "eq"] = Field(
        default="gt",
        validation_alias=AliasChoices("comparison", "comparator"),
    )
    threshold: float = 0.0
    required_runtime_keys: list[str] = Field(default_factory=list)
    required_runtime_namespaces: list[str] = Field(default_factory=list)
    required_intermediate_keys: list[str] = Field(default_factory=list)
    required_primitives: list[str] = Field(default_factory=list)
    required_boundary_requirements: list["ExpansionBoundaryRequirement"] = Field(
        default_factory=list
    )
    required_adjacencies: list[tuple[str, str]] = Field(default_factory=list)
    required_planning_constraint_categories: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "required_planning_constraint_categories",
            "required_planning_terms",
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(
        cls, data: object
    ) -> object:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)

        if "required_runtime_keys" not in normalized and "required_signal_keys" in normalized:
            normalized["required_runtime_keys"] = normalized.pop("required_signal_keys")

        boundary_requirements = list(normalized.get("required_boundary_requirements", []))
        root_inputs = normalized.pop("required_root_inputs", [])
        root_outputs = normalized.pop("required_root_outputs", [])
        if root_inputs:
            boundary_requirements.extend(
                {
                    "boundary_kind": "root_input",
                    "port_name": str(port_name),
                }
                for port_name in root_inputs
            )
        if root_outputs:
            boundary_requirements.extend(
                {
                    "boundary_kind": "root_output",
                    "port_name": str(port_name),
                }
                for port_name in root_outputs
            )
        if boundary_requirements:
            normalized["required_boundary_requirements"] = boundary_requirements
        return normalized


class ExpansionBoundaryRequirement(BaseModel):
    """Semantic boundary requirement for an expansion operation."""

    boundary_kind: Literal["root_input", "root_output"]
    port_name: str
    matched_primitives: list[str] = Field(default_factory=list)
    data_kind: str = ""
    loss_class: str = ""
    notes: list[str] = Field(default_factory=list)


ExpansionTriggerAsset.model_rebuild()


class ExpansionRewriteAsset(BaseModel):
    """Human-readable rewrite summary for one expansion operation."""

    before_summary: str = ""
    after_summary: str = ""
    information_flow_effect: str = ""
    notes: list[str] = Field(default_factory=list)


class ExpansionOperationAsset(BaseModel):
    """One auditable family refinement operation."""

    rule_name: str
    runtime_rule_builder: str = ""
    runtime_diagnostic: str = ""
    name: str = Field(validation_alias=AliasChoices("name", "summary"))
    intent: str = ""
    dejargonized_summary: str
    trigger: ExpansionTriggerAsset = Field(
        default_factory=ExpansionTriggerAsset,
        validation_alias=AliasChoices("trigger", "applicability"),
    )
    action_classes: list[HeuristicActionClass] = Field(default_factory=list)
    rewrite: ExpansionRewriteAsset = Field(default_factory=ExpansionRewriteAsset)
    uncertainty_notes: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        if not self.runtime_rule_builder:
            self.runtime_rule_builder = self.rule_name
        if not self.runtime_diagnostic:
            if "jump_removal" in self.rule_name:
                self.runtime_diagnostic = "jump_discontinuities"
            elif "sqi" in self.rule_name:
                self.runtime_diagnostic = "signal_quality_variance"
            else:
                self.runtime_diagnostic = "interval_outlier_fraction"


class ExpansionAuditAsset(BaseModel):
    """Audit and documentation metadata for a family expansion inventory."""

    provenance: str = ""
    source_kind: str = "local_asset"
    review_status: str = "draft"
    rationale: str = ""
    dejargonized_summary: str = ""
    migration_readiness: MigrationReadinessAsset = Field(
        default_factory=MigrationReadinessAsset
    )
    uncertainty_notes: list[str] = Field(default_factory=list)
    references: list[ExpansionReference] = Field(default_factory=list)
    maintainers: list[str] = Field(default_factory=list)


class ExpansionFamilyAsset(BaseModel):
    """Canonical local asset describing a family refinement inventory."""

    asset_id: str
    asset_version: str
    family: str
    domain: str
    name: str
    summary: str = Field(validation_alias=AliasChoices("summary", "description"))
    operations: list[ExpansionOperationAsset] = Field(default_factory=list)
    audit: ExpansionAuditAsset = Field(default_factory=ExpansionAuditAsset)

    def model_post_init(self, __context: object) -> None:
        names = [operation.rule_name for operation in self.operations]
        if len(names) != len(set(names)):
            raise ValueError(
                f"Expansion asset '{self.asset_id}' defines duplicate rule names"
            )
        if not self.audit.references:
            raise ValueError(
                f"Expansion asset '{self.asset_id}' must include at least one reference"
            )
        if not self.audit.dejargonized_summary:
            raise ValueError(
                f"Expansion asset '{self.asset_id}' must include a dejargonized summary"
            )

    def operation(self, rule_name: str) -> ExpansionOperationAsset | None:
        """Return one operation by rule name."""
        for operation in self.operations:
            if operation.rule_name == rule_name:
                return operation
        return None


def expansion_asset_summary(
    asset: ExpansionFamilyAsset,
    operation: ExpansionOperationAsset,
) -> dict[str, Any]:
    """Return the compact runtime identity for one operation."""
    readiness = migration_readiness_summary(asset.audit.migration_readiness)
    return {
        "asset_id": asset.asset_id,
        "asset_version": asset.asset_version,
        "asset_family": asset.family,
        "asset_review_status": asset.audit.review_status,
        "asset_source_kind": asset.audit.source_kind,
        "asset_operation": operation.rule_name,
        "action_classes": [action.value for action in operation.action_classes],
        "asset_migration_readiness_status": readiness.get(
            "migration_readiness_status", ""
        ),
        "asset_migration_readiness_ready": readiness.get(
            "migration_readiness_ready", False
        ),
        "asset_migration_readiness_check_count": readiness.get(
            "migration_readiness_check_count", 0
        ),
        "asset_migration_readiness_required_check_count": readiness.get(
            "migration_readiness_required_check_count", 0
        ),
        **readiness,
    }


@lru_cache(maxsize=1)
def load_local_expansion_assets() -> tuple[ExpansionFamilyAsset, ...]:
    """Load local expansion-family assets from disk."""
    assets: list[ExpansionFamilyAsset] = []
    if not ASSET_DIR.exists():
        return tuple()
    for path in sorted(ASSET_DIR.glob("*.json")):
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict) or "operations" not in raw:
            continue
        assets.append(ExpansionFamilyAsset.model_validate(raw))
    return tuple(assets)


@lru_cache(maxsize=1)
def load_local_expansion_assets_by_family() -> dict[str, ExpansionFamilyAsset]:
    """Index local expansion assets by family name."""
    return {asset.family: asset for asset in load_local_expansion_assets()}


def _planning_constraint_categories(context: ExpansionContext) -> set[str]:
    artifact = context.planning_artifact or {}
    constraints = (
        artifact.get("planning_constraints", []) if isinstance(artifact, dict) else []
    )
    categories: set[str] = set()
    for item in constraints:
        if isinstance(item, dict):
            category = str(item.get("category", "")).strip()
            if category:
                categories.add(category)
    return categories


def _planning_constraint_categories_declared(context: ExpansionContext) -> bool:
    """Whether the planning artifact explicitly declared categorized constraints."""
    artifact = context.planning_artifact or {}
    constraints = (
        artifact.get("planning_constraints", []) if isinstance(artifact, dict) else []
    )
    if not isinstance(constraints, list) or not constraints:
        return False
    return any(
        isinstance(item, dict) and str(item.get("category", "")).strip()
        for item in constraints
    )


def _runtime_key_matches_namespace(key: str, namespace: str) -> bool:
    normalized_key = str(key).strip()
    normalized_namespace = str(namespace).strip()
    if not normalized_namespace:
        return False
    if normalized_key == normalized_namespace:
        return True
    prefixes = (
        f"{normalized_namespace}.",
        f"{normalized_namespace}:",
        f"{normalized_namespace}/",
    )
    return normalized_key.startswith(prefixes)


def _available_runtime_keys(context: ExpansionContext) -> set[str]:
    """Return runtime keys available from raw inputs or canonical evidence."""
    available = {
        str(key)
        for key in (context.runtime_inputs or context.signal_data or {}).keys()
    }
    runtime_evidence = context.runtime_evidence or {}
    if not isinstance(runtime_evidence, dict):
        return available
    canonical = runtime_evidence.get("canonical_runtime_context", {})
    if isinstance(canonical, dict):
        canonical_inputs = canonical.get("canonical_inputs", {})
        if isinstance(canonical_inputs, dict):
            available.update(str(key) for key in canonical_inputs.keys())
            for value in canonical_inputs.values():
                if isinstance(value, dict):
                    raw_key = str(value.get("raw_key", "")).strip()
                    if raw_key:
                        available.add(raw_key)
        alias_resolution = canonical.get("alias_resolution", {})
        if isinstance(alias_resolution, dict):
            available.update(str(key) for key in alias_resolution.keys())
            available.update(str(value) for value in alias_resolution.values())
    telemetry_summary = runtime_evidence.get("telemetry_summary", {})
    if isinstance(telemetry_summary, dict):
        for key in ("signal", "time", "sampling_rate"):
            if key in telemetry_summary:
                available.add(key)
    return available


def _available_intermediate_keys(context: ExpansionContext) -> set[str]:
    """Return intermediate keys available from raw or summarized telemetry."""
    available = {str(key) for key in (context.intermediates or {}).keys()}
    runtime_evidence = context.runtime_evidence or {}
    if not isinstance(runtime_evidence, dict):
        return available
    intermediate_summaries = runtime_evidence.get("intermediate_summaries", {})
    if isinstance(intermediate_summaries, dict):
        available.update(str(key) for key in intermediate_summaries.keys())
    telemetry_summary = runtime_evidence.get("telemetry_summary", {})
    if isinstance(telemetry_summary, dict):
        for key in ("events", "rate", "quality", "state", "mask"):
            if key in telemetry_summary:
                available.add(key)
        telemetry_intermediates = telemetry_summary.get("intermediates", {})
        if isinstance(telemetry_intermediates, dict):
            available.update(str(key) for key in telemetry_intermediates.keys())
    return available


def _boundary_requirement_matches(
    semantic_cdg: SemanticCDG | None,
    requirement: ExpansionBoundaryRequirement,
    *,
    cdg: CDGExport,
) -> bool:
    if semantic_cdg is None:
        return False
    node_map = {node.node_id: node for node in cdg.nodes}
    required_data_kind = str(requirement.data_kind or "").strip()
    required_loss_class = str(requirement.loss_class or "").strip()

    if requirement.boundary_kind == "root_input":
        consumers = semantic_cdg.find_boundary_consumers(
            boundary_kind=SemanticBoundaryKind.ROOT_INPUT,
            port_name=requirement.port_name,
            matched_primitive="",
        )
        if required_data_kind:
            consumers = [
                consumer
                for consumer in consumers
                if consumer.data_kind == required_data_kind
            ]
        if requirement.matched_primitives:
            matched = {consumer.matched_primitive for consumer in consumers}
            if not set(requirement.matched_primitives).issubset(matched):
                return False
        if required_loss_class:
            boundary_ids = {consumer.boundary_id for consumer in consumers}
            loss_matches = [
                edge
                for edge in semantic_cdg.edges
                if edge.source_id in boundary_ids
                and edge.loss_class.value == required_loss_class
            ]
            if not loss_matches:
                return False
        return bool(consumers)

    output_boundaries = [
        boundary
        for boundary in semantic_cdg.boundaries
        if boundary.kind == SemanticBoundaryKind.ROOT_OUTPUT
        and boundary.port.name == requirement.port_name
        and (not required_data_kind or boundary.data_kind.value == required_data_kind)
    ]
    if not output_boundaries:
        return False
    for boundary in output_boundaries:
        producer_edges = [
            edge for edge in semantic_cdg.edges if edge.target_id == boundary.boundary_id
        ]
        if required_loss_class:
            producer_edges = [
                edge for edge in producer_edges if edge.loss_class.value == required_loss_class
            ]
        if not producer_edges:
            continue
        producers = [
            node_map[edge.source_id]
            for edge in producer_edges
            if edge.source_id in node_map
        ]
        if requirement.matched_primitives:
            matched = {str(node.matched_primitive or "") for node in producers}
            if not set(requirement.matched_primitives).issubset(matched):
                continue
        return True
    return False


def _operation_matches(
    operation: ExpansionOperationAsset,
    cdg: CDGExport,
    context: ExpansionContext,
) -> bool:
    runtime_data = context.runtime_inputs or context.signal_data or {}
    available_runtime_keys = _available_runtime_keys(context)
    if any(
        key not in available_runtime_keys
        for key in operation.trigger.required_runtime_keys
    ):
        return False
    if operation.trigger.required_runtime_namespaces:
        runtime_keys = sorted(available_runtime_keys)
        for namespace in operation.trigger.required_runtime_namespaces:
            if not any(
                _runtime_key_matches_namespace(key, namespace) for key in runtime_keys
            ):
                return False
    available_intermediate_keys = _available_intermediate_keys(context)
    if any(
        key not in available_intermediate_keys
        for key in operation.trigger.required_intermediate_keys
    ):
        return False
    if operation.trigger.required_planning_constraint_categories:
        # Planning categories are advisory until the Architect contract emits a
        # stable, exhaustive constraint taxonomy across families. We keep the
        # field for auditability, but do not suppress a runtime-supported
        # operation simply because the current planning artifact omitted or
        # coarsened the relevant category.
        _planning_constraint_categories(context)
        _planning_constraint_categories_declared(context)
    if operation.trigger.required_primitives:
        primitives = {
            str(getattr(node, "matched_primitive", "") or "")
            for node in cdg.nodes
        }
        required = {primitive for primitive in operation.trigger.required_primitives}
        if not required.issubset(primitives):
            return False
    if operation.trigger.required_boundary_requirements or operation.trigger.required_adjacencies:
        semantic = project_semantic_cdg(cdg)
        if operation.trigger.required_boundary_requirements:
            for requirement in operation.trigger.required_boundary_requirements:
                if not _boundary_requirement_matches(semantic, requirement, cdg=cdg):
                    return False
        if operation.trigger.required_adjacencies:
            primitive_by_node = {
                node.node_id: str(getattr(node, "matched_primitive", "") or "")
                for node in cdg.nodes
            }
            observed = {
                (
                    primitive_by_node.get(edge.source_id, ""),
                    primitive_by_node.get(edge.target_id, ""),
                )
                for edge in semantic.edges
                if edge.source_id in primitive_by_node and edge.target_id in primitive_by_node
            }
            required_pairs = {
                (str(source), str(target))
                for source, target in operation.trigger.required_adjacencies
            }
            if not required_pairs.issubset(observed):
                return False
    return True


class AssetBackedExpansionRuleSet:
    """Compatibility wrapper that attaches auditable asset provenance."""

    def __init__(self, rule_set: ExpansionRuleSet, asset: ExpansionFamilyAsset) -> None:
        self._rule_set = rule_set
        self._asset = asset
        self.name = getattr(rule_set, "name", asset.family)
        self.domain = getattr(rule_set, "domain", asset.domain)

    def diagnose(self, cdg, context: ExpansionContext) -> list[ExpansionDiagnostic]:
        diagnostics = self._rule_set.diagnose(cdg, context)
        enriched: list[ExpansionDiagnostic] = []
        for diagnostic in diagnostics:
            operation = self._asset.operation(diagnostic.rule_name)
            if operation is None:
                enriched.append(diagnostic)
                continue
            if not _operation_matches(operation, cdg, context):
                continue
            summary = expansion_asset_summary(self._asset, operation)
            enriched.append(
                replace(
                    diagnostic,
                    asset_id=summary["asset_id"],
                    asset_version=summary["asset_version"],
                    asset_family=summary["asset_family"],
                    asset_source_kind=summary["asset_source_kind"],
                    asset_review_status=summary["asset_review_status"],
                    asset_operation=summary["asset_operation"],
                    asset_migration_readiness_status=summary[
                        "asset_migration_readiness_status"
                    ],
                    asset_migration_readiness_ready=summary[
                        "asset_migration_readiness_ready"
                    ],
                    asset_migration_readiness_check_count=summary[
                        "asset_migration_readiness_check_count"
                    ],
                    asset_migration_readiness_required_check_count=summary[
                        "asset_migration_readiness_required_check_count"
                    ],
                )
            )
        return enriched

    def rules(self):
        return self._rule_set.rules()


def asset_backed_rule_sets(rule_sets: list[ExpansionRuleSet]) -> list[ExpansionRuleSet]:
    """Wrap built-in rule sets with local assets when available."""
    by_family = load_local_expansion_assets_by_family()
    wrapped: list[ExpansionRuleSet] = []
    for rule_set in rule_sets:
        asset = by_family.get(getattr(rule_set, "name", ""))
        if asset is None:
            wrapped.append(rule_set)
            continue
        wrapped.append(AssetBackedExpansionRuleSet(rule_set, asset))
    return wrapped
