"""Auditable local expansion-family assets with a runtime compatibility layer."""

from __future__ import annotations

import json
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field

from sciona.architect.handoff import CDGExport
from sciona.architect.semantic_graph import project_semantic_cdg
from sciona.principal.expansion import (
    ExpansionContext,
    ExpansionDiagnostic,
    ExpansionRuleSet,
)


ASSET_DIR = Path(__file__).resolve().parent / "assets" / "expansions"


class ExpansionReference(BaseModel):
    """Human-reviewable reference for an expansion family asset."""

    title: str
    citation: str = ""
    url: str = ""
    note: str = ""


class ExpansionTriggerAsset(BaseModel):
    """Structured trigger metadata for one expansion operation."""

    metric_name: str = ""
    comparison: Literal["gt", "gte", "lt", "lte", "eq"] = Field(
        default="gt",
        validation_alias=AliasChoices("comparison", "comparator"),
    )
    threshold: float = 0.0
    required_signal_keys: list[str] = Field(default_factory=list)
    required_intermediate_keys: list[str] = Field(default_factory=list)
    required_primitives: list[str] = Field(default_factory=list)
    required_root_inputs: list[str] = Field(default_factory=list)
    required_adjacencies: list[tuple[str, str]] = Field(default_factory=list)
    required_planning_constraint_categories: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "required_planning_constraint_categories",
            "required_planning_terms",
        ),
    )


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
) -> dict[str, str]:
    """Return the compact runtime identity for one operation."""
    return {
        "asset_id": asset.asset_id,
        "asset_version": asset.asset_version,
        "asset_family": asset.family,
        "asset_review_status": asset.audit.review_status,
        "asset_source_kind": asset.audit.source_kind,
        "asset_operation": operation.rule_name,
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


def _operation_matches(
    operation: ExpansionOperationAsset,
    cdg: CDGExport,
    context: ExpansionContext,
) -> bool:
    if any(
        key not in (context.signal_data or {})
        for key in operation.trigger.required_signal_keys
    ):
        return False
    if any(
        key not in (context.intermediates or {})
        for key in operation.trigger.required_intermediate_keys
    ):
        return False
    if operation.trigger.required_planning_constraint_categories:
        categories = _planning_constraint_categories(context)
        required = {
            str(category)
            for category in operation.trigger.required_planning_constraint_categories
        }
        if not required.issubset(categories):
            return False
    if operation.trigger.required_primitives:
        primitives = {
            str(getattr(node, "matched_primitive", "") or "")
            for node in cdg.nodes
        }
        required = {primitive for primitive in operation.trigger.required_primitives}
        if not required.issubset(primitives):
            return False
    if operation.trigger.required_root_inputs or operation.trigger.required_adjacencies:
        semantic = project_semantic_cdg(cdg)
        if operation.trigger.required_root_inputs:
            for root_input in operation.trigger.required_root_inputs:
                if not semantic.find_root_input_consumers(root_input):
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
