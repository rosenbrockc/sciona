"""Auditable local skeleton-family assets with a runtime compatibility layer."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, BaseModel, Field

from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
    SkeletonGraph,
)
from sciona.architect.planning_contract import (
    PlanningConstraint,
    PlanningConstraintCategory,
)
from sciona.asset_atom_registry import unknown_registered_atom_references
from sciona.asset_migration import (
    MigrationReadinessAsset,
    migration_readiness_summary,
)


ASSET_DIR = Path(__file__).resolve().parent / "assets" / "skeletons"


class SkeletonReference(BaseModel):
    """Human-reviewable reference for a skeleton family asset."""

    title: str
    citation: str = ""
    url: str = ""
    note: str = ""


class SkeletonStageAsset(BaseModel):
    """One conceptual stage within a skeleton family asset."""

    stage_id: str
    name: str
    description: str
    dejargonized_description: str = ""
    concept_type: ConceptType
    inputs: list[IOSpec] = Field(default_factory=list)
    outputs: list[IOSpec] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    guarantees: list[str] = Field(default_factory=list)
    family_notes: list[str] = Field(default_factory=list)
    matched_primitive: str = ""


class SkeletonEdgeAsset(BaseModel):
    """Typed edge plus semantic expectations between skeleton stages."""

    source_stage_id: str
    target_stage_id: str
    output_name: str
    input_name: str
    source_type: str
    target_type: str
    data_kind: str = ""
    provenance: str = ""
    loss_class: str = "preserving"
    alignment_expectation: str = ""
    notes: list[str] = Field(default_factory=list)


class SkeletonAuditAsset(BaseModel):
    """Minimum viable audit and documentation metadata for a skeleton asset."""

    provenance: str = ""
    source_kind: str = "local_asset"
    review_status: str = "draft"
    rationale: str = ""
    dejargonized_summary: str = ""
    migration_readiness: MigrationReadinessAsset = Field(
        default_factory=MigrationReadinessAsset
    )
    provenance_notes: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    references: list[SkeletonReference] = Field(default_factory=list)
    maintainers: list[str] = Field(default_factory=list)


class SkeletonFamilyAsset(BaseModel):
    """Canonical local asset describing a reusable algorithm-family scaffold."""

    asset_id: str
    asset_version: str
    family: str
    paradigm: ConceptType
    name: str
    summary: str = Field(validation_alias=AliasChoices("summary", "description"))
    dejargonized_summary: str = ""
    canonical_for_paradigm: bool = False
    variant_hints: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("variant_hints", "variant_aliases"),
    )
    stages: list[SkeletonStageAsset] = Field(default_factory=list)
    edges: list[SkeletonEdgeAsset] = Field(default_factory=list)
    planning_constraints: list[PlanningConstraint | str] = Field(default_factory=list)
    audit: SkeletonAuditAsset = Field(default_factory=SkeletonAuditAsset)

    def planning_constraint_models(self) -> list[PlanningConstraint]:
        """Normalize legacy string constraints into typed planning constraints."""
        normalized: list[PlanningConstraint] = []
        for item in self.planning_constraints:
            if isinstance(item, PlanningConstraint):
                normalized.append(item)
                continue
            normalized.append(
                PlanningConstraint(
                    category=PlanningConstraintCategory.STAGE,
                    subject=self.family,
                    statement=str(item),
                    source_stage="skeleton_asset",
                )
            )
        return normalized

    def model_post_init(self, __context: object) -> None:
        stage_ids = {stage.stage_id for stage in self.stages}
        missing = sorted(
            {
                stage_id
                for edge in self.edges
                for stage_id in (edge.source_stage_id, edge.target_stage_id)
                if stage_id not in stage_ids
            }
        )
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(
                f"Skeleton asset references unknown stage ids: {missing_str}"
            )
        if not self.audit.references:
            raise ValueError(
                f"Skeleton asset '{self.asset_id}' must include at least one reference"
            )
        if not (self.dejargonized_summary or self.audit.dejargonized_summary):
            raise ValueError(
                f"Skeleton asset '{self.asset_id}' must include a dejargonized summary"
            )

    def to_skeleton_graph(self) -> SkeletonGraph:
        """Convert the asset into the runtime skeleton graph format."""
        stage_nodes = [
            AlgorithmicNode(
                node_id=stage.stage_id,
                name=stage.name,
                description=stage.description,
                concept_type=stage.concept_type,
                inputs=[port.model_copy(deep=True) for port in stage.inputs],
                outputs=[port.model_copy(deep=True) for port in stage.outputs],
                status=NodeStatus.PENDING,
                depth=1,
                matched_primitive=stage.matched_primitive or None,
            )
            for stage in self.stages
        ]
        stage_lookup = {node.node_id for node in stage_nodes}
        edges = [
            DependencyEdge(
                source_id=edge.source_stage_id,
                target_id=edge.target_stage_id,
                output_name=edge.output_name,
                input_name=edge.input_name,
                source_type=edge.source_type,
                target_type=edge.target_type,
            )
            for edge in self.edges
            if edge.source_stage_id in stage_lookup
            and edge.target_stage_id in stage_lookup
        ]
        return SkeletonGraph(
            paradigm=self.paradigm,
            name=self.name,
            description=self.summary,
            template_nodes=stage_nodes,
            template_edges=edges,
            variants=list(self.variant_hints),
            metadata={
                "asset": skeleton_asset_summary(self),
                "audit": self.audit.model_dump(mode="json"),
                "edge_semantics": [
                    edge.model_dump(mode="json") for edge in self.edges
                ],
                "stage_docs": [
                    {
                        "stage_id": stage.stage_id,
                        "name": stage.name,
                        "dejargonized_description": stage.dejargonized_description,
                        "preconditions": list(stage.preconditions),
                        "guarantees": list(stage.guarantees),
                        "family_notes": list(stage.family_notes),
                        "matched_primitive": stage.matched_primitive,
                    }
                    for stage in self.stages
                ],
                "planning_constraints": [
                    constraint.model_dump(mode="json")
                    for constraint in self.planning_constraint_models()
                ],
                "dejargonized_summary": (
                    self.dejargonized_summary or self.audit.dejargonized_summary
                ),
            },
        )


def skeleton_asset_summary(
    asset: SkeletonFamilyAsset | SkeletonGraph | dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the compact identity payload used at runtime."""
    if asset is None:
        return {}
    if isinstance(asset, SkeletonFamilyAsset):
        return {
            "asset_id": asset.asset_id,
            "asset_version": asset.asset_version,
            "family": asset.family,
            "paradigm": asset.paradigm.value,
            "name": asset.name,
            "variant_hints": list(asset.variant_hints),
            "review_status": asset.audit.review_status,
            "source_kind": asset.audit.source_kind,
            **migration_readiness_summary(asset.audit.migration_readiness),
        }
    if isinstance(asset, SkeletonGraph):
        metadata = asset.metadata or {}
        if isinstance(metadata.get("asset"), dict):
            return dict(metadata["asset"])
        return {}
    if isinstance(asset, dict):
        summary = {
            key: asset[key]
            for key in (
                "asset_id",
                "asset_version",
                "family",
                "paradigm",
                "name",
                "variant_hints",
                "review_status",
                "source_kind",
                "migration_readiness_status",
                "migration_readiness_target_repository",
                "migration_readiness_target_scope",
                "migration_readiness_rationale",
                "migration_readiness_check_count",
                "migration_readiness_required_check_count",
                "migration_readiness_completed_required_check_count",
                "migration_readiness_ready",
                "migration_readiness_check_ids",
            )
            if key in asset
        }
        return summary | migration_readiness_summary(asset.get("migration_readiness"))
    return {}


@lru_cache(maxsize=1)
def load_local_skeleton_assets() -> tuple[SkeletonFamilyAsset, ...]:
    """Load local skeleton assets from disk."""
    assets: list[SkeletonFamilyAsset] = []
    if not ASSET_DIR.exists():
        return tuple()
    for path in sorted(ASSET_DIR.glob("*.json")):
        asset = SkeletonFamilyAsset.model_validate_json(path.read_text())
        _validate_registered_stage_hints(asset, path=path)
        assets.append(asset)
    return tuple(assets)


@lru_cache(maxsize=1)
def load_local_skeleton_graphs() -> tuple[
    dict[ConceptType, SkeletonGraph],
    dict[str, SkeletonGraph],
]:
    """Build asset-backed runtime registries."""
    by_paradigm: dict[ConceptType, SkeletonGraph] = {}
    by_name: dict[str, SkeletonGraph] = {}
    for asset in load_local_skeleton_assets():
        graph = asset.to_skeleton_graph()
        if asset.canonical_for_paradigm:
            by_paradigm[asset.paradigm] = graph
        by_name[asset.asset_id] = graph
        for hint in asset.variant_hints:
            by_name[hint] = graph
    return by_paradigm, by_name


def resolve_local_skeleton_asset(
    concept_type: ConceptType,
    *,
    variant: str | None = None,
) -> SkeletonGraph | None:
    """Resolve a local skeleton asset conservatively."""
    by_paradigm, by_name = load_local_skeleton_graphs()
    if variant:
        resolved = by_name.get(variant)
        if resolved is not None and resolved.paradigm == concept_type:
            return resolved
        return None
    return by_paradigm.get(concept_type)


def _validate_registered_stage_hints(
    asset: SkeletonFamilyAsset,
    *,
    path: Path,
) -> None:
    unknown = unknown_registered_atom_references(
        stage.matched_primitive for stage in asset.stages
    )
    if not unknown:
        return
    joined = ", ".join(unknown)
    raise ValueError(
        f"Skeleton asset '{asset.asset_id}' at '{path}' references unknown registered atoms: {joined}"
    )
