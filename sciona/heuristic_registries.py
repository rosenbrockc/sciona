"""Auditable family-local interpretation of canonical heuristics."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from sciona.heuristics import (
    HeuristicActionClass,
    HeuristicProducerKind,
    known_heuristic_ids,
)


ASSET_DIR = (
    Path(__file__).resolve().parent / "principal" / "assets" / "heuristic_registries"
)


class HeuristicRegistryReference(BaseModel):
    """Human-reviewable reference for a family heuristic registry."""

    title: str
    citation: str = ""
    url: str = ""
    note: str = ""


class HeuristicRegistryEntry(BaseModel):
    """One family-local interpretation of a canonical heuristic."""

    heuristic_id: str
    sanctioned_producer_kinds: list[HeuristicProducerKind] = Field(default_factory=list)
    supported_action_classes: list[HeuristicActionClass] = Field(default_factory=list)
    action_priority: list[HeuristicActionClass] = Field(default_factory=list)
    expected_evidence_strength: Literal["weak", "moderate", "strong"] = "moderate"
    admissibility_notes: list[str] = Field(default_factory=list)
    escalation_conditions: list[str] = Field(default_factory=list)
    family_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_entry(self) -> "HeuristicRegistryEntry":
        if self.heuristic_id not in set(known_heuristic_ids()):
            raise ValueError(
                "Family registries must reference known canonical heuristic ids: "
                f"{self.heuristic_id}"
            )
        if self.action_priority:
            allowed = set(self.supported_action_classes)
            invalid = [item for item in self.action_priority if item not in allowed]
            if invalid:
                raise ValueError(
                    "action_priority must be a subset of supported_action_classes"
                )
        return self


class HeuristicRegistryAudit(BaseModel):
    """Audit metadata for a family heuristic registry asset."""

    provenance: str = ""
    source_kind: str = "local_asset"
    review_status: str = "draft"
    rationale: str = ""
    dejargonized_summary: str = ""
    uncertainty_notes: list[str] = Field(default_factory=list)
    references: list[HeuristicRegistryReference] = Field(default_factory=list)
    maintainers: list[str] = Field(default_factory=list)


class HeuristicFamilyRegistry(BaseModel):
    """Auditable family-local heuristic interpretation asset."""

    asset_id: str
    asset_version: str
    family: str
    family_aliases: list[str] = Field(default_factory=list)
    skeleton_scope: str = ""
    name: str
    summary: str
    dejargonized_summary: str = ""
    entries: list[HeuristicRegistryEntry] = Field(default_factory=list)
    audit: HeuristicRegistryAudit = Field(default_factory=HeuristicRegistryAudit)

    @model_validator(mode="after")
    def _validate_registry(self) -> "HeuristicFamilyRegistry":
        if not (self.dejargonized_summary or self.audit.dejargonized_summary):
            raise ValueError(
                f"Heuristic registry '{self.asset_id}' must include a dejargonized summary"
            )
        if not self.audit.references:
            raise ValueError(
                f"Heuristic registry '{self.asset_id}' must include at least one reference"
            )
        heuristic_ids = [entry.heuristic_id for entry in self.entries]
        duplicates = sorted(
            {item for item in heuristic_ids if heuristic_ids.count(item) > 1}
        )
        if duplicates:
            raise ValueError(
                f"Heuristic registry '{self.asset_id}' defines duplicate heuristic ids: "
                + ", ".join(duplicates)
            )
        return self


def heuristic_registry_summary(
    registry: HeuristicFamilyRegistry | dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a compact summary payload for runtime/debug surfaces."""
    if registry is None:
        return {}
    if isinstance(registry, HeuristicFamilyRegistry):
        return {
            "asset_id": registry.asset_id,
            "asset_version": registry.asset_version,
            "family": registry.family,
            "family_aliases": list(registry.family_aliases),
            "skeleton_scope": registry.skeleton_scope,
            "heuristic_count": len(registry.entries),
            "review_status": registry.audit.review_status,
            "source_kind": registry.audit.source_kind,
        }
    if isinstance(registry, dict):
        return {
            "asset_id": str(registry.get("asset_id", "") or ""),
            "asset_version": str(registry.get("asset_version", "") or ""),
            "family": str(registry.get("family", "") or ""),
            "family_aliases": list(registry.get("family_aliases", []) or []),
            "skeleton_scope": str(registry.get("skeleton_scope", "") or ""),
            "heuristic_count": len(registry.get("entries", []) or []),
            "review_status": str(registry.get("review_status", "") or ""),
            "source_kind": str(registry.get("source_kind", "") or ""),
        }
    return {}


@lru_cache(maxsize=1)
def load_local_heuristic_registries() -> tuple[HeuristicFamilyRegistry, ...]:
    """Load local family heuristic registries from disk."""
    assets: list[HeuristicFamilyRegistry] = []
    if not ASSET_DIR.exists():
        return tuple()
    for path in sorted(ASSET_DIR.glob("*.json")):
        raw = json.loads(path.read_text())
        assets.append(HeuristicFamilyRegistry.model_validate(raw))
    return tuple(assets)


@lru_cache(maxsize=1)
def load_local_heuristic_registries_by_family() -> dict[str, HeuristicFamilyRegistry]:
    """Index local heuristic registries by family."""
    return {asset.family: asset for asset in load_local_heuristic_registries()}


def resolve_local_heuristic_registry(
    family: str,
    *,
    skeleton_scope: str | None = None,
) -> HeuristicFamilyRegistry | None:
    """Resolve the best local family registry for a family and optional scope."""
    registries = load_local_heuristic_registries_by_family()
    registry = registries.get(family)
    if registry is None:
        for candidate in load_local_heuristic_registries():
            if family in set(candidate.family_aliases):
                registry = candidate
                break
    if registry is None:
        return None
    if (
        skeleton_scope
        and registry.skeleton_scope
        and registry.skeleton_scope != skeleton_scope
    ):
        return None
    return registry
