"""Auditable family-local interpretation of canonical heuristics."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from sciona.asset_migration import (
    MigrationReadinessAsset,
    migration_readiness_summary,
)
from sciona.heuristics import (
    HeuristicActionClass,
    HeuristicProducerKind,
    known_heuristic_ids,
)


ASSET_DIR = (
    Path(__file__).resolve().parent / "principal" / "assets" / "heuristic_registries"
)
DEFAULT_AGEO_ATOMS_ROOT = (Path(__file__).resolve().parent.parent / "ageo-atoms").resolve()
EXTERNAL_ASSET_DIR_CANDIDATES = (
    ("data", "heuristics", "families"),
    ("data", "heuristics", "family_registries"),
    ("data", "heuristics", "registries"),
    ("data", "heuristics", "heuristic_registries"),
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
    source_repository: str = ""
    source_path: str = ""
    review_status: str = "draft"
    rationale: str = ""
    dejargonized_summary: str = ""
    compatibility_status: str = ""
    compatibility_notes: list[str] = Field(default_factory=list)
    migration_readiness: MigrationReadinessAsset = Field(
        default_factory=MigrationReadinessAsset
    )
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
            "source_repository": registry.audit.source_repository,
            "source_path": registry.audit.source_path,
            "compatibility_status": registry.audit.compatibility_status,
            "compatibility_notes": list(registry.audit.compatibility_notes),
            **migration_readiness_summary(registry.audit.migration_readiness),
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
            "source_repository": str(registry.get("source_repository", "") or ""),
            "source_path": str(registry.get("source_path", "") or ""),
            "compatibility_status": str(registry.get("compatibility_status", "") or ""),
            "compatibility_notes": list(registry.get("compatibility_notes", []) or []),
            **migration_readiness_summary(registry.get("migration_readiness")),
        }
    return {}


def heuristic_asset_root(ageo_atoms_root: str | Path | None = None) -> Path:
    """Return the canonical sibling repository root for shared heuristic assets."""
    if ageo_atoms_root is not None:
        return Path(ageo_atoms_root).expanduser().resolve()
    configured = os.environ.get("SCIONA_AGEO_ATOMS_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_AGEO_ATOMS_ROOT


def heuristic_registry_external_asset_dir(
    ageo_atoms_root: str | Path | None = None,
) -> Path:
    """Return the preferred external family-registry directory path."""
    root = heuristic_asset_root(ageo_atoms_root)
    for relative in EXTERNAL_ASSET_DIR_CANDIDATES:
        candidate = root.joinpath(*relative)
        if candidate.exists():
            return candidate
    return root.joinpath(*EXTERNAL_ASSET_DIR_CANDIDATES[0])


def _registry_with_source_metadata(
    registry: HeuristicFamilyRegistry,
    *,
    source_kind: str,
    source_repository: str,
    source_path: Path,
    compatibility_status: str,
    compatibility_notes: list[str] | None = None,
) -> HeuristicFamilyRegistry:
    audit = registry.audit.model_copy(
        update={
            "source_kind": source_kind,
            "source_repository": source_repository,
            "source_path": str(source_path),
            "compatibility_status": compatibility_status,
            "compatibility_notes": list(compatibility_notes or []),
        }
    )
    return registry.model_copy(update={"audit": audit})


def _load_heuristic_registries_from_dir(
    asset_dir: Path,
    *,
    source_kind: str,
    source_repository: str,
) -> tuple[HeuristicFamilyRegistry, ...]:
    assets: list[HeuristicFamilyRegistry] = []
    families_seen: set[str] = set()
    if not asset_dir.exists():
        return tuple()
    for path in sorted(asset_dir.glob("*.json")):
        raw = json.loads(path.read_text())
        registry = HeuristicFamilyRegistry.model_validate(raw)
        if registry.family in families_seen:
            raise ValueError(
                "Duplicate heuristic registry family in "
                f"{source_repository}: {registry.family}"
            )
        families_seen.add(registry.family)
        assets.append(
            _registry_with_source_metadata(
                registry,
                source_kind=source_kind,
                source_repository=source_repository,
                source_path=path,
                compatibility_status="",
            )
        )
    return tuple(assets)


@lru_cache(maxsize=1)
def load_local_heuristic_registries() -> tuple[HeuristicFamilyRegistry, ...]:
    """Load local family heuristic registries from disk."""
    return _load_heuristic_registries_from_dir(
        ASSET_DIR,
        source_kind="local_asset",
        source_repository="ageo-matcher",
    )


@lru_cache(maxsize=1)
def load_local_heuristic_registries_by_family() -> dict[str, HeuristicFamilyRegistry]:
    """Index local heuristic registries by family."""
    return {asset.family: asset for asset in load_local_heuristic_registries()}


@lru_cache(maxsize=4)
def _load_external_heuristic_registries_cached(
    ageo_atoms_root: str,
) -> tuple[HeuristicFamilyRegistry, ...]:
    return _load_heuristic_registries_from_dir(
        heuristic_registry_external_asset_dir(ageo_atoms_root),
        source_kind="shared_asset",
        source_repository="../ageo-atoms",
    )


def load_external_heuristic_registries(
    ageo_atoms_root: str | Path | None = None,
) -> tuple[HeuristicFamilyRegistry, ...]:
    """Load family heuristic registries from the shared sibling repository."""
    root = heuristic_asset_root(ageo_atoms_root)
    return _load_external_heuristic_registries_cached(str(root))


def load_external_heuristic_registries_by_family(
    ageo_atoms_root: str | Path | None = None,
) -> dict[str, HeuristicFamilyRegistry]:
    """Index external heuristic registries by family."""
    return {
        asset.family: asset
        for asset in load_external_heuristic_registries(ageo_atoms_root)
    }


def _selected_registry(
    *,
    local_registry: HeuristicFamilyRegistry | None,
    external_registry: HeuristicFamilyRegistry | None,
    external_asset_dir_exists: bool,
) -> HeuristicFamilyRegistry | None:
    if external_registry is not None and local_registry is not None:
        if external_registry.asset_version != local_registry.asset_version:
            notes = [
                "external registry selected over local fallback despite asset_version mismatch",
                f"external={external_registry.asset_version}",
                f"local={local_registry.asset_version}",
            ]
            return _registry_with_source_metadata(
                external_registry,
                source_kind=external_registry.audit.source_kind,
                source_repository=external_registry.audit.source_repository,
                source_path=Path(external_registry.audit.source_path),
                compatibility_status="version_mismatch",
                compatibility_notes=notes,
            )
        return _registry_with_source_metadata(
            external_registry,
            source_kind=external_registry.audit.source_kind,
            source_repository=external_registry.audit.source_repository,
            source_path=Path(external_registry.audit.source_path),
            compatibility_status="external_preferred",
            compatibility_notes=[
                "external registry selected over local transitional fallback"
            ],
        )
    if external_registry is not None:
        return _registry_with_source_metadata(
            external_registry,
            source_kind=external_registry.audit.source_kind,
            source_repository=external_registry.audit.source_repository,
            source_path=Path(external_registry.audit.source_path),
            compatibility_status="external_only",
            compatibility_notes=[],
        )
    if local_registry is not None:
        notes: list[str] = []
        if not external_asset_dir_exists:
            notes.append(
                "shared heuristic registry directory not found; using local fallback"
            )
        else:
            notes.append(
                "no shared heuristic registry for family; using local fallback"
            )
        return _registry_with_source_metadata(
            local_registry,
            source_kind=local_registry.audit.source_kind,
            source_repository=local_registry.audit.source_repository,
            source_path=Path(local_registry.audit.source_path),
            compatibility_status="local_fallback",
            compatibility_notes=notes,
        )
    return None


def load_heuristic_registries(
    ageo_atoms_root: str | Path | None = None,
) -> tuple[HeuristicFamilyRegistry, ...]:
    """Load heuristic registries with external-first precedence and local fallback."""
    local_by_family = load_local_heuristic_registries_by_family()
    external_by_family = load_external_heuristic_registries_by_family(ageo_atoms_root)
    external_dir_exists = heuristic_registry_external_asset_dir(ageo_atoms_root).exists()
    families = sorted(set(local_by_family) | set(external_by_family))
    selected: list[HeuristicFamilyRegistry] = []
    for family in families:
        registry = _selected_registry(
            local_registry=local_by_family.get(family),
            external_registry=external_by_family.get(family),
            external_asset_dir_exists=external_dir_exists,
        )
        if registry is not None:
            selected.append(registry)
    return tuple(selected)


def load_heuristic_registries_by_family(
    ageo_atoms_root: str | Path | None = None,
) -> dict[str, HeuristicFamilyRegistry]:
    """Index selected heuristic registries by family."""
    return {asset.family: asset for asset in load_heuristic_registries(ageo_atoms_root)}


def heuristic_registry_compatibility_report(
    ageo_atoms_root: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize source selection and migration visibility for family registries."""
    external_root = heuristic_asset_root(ageo_atoms_root)
    external_dir = heuristic_registry_external_asset_dir(external_root)
    local_by_family = load_local_heuristic_registries_by_family()
    external_by_family = load_external_heuristic_registries_by_family(external_root)
    external_dir_exists = external_dir.exists()
    records: list[dict[str, Any]] = []
    for family in sorted(set(local_by_family) | set(external_by_family)):
        selected = _selected_registry(
            local_registry=local_by_family.get(family),
            external_registry=external_by_family.get(family),
            external_asset_dir_exists=external_dir_exists,
        )
        if selected is None:
            continue
        summary = heuristic_registry_summary(selected)
        summary.update(
            {
                "local_available": family in local_by_family,
                "external_available": family in external_by_family,
            }
        )
        records.append(summary)
    warnings: list[str] = []
    if not external_dir_exists:
        warnings.append(f"shared_heuristic_registry_dir_missing:{external_dir}")
    return {
        "ageo_atoms_root": str(external_root),
        "external_asset_dir": str(external_dir),
        "external_asset_dir_exists": external_dir_exists,
        "local_asset_dir": str(ASSET_DIR),
        "record_count": len(records),
        "records": records,
        "warnings": warnings,
    }


def clear_heuristic_registry_caches() -> None:
    """Clear loader caches used by tests and migration tooling."""
    load_local_heuristic_registries.cache_clear()
    load_local_heuristic_registries_by_family.cache_clear()
    _load_external_heuristic_registries_cached.cache_clear()


def resolve_heuristic_registry(
    family: str,
    *,
    skeleton_scope: str | None = None,
    ageo_atoms_root: str | Path | None = None,
) -> HeuristicFamilyRegistry | None:
    """Resolve the best registry from shared assets first, then local fallback."""
    registries = load_heuristic_registries_by_family(ageo_atoms_root)
    registry = registries.get(family)
    if registry is None:
        for candidate in load_heuristic_registries(ageo_atoms_root):
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


def resolve_local_heuristic_registry(
    family: str,
    *,
    skeleton_scope: str | None = None,
) -> HeuristicFamilyRegistry | None:
    """Compatibility shim for legacy callers expecting a local-only resolver."""
    return resolve_heuristic_registry(family, skeleton_scope=skeleton_scope)
