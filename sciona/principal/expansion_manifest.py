"""Expansion/refinement inventory manifests and release-gate checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from sciona.atom_identity import candidate_atom_provider_roots
from sciona.principal.expansion_assets import (
    AssetBackedExpansionRuleSet,
    ExpansionFamilyAsset,
    load_local_expansion_assets,
)


MANIFEST_VERSION = "sciona.expansion_inventory.v1"
MANIFEST_SINKS = ("supabase", "memgraph", "sqlite")
EMPTY_INVENTORY_MARKER_NAMES = ("README.md", "EMPTY_INVENTORY.md")


@dataclass(frozen=True)
class ProviderExpansionInventory:
    """Expansion inventory status for one provider repository."""

    provider_root: str
    has_expansion_assets: bool
    has_empty_inventory_marker: bool
    expansion_asset_count: int
    marker_path: str = ""

    @property
    def is_documented(self) -> bool:
        return self.has_expansion_assets or self.has_empty_inventory_marker


@dataclass(frozen=True)
class ExpansionManifestClosureReport:
    """Release-gate report for expansion inventory publication."""

    asset_count: int
    operation_count: int
    provider_inventories: tuple[ProviderExpansionInventory, ...]
    missing_provider_inventory_roots: tuple[str, ...]
    missing_asset_backed_rule_sets: tuple[str, ...]
    missing_runtime_rules: tuple[str, ...]
    manifest_sink_mismatches: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not (
            self.missing_provider_inventory_roots
            or self.missing_asset_backed_rule_sets
            or self.missing_runtime_rules
            or self.manifest_sink_mismatches
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": MANIFEST_VERSION,
            "ok": self.ok,
            "asset_count": self.asset_count,
            "operation_count": self.operation_count,
            "provider_inventories": [
                {
                    "provider_root": inventory.provider_root,
                    "has_expansion_assets": inventory.has_expansion_assets,
                    "has_empty_inventory_marker": inventory.has_empty_inventory_marker,
                    "expansion_asset_count": inventory.expansion_asset_count,
                    "marker_path": inventory.marker_path,
                    "is_documented": inventory.is_documented,
                }
                for inventory in self.provider_inventories
            ],
            "missing_provider_inventory_roots": list(self.missing_provider_inventory_roots),
            "missing_asset_backed_rule_sets": list(self.missing_asset_backed_rule_sets),
            "missing_runtime_rules": list(self.missing_runtime_rules),
            "manifest_sink_mismatches": list(self.manifest_sink_mismatches),
        }


def build_expansion_inventory_manifest(
    assets: Iterable[ExpansionFamilyAsset] | None = None,
) -> dict[str, Any]:
    """Build one canonical expansion inventory payload for all manifest sinks."""
    loaded_assets = tuple(assets) if assets is not None else load_local_expansion_assets()
    rows = tuple(_operation_manifest_row(asset, operation) for asset in loaded_assets for operation in asset.operations)
    row_dicts = [dict(row) for row in rows]
    return {
        "manifest_version": MANIFEST_VERSION,
        "asset_count": len(loaded_assets),
        "operation_count": len(rows),
        "operation_keys": [row["operation_key"] for row in row_dicts],
        "sinks": {
            sink: {
                "manifest_version": MANIFEST_VERSION,
                "operation_count": len(row_dicts),
                "operations": row_dicts,
            }
            for sink in MANIFEST_SINKS
        },
    }


def check_expansion_manifest_closure(
    *,
    provider_roots: Iterable[str | Path] | None = None,
    assets: Iterable[ExpansionFamilyAsset] | None = None,
    rule_sets: Sequence[Any] | None = None,
    require_asset_backed_rule_sets: bool = True,
) -> ExpansionManifestClosureReport:
    """Validate provider, runtime-rule, and manifest-sink closure."""
    loaded_assets = tuple(assets) if assets is not None else load_local_expansion_assets()
    if rule_sets is None:
        from sciona.principal.expansion_rules import default_rule_sets

        rule_sets = tuple(default_rule_sets())
    else:
        rule_sets = tuple(rule_sets)
    roots = tuple(
        Path(root).expanduser().resolve()
        for root in (provider_roots if provider_roots is not None else candidate_atom_provider_roots())
        if Path(root).expanduser().exists()
    )
    provider_inventories = tuple(_provider_inventory(root) for root in roots)
    missing_provider_roots = tuple(
        inventory.provider_root
        for inventory in provider_inventories
        if not inventory.is_documented
    )
    missing_asset_backed = _missing_asset_backed_rule_sets(
        loaded_assets,
        rule_sets,
        require_asset_backed=require_asset_backed_rule_sets,
    )
    missing_runtime_rules = _missing_runtime_rules(loaded_assets, rule_sets)
    sink_mismatches = _manifest_sink_mismatches(build_expansion_inventory_manifest(loaded_assets))
    operation_count = sum(len(asset.operations) for asset in loaded_assets)
    return ExpansionManifestClosureReport(
        asset_count=len(loaded_assets),
        operation_count=operation_count,
        provider_inventories=provider_inventories,
        missing_provider_inventory_roots=missing_provider_roots,
        missing_asset_backed_rule_sets=missing_asset_backed,
        missing_runtime_rules=missing_runtime_rules,
        manifest_sink_mismatches=sink_mismatches,
    )


def _operation_manifest_row(asset: ExpansionFamilyAsset, operation: Any) -> dict[str, Any]:
    operation_key = f"{asset.asset_id}:{operation.rule_name}"
    return {
        "operation_key": operation_key,
        "artifact_kind": "expansion_operation",
        "asset_id": asset.asset_id,
        "asset_version": asset.asset_version,
        "asset_family": asset.family,
        "asset_domain": asset.domain,
        "asset_name": asset.name,
        "operation_rule_name": operation.rule_name,
        "operation_id": operation.operation_id,
        "operation_type": operation.operation_type,
        "operation_name": operation.name,
        "applies_to": operation.applies_to,
        "prerequisite_operations": list(operation.prerequisite_operations),
        "runtime_rule_builder": operation.runtime_rule_builder,
        "runtime_diagnostic": operation.runtime_diagnostic,
        "review_status": asset.audit.review_status,
        "source_kind": asset.audit.source_kind,
        "dejargonized_summary": operation.dejargonized_summary,
    }


def _provider_inventory(root: Path) -> ProviderExpansionInventory:
    expansions_dir = root / "data" / "expansions"
    assets = tuple(expansions_dir.glob("*.json")) if expansions_dir.exists() else tuple()
    marker_path = ""
    for marker_name in EMPTY_INVENTORY_MARKER_NAMES:
        candidate = expansions_dir / marker_name
        if candidate.exists():
            marker_path = str(candidate)
            break
    return ProviderExpansionInventory(
        provider_root=str(root),
        has_expansion_assets=bool(assets),
        has_empty_inventory_marker=bool(marker_path),
        expansion_asset_count=len(assets),
        marker_path=marker_path,
    )


def _missing_asset_backed_rule_sets(
    assets: tuple[ExpansionFamilyAsset, ...],
    rule_sets: Sequence[Any],
    *,
    require_asset_backed: bool,
) -> tuple[str, ...]:
    by_name = {str(getattr(rule_set, "name", "")): rule_set for rule_set in rule_sets}
    missing: list[str] = []
    for asset in assets:
        rule_set = by_name.get(asset.family)
        if rule_set is None:
            missing.append(asset.family)
            continue
        if require_asset_backed and not isinstance(rule_set, AssetBackedExpansionRuleSet):
            missing.append(asset.family)
    return tuple(sorted(missing))


def _missing_runtime_rules(
    assets: tuple[ExpansionFamilyAsset, ...],
    rule_sets: Sequence[Any],
) -> tuple[str, ...]:
    by_name = {str(getattr(rule_set, "name", "")): rule_set for rule_set in rule_sets}
    missing: list[str] = []
    for asset in assets:
        rule_set = by_name.get(asset.family)
        if rule_set is None:
            missing.extend(f"{asset.family}:{operation.rule_name}" for operation in asset.operations)
            continue
        rule_names = {str(getattr(rule, "name", "")) for rule in rule_set.rules()}
        missing.extend(
            f"{asset.family}:{operation.rule_name}"
            for operation in asset.operations
            if operation.rule_name not in rule_names
        )
    return tuple(sorted(missing))


def _manifest_sink_mismatches(manifest: dict[str, Any]) -> tuple[str, ...]:
    sinks = manifest.get("sinks", {})
    expected_keys = tuple(manifest.get("operation_keys", ()))
    mismatches: list[str] = []
    for sink in MANIFEST_SINKS:
        sink_payload = sinks.get(sink, {})
        sink_keys = tuple(row.get("operation_key", "") for row in sink_payload.get("operations", ()))
        if sink_keys != expected_keys:
            mismatches.append(sink)
    return tuple(mismatches)

