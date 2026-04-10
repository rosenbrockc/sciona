"""Auditable family-local usability rule registries."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
from sciona.usability import (
    CanonicalUsabilityReasonDefinition,
    UsabilityProvenanceKind,
    UsabilityReasonKind,
    UsabilityScope,
    canonical_usability_reason_definition,
    known_usability_blocking_reason_codes,
    known_usability_reason_codes,
    known_usability_warning_reason_codes,
)


_BLOCKING_REASON_CODES = set(known_usability_blocking_reason_codes())
_WARNING_REASON_CODES = set(known_usability_warning_reason_codes())
_ALL_REASON_CODES = set(known_usability_reason_codes())
_REDEFINITION_PHRASES = (
    "redefines the shared meaning",
    "redefine the shared meaning",
    "overrides the shared meaning",
    "shared meaning becomes",
    "shared meaning is replaced",
    "means only",
    "renames the shared reason",
    "relabels the shared reason",
)


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def _entry_redefines_shared_meaning(
    definition: CanonicalUsabilityReasonDefinition,
    notes: list[str],
) -> bool:
    normalized_shared = {
        _normalize_text(definition.shared_meaning),
        _normalize_text(definition.governance_rationale),
    }
    for note in notes:
        normalized_note = _normalize_text(note)
        if not normalized_note:
            continue
        if any(phrase in normalized_note for phrase in _REDEFINITION_PHRASES):
            return True
        if normalized_note in normalized_shared:
            return True
    return False


class UsabilityRegistryReference(BaseModel):
    """Human-reviewable reference for a family usability registry."""

    title: str
    citation: str = ""
    url: str = ""
    note: str = ""


class UsabilityRegistryEntry(BaseModel):
    """One family-local interpretation of a canonical usability reason."""

    reason_code: str
    reason_kind: UsabilityReasonKind
    supported_scopes: list[UsabilityScope] = Field(default_factory=list)
    sanctioned_provenance_kinds: list[UsabilityProvenanceKind] = Field(default_factory=list)
    expected_evidence_strength: Literal["weak", "moderate", "strong"] = "moderate"
    admissibility_notes: list[str] = Field(default_factory=list)
    escalation_conditions: list[str] = Field(default_factory=list)
    family_notes: list[str] = Field(default_factory=list)
    references: list[UsabilityRegistryReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_entry(self) -> "UsabilityRegistryEntry":
        definition = canonical_usability_reason_definition(self.reason_code)
        if self.reason_code not in _ALL_REASON_CODES:
            raise ValueError(
                "Family usability registries must reference known canonical reason codes: "
                f"{self.reason_code}"
            )
        if (
            self.reason_kind == UsabilityReasonKind.BLOCKING
            and self.reason_code not in _BLOCKING_REASON_CODES
        ):
            raise ValueError(f"Blocking reason kind cannot use warning code: {self.reason_code}")
        if (
            self.reason_kind == UsabilityReasonKind.WARNING
            and self.reason_code not in _WARNING_REASON_CODES
        ):
            raise ValueError(f"Warning reason kind cannot use blocking code: {self.reason_code}")
        if definition.kind != self.reason_kind:
            raise ValueError(
                f"Family usability registry entry '{self.reason_code}' cannot redefine canonical reason kind"
            )
        if not self.supported_scopes:
            raise ValueError(
                f"Usability registry entry '{self.reason_code}' must declare supported scopes"
            )
        if not self.sanctioned_provenance_kinds:
            raise ValueError(
                f"Usability registry entry '{self.reason_code}' must declare sanctioned provenance kinds"
            )
        if not self.admissibility_notes:
            raise ValueError(f"Usability registry entry '{self.reason_code}' must include admissibility notes")
        if not self.escalation_conditions:
            raise ValueError(
                f"Usability registry entry '{self.reason_code}' must include escalation conditions"
            )
        if not self.family_notes:
            raise ValueError(f"Usability registry entry '{self.reason_code}' must include family notes")
        if not self.references:
            raise ValueError(
                f"Usability registry entry '{self.reason_code}' must include at least one reference"
            )
        if _entry_redefines_shared_meaning(definition, self.family_notes):
            raise ValueError(
                f"Usability registry entry '{self.reason_code}' appears to redefine shared canonical meaning"
            )
        return self


class UsabilityRegistryAudit(BaseModel):
    """Audit metadata for a family usability registry asset."""

    provenance: str = ""
    source_kind: str = "local_asset"
    review_status: Literal["draft", "transitional", "reviewed", "canonical"] = "draft"
    rationale: str = ""
    dejargonized_summary: str = ""
    uncertainty_notes: list[str] = Field(default_factory=list)
    references: list[UsabilityRegistryReference] = Field(default_factory=list)
    maintainers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_audit(self) -> "UsabilityRegistryAudit":
        if not self.provenance.strip():
            raise ValueError("Usability registry audit must include provenance")
        if not self.source_kind.strip():
            raise ValueError("Usability registry audit must include source_kind")
        if not self.rationale.strip():
            raise ValueError("Usability registry audit must include rationale")
        if not self.dejargonized_summary.strip():
            raise ValueError(
                "Usability registry audit must include a dejargonized summary"
            )
        if not self.references:
            raise ValueError(
                "Usability registry audit must include at least one reference"
            )
        if self.review_status in {"draft", "transitional"} and not self.uncertainty_notes:
            raise ValueError(
                "Draft or transitional usability registries must include uncertainty notes"
            )
        if self.review_status in {"draft", "transitional", "reviewed"} and not self.maintainers:
            raise ValueError(
                "Non-canonical usability registries must declare maintainers"
            )
        return self


class UsabilityFamilyRegistry(BaseModel):
    """Auditable family-local usability interpretation asset."""

    asset_id: str
    asset_version: str
    family: str
    family_aliases: list[str] = Field(default_factory=list)
    name: str
    summary: str
    dejargonized_summary: str = ""
    entries: list[UsabilityRegistryEntry] = Field(default_factory=list)
    audit: UsabilityRegistryAudit = Field(default_factory=UsabilityRegistryAudit)

    @model_validator(mode="after")
    def _validate_registry(self) -> "UsabilityFamilyRegistry":
        if not self.asset_id.strip():
            raise ValueError("Usability registry must include asset_id")
        if not self.asset_version.strip():
            raise ValueError("Usability registry must include asset_version")
        if not self.family.strip():
            raise ValueError("Usability registry must include family")
        if not self.name.strip():
            raise ValueError("Usability registry must include name")
        if not self.summary.strip():
            raise ValueError("Usability registry must include summary")
        if not (self.dejargonized_summary or self.audit.dejargonized_summary):
            raise ValueError(
                f"Usability registry '{self.asset_id}' must include a dejargonized summary"
            )
        if not self.audit.references:
            raise ValueError(
                f"Usability registry '{self.asset_id}' must include at least one reference"
            )
        reason_codes = [entry.reason_code for entry in self.entries]
        duplicates = sorted({item for item in reason_codes if reason_codes.count(item) > 1})
        if duplicates:
            raise ValueError(
                f"Usability registry '{self.asset_id}' defines duplicate reason codes: "
                + ", ".join(duplicates)
            )
        alias_duplicates = sorted(
            {item for item in self.family_aliases if self.family_aliases.count(item) > 1}
        )
        if alias_duplicates:
            raise ValueError(
                f"Usability registry '{self.asset_id}' defines duplicate family aliases: "
                + ", ".join(alias_duplicates)
            )
        return self


def usability_registry_summary(
    registry: UsabilityFamilyRegistry | dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a compact summary payload for runtime/debug surfaces."""
    if registry is None:
        return {}
    if isinstance(registry, UsabilityFamilyRegistry):
        return {
            "asset_id": registry.asset_id,
            "asset_version": registry.asset_version,
            "family": registry.family,
            "family_aliases": list(registry.family_aliases),
            "reason_count": len(registry.entries),
            "review_status": registry.audit.review_status,
            "source_kind": registry.audit.source_kind,
        }
    if isinstance(registry, dict):
        return {
            "asset_id": str(registry.get("asset_id", "") or ""),
            "asset_version": str(registry.get("asset_version", "") or ""),
            "family": str(registry.get("family", "") or ""),
            "family_aliases": list(registry.get("family_aliases", []) or []),
            "reason_count": len(registry.get("entries", []) or []),
            "review_status": str(registry.get("review_status", "") or ""),
            "source_kind": str(registry.get("source_kind", "") or ""),
        }
    return {}


ASSET_DIR = Path(__file__).resolve().parent / "principal" / "assets" / "usability_registries"


@lru_cache(maxsize=1)
def load_local_usability_registries() -> tuple[UsabilityFamilyRegistry, ...]:
    """Load local family usability registries from disk."""
    assets: list[UsabilityFamilyRegistry] = []
    if not ASSET_DIR.exists():
        return tuple()
    for path in sorted(ASSET_DIR.glob("*.json")):
        raw = json.loads(path.read_text())
        assets.append(UsabilityFamilyRegistry.model_validate(raw))
    return tuple(assets)


@lru_cache(maxsize=1)
def load_local_usability_registries_by_family() -> dict[str, UsabilityFamilyRegistry]:
    """Index local usability registries by family."""
    registries = load_local_usability_registries()
    by_family = {asset.family: asset for asset in registries}
    for asset in registries:
        for alias in asset.family_aliases:
            by_family.setdefault(alias, asset)
    return by_family


def resolve_local_usability_registry(family: str) -> UsabilityFamilyRegistry | None:
    """Resolve the best local family usability registry for a family."""
    return load_local_usability_registries_by_family().get(family)


def known_usability_blocking_reason_codes() -> tuple[str, ...]:
    """Return canonical blocking reason codes."""

    return tuple(sorted(_BLOCKING_REASON_CODES))


def known_usability_warning_reason_codes() -> tuple[str, ...]:
    """Return canonical warning reason codes."""

    return tuple(sorted(_WARNING_REASON_CODES))


def known_usability_reason_codes() -> tuple[str, ...]:
    """Return all canonical usability reason codes."""

    return tuple(sorted(_ALL_REASON_CODES))


__all__ = [
    "UsabilityFamilyRegistry",
    "UsabilityRegistryAudit",
    "UsabilityRegistryEntry",
    "UsabilityRegistryReference",
    "known_usability_blocking_reason_codes",
    "known_usability_reason_codes",
    "known_usability_warning_reason_codes",
    "load_local_usability_registries",
    "load_local_usability_registries_by_family",
    "resolve_local_usability_registry",
    "usability_registry_summary",
]
