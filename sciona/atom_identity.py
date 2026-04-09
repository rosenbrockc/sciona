"""Shared atom identity and provider helpers.

This module is the thin migration seam between the current single-package
`ageoa` world and the intended federated `sciona.atoms.*` future.
"""

from __future__ import annotations

import os
from pathlib import Path

LEGACY_ATOM_PACKAGE_PREFIX = "ageoa"
FEDERATED_ATOM_NAMESPACE_PREFIX = "sciona.atoms"
DEFAULT_PROVIDER_ID = "core.ageo_atoms"
DEFAULT_PROVIDER_ROOT = (Path(__file__).resolve().parents[1].parent / "ageo-atoms").resolve()
ATOM_METADATA_GLOB_CANDIDATES: tuple[str, ...] = (
    "ageoa/**/heuristic_metadata.json",
    "sciona/atoms/**/heuristic_metadata.json",
)


def known_atom_package_prefixes() -> tuple[str, ...]:
    """Return recognized import namespace prefixes for atom packages."""
    configured = str(os.environ.get("SCIONA_ATOM_PACKAGE_PREFIXES", "") or "").strip()
    prefixes: list[str] = []
    if configured:
        prefixes.extend(
            token.strip().rstrip(".")
            for token in configured.split(",")
            if token.strip()
        )
    prefixes.extend((LEGACY_ATOM_PACKAGE_PREFIX, FEDERATED_ATOM_NAMESPACE_PREFIX))
    deduped: list[str] = []
    seen: set[str] = set()
    for prefix in prefixes:
        if prefix in seen:
            continue
        seen.add(prefix)
        deduped.append(prefix)
    return tuple(deduped)


def candidate_atom_provider_roots() -> tuple[Path, ...]:
    """Return configured or default external atom-provider roots."""
    roots: list[Path] = []
    configured_multi = str(os.environ.get("SCIONA_ATOM_PROVIDER_ROOTS", "") or "").strip()
    if configured_multi:
        roots.extend(
            Path(token).expanduser()
            for token in configured_multi.split(os.pathsep)
            if token.strip()
        )
    configured_legacy = str(os.environ.get("SCIONA_AGEO_ATOMS_ROOT", "") or "").strip()
    if configured_legacy:
        roots.append(Path(configured_legacy).expanduser())
    roots.append(DEFAULT_PROVIDER_ROOT)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return tuple(deduped)


def atom_provider_id_for_fqdn(import_fqdn: str) -> str:
    """Infer a coarse provider identifier from the current import namespace."""
    text = str(import_fqdn or "").strip()
    if not text:
        return DEFAULT_PROVIDER_ID
    if text.startswith(LEGACY_ATOM_PACKAGE_PREFIX + ".") or text == LEGACY_ATOM_PACKAGE_PREFIX:
        return DEFAULT_PROVIDER_ID
    if text.startswith(FEDERATED_ATOM_NAMESPACE_PREFIX + ".") or text == FEDERATED_ATOM_NAMESPACE_PREFIX:
        parts = text.split(".")
        if len(parts) >= 3:
            return ".".join(parts[:3])
        return FEDERATED_ATOM_NAMESPACE_PREFIX
    return DEFAULT_PROVIDER_ID


def logical_atom_id_from_fqdn(import_fqdn: str) -> str:
    """Strip known namespace/package prefixes from an import FQDN."""
    text = str(import_fqdn or "").strip()
    if not text:
        return ""
    for prefix in sorted(known_atom_package_prefixes(), key=len, reverse=True):
        dotted = prefix + "."
        if text.startswith(dotted):
            return text[len(dotted) :]
        if text == prefix:
            return ""
    return text


def infer_source_family(
    label: str,
    *,
    fallback: str = "",
    preserve_namespace: bool = True,
) -> str:
    """Infer a stable source-family label from an atom/template identifier."""
    text = str(label or "").strip()
    if not text:
        return str(fallback or "").strip()

    for prefix in sorted(known_atom_package_prefixes(), key=len, reverse=True):
        dotted = prefix + "."
        if not text.startswith(dotted):
            continue
        remainder = text[len(dotted) :]
        remainder_parts = [part for part in remainder.split(".") if part]
        if not remainder_parts:
            return prefix if preserve_namespace else str(fallback or "").strip()
        family = remainder_parts[0]
        if preserve_namespace:
            return f"{prefix}.{family}"
        return family

    if "." in text:
        return text.split(".", 1)[0]
    return str(fallback or text).strip()

