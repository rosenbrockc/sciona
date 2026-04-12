"""Shared atom identity and provider helpers.

This module is the thin migration seam between the current single-package
`ageoa` world and the intended federated `sciona.atoms.*` future.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

LEGACY_ATOM_PACKAGE_PREFIX = "ageoa"
FEDERATED_ATOM_NAMESPACE_PREFIX = "sciona.atoms"
DEFAULT_PROVIDER_ID = "core.ageo_atoms"
DEFAULT_NAMESPACE_PROVIDER_ROOT = (
    Path(__file__).resolve().parents[1].parent / "sciona-atoms"
).resolve()
DEFAULT_PROVIDER_ROOT = (
    Path(__file__).resolve().parents[1].parent / "ageo-atoms"
).resolve()
ATOM_PROVIDER_ID_PREFIX_ENV = "SCIONA_ATOM_PROVIDER_PREFIX_TO_ID"
ATOM_METADATA_GLOB_CANDIDATES: tuple[str, ...] = (
    "ageoa/**/heuristic_metadata.json",
    "sciona/atoms/**/heuristic_metadata.json",
    "src/sciona/atoms/**/heuristic_metadata.json",
)


def _dedupe_in_order(values: Iterable[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return tuple(deduped)


def _normalized_prefixes(values: Iterable[str]) -> tuple[str, ...]:
    return _dedupe_in_order(
        token
        for raw in values
        if (token := str(raw or "").strip().rstrip("."))
    )


def _configured_atom_package_prefixes() -> tuple[str, ...]:
    configured = str(os.environ.get("SCIONA_ATOM_PACKAGE_PREFIXES", "") or "").strip()
    if not configured:
        return tuple()
    return _normalized_prefixes(configured.split(","))


def _provider_id_prefix_pairs() -> tuple[tuple[str, str], ...]:
    configured = str(os.environ.get(ATOM_PROVIDER_ID_PREFIX_ENV, "") or "").strip()
    pairs: list[tuple[str, str]] = []
    seen_prefixes: set[str] = set()
    if not configured:
        return tuple()

    for raw in configured.split(","):
        token = str(raw or "").strip()
        if "=" not in token:
            continue
        prefix_text, provider_id_text = token.split("=", 1)
        prefix = str(prefix_text or "").strip().rstrip(".")
        provider_id = str(provider_id_text or "").strip()
        if not prefix or not provider_id or prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        pairs.append((prefix, provider_id))
    return tuple(pairs)


def _match_known_atom_prefix(import_fqdn: str) -> tuple[str, str]:
    text = str(import_fqdn or "").strip()
    if not text:
        return "", ""
    for prefix in sorted(known_atom_package_prefixes(), key=len, reverse=True):
        dotted = prefix + "."
        if text.startswith(dotted):
            return prefix, text[len(dotted) :]
        if text == prefix:
            return prefix, ""
    return "", text


def known_atom_package_prefixes() -> tuple[str, ...]:
    """Return recognized import namespace prefixes for atom packages."""
    return _normalized_prefixes(
        (
            *_configured_atom_package_prefixes(),
            LEGACY_ATOM_PACKAGE_PREFIX,
            FEDERATED_ATOM_NAMESPACE_PREFIX,
        )
    )


def candidate_atom_provider_roots() -> tuple[Path, ...]:
    """Return provider roots in precedence order.

    Precedence is:
    1. `SCIONA_ATOM_PROVIDER_ROOTS`
    2. `SCIONA_AGEO_ATOMS_ROOT`
    3. the namespace pilot sibling `../sciona-atoms` root
    4. the legacy sibling `../ageo-atoms` root
    """
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
    roots.append(DEFAULT_NAMESPACE_PROVIDER_ROOT)
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
    """Return the configured or inferred provider identifier for an atom import."""
    text = str(import_fqdn or "").strip()
    if not text:
        return DEFAULT_PROVIDER_ID

    for prefix, provider_id in sorted(
        _provider_id_prefix_pairs(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if text == prefix or text.startswith(prefix + "."):
            return provider_id

    matched_prefix, _ = _match_known_atom_prefix(text)
    if matched_prefix == LEGACY_ATOM_PACKAGE_PREFIX:
        return DEFAULT_PROVIDER_ID
    if matched_prefix == FEDERATED_ATOM_NAMESPACE_PREFIX:
        parts = text.split(".")
        if len(parts) >= 3:
            return ".".join(parts[:3])
        return FEDERATED_ATOM_NAMESPACE_PREFIX
    return DEFAULT_PROVIDER_ID


def logical_atom_id_from_fqdn(import_fqdn: str) -> str:
    """Strip known namespace/package prefixes from an import FQDN."""
    _, remainder = _match_known_atom_prefix(import_fqdn)
    return remainder


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

    prefix, remainder = _match_known_atom_prefix(text)
    if prefix:
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
