"""Shared helpers for Supabase backfill scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

if TYPE_CHECKING:
    from supabase import Client

try:
    from sciona.atoms.provider_inventory import (
        discover_audit_manifest_path as _provider_discover_audit_manifest_path,
        discover_artifact_roots as _provider_discover_artifact_roots,
        discover_references_registry_path as _provider_discover_references_registry_path,
        iter_provider_artifact_files as _provider_iter_artifact_files,
        namespace_prefix_for_artifact_root as _provider_namespace_prefix_for_artifact_root,
        provider_repo_roots as _provider_repo_roots,
    )
except Exception:
    _provider_discover_audit_manifest_path = None
    _provider_discover_artifact_roots = None
    _provider_discover_references_registry_path = None
    _provider_iter_artifact_files = None
    _provider_namespace_prefix_for_artifact_root = None
    _provider_repo_roots = None


DEFAULT_ATOMS_ROOT = "../sciona-atoms/src/sciona/atoms"
DEFAULT_PROVIDER_REPO_ROOT = "../sciona-atoms"
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_SOURCES_FILE = _REPO_ROOT / "sources.yml"
_ARTIFACT_ROOT_CANDIDATES: tuple[Path, ...] = (
    Path("src/sciona/atoms"),
    Path("sciona/atoms"),
)
_AUDIT_MANIFEST_RELATIVE = Path("data/audit_manifest.json")
_REFERENCES_REGISTRY_RELATIVE = Path("data/references/registry.json")
_NAMESPACE_ANCHORS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("src", "sciona", "atoms"), ("sciona", "atoms")),
    (("sciona", "atoms"), ("sciona", "atoms")),
)


def _dedupe_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return tuple(deduped)


def _find_anchor(parts: Sequence[str], anchor: Sequence[str]) -> int | None:
    width = len(anchor)
    if width == 0 or len(parts) < width:
        return None
    for index in range(len(parts) - width + 1):
        if tuple(parts[index : index + width]) == tuple(anchor):
            return index
    return None


def configured_provider_repo_roots() -> tuple[Path, ...]:
    """Return provider repository roots from env or ``sources.yml``."""
    if _provider_repo_roots is not None:
        try:
            return tuple(Path(path).resolve() for path in _provider_repo_roots())
        except Exception:
            pass

    configured = str(os.environ.get("SCIONA_ATOM_PROVIDER_ROOTS", "") or "").strip()
    if configured:
        return _dedupe_paths(
            Path(token)
            for token in configured.split(os.pathsep)
            if str(token).strip()
        )

    roots: list[Path] = []
    try:
        from sciona.sources import load_sources, resolve_source
    except Exception:
        load_sources = resolve_source = None  # type: ignore[assignment]

    if load_sources is not None and resolve_source is not None and _SOURCES_FILE.exists():
        try:
            config = load_sources(_SOURCES_FILE)
        except Exception:
            config = None
        if config is not None:
            for source in config.sources:
                if not source.path:
                    continue
                try:
                    roots.append(resolve_source(source, base_dir=_REPO_ROOT))
                except Exception:
                    continue

    roots.append((_REPO_ROOT / DEFAULT_PROVIDER_REPO_ROOT).resolve())
    return _dedupe_paths(roots)


def provider_artifact_roots() -> tuple[Path, ...]:
    """Return filesystem roots that may contain file-backed atom artifacts."""
    if _provider_discover_artifact_roots is not None:
        try:
            return tuple(Path(path).resolve() for path in _provider_discover_artifact_roots())
        except Exception:
            pass

    configured = str(os.environ.get("SCIONA_ATOM_ARTIFACT_ROOTS", "") or "").strip()
    if configured:
        return _dedupe_paths(
            Path(token)
            for token in configured.split(os.pathsep)
            if str(token).strip()
        )

    roots: list[Path] = []
    for repo_root in configured_provider_repo_roots():
        for relative in _ARTIFACT_ROOT_CANDIDATES:
            candidate = repo_root / relative
            if candidate.exists():
                roots.append(candidate)

    if roots:
        return _dedupe_paths(roots)
    return (Path(DEFAULT_ATOMS_ROOT).expanduser().resolve(),)


def iter_provider_artifact_files(
    filename: str,
    *,
    roots: Sequence[Path] | None = None,
) -> list[Path]:
    """Return matching artifact files across all configured provider roots."""
    if _provider_iter_artifact_files is not None:
        try:
            return [
                Path(path).resolve()
                for path in _provider_iter_artifact_files(filename, roots=roots)
            ]
        except Exception:
            pass

    search_roots = tuple(roots) if roots is not None else provider_artifact_roots()
    matches: list[Path] = []
    for root in search_roots:
        for path in sorted(root.rglob(filename)):
            if "__pycache__" in path.parts:
                continue
            matches.append(path)
    return sorted(_dedupe_paths(matches))


def namespace_prefix_for_artifact_root(root: Path) -> tuple[str, ...]:
    """Return the dotted namespace prefix implied by an artifact root."""
    if _provider_namespace_prefix_for_artifact_root is not None:
        try:
            return tuple(_provider_namespace_prefix_for_artifact_root(root))
        except Exception:
            pass

    parts = root.parts
    for anchor, prefix in _NAMESPACE_ANCHORS:
        index = _find_anchor(parts, anchor)
        if index is not None and index + len(anchor) == len(parts):
            return prefix
    return (root.name,)


def _discover_shared_data_path(env_var: str, relative_path: Path) -> Path:
    configured = str(os.environ.get(env_var, "") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    for repo_root in configured_provider_repo_roots():
        candidate = repo_root / relative_path
        if candidate.exists():
            return candidate.resolve()
    return (_REPO_ROOT / DEFAULT_PROVIDER_REPO_ROOT / relative_path).resolve()


def atoms_root_from_env() -> Path:
    """Return the configured single atom-source root used for file backfills."""
    configured = str(os.environ.get("SCIONA_ATOM_ARTIFACT_ROOT", "") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    roots = provider_artifact_roots()
    if roots:
        return roots[0]
    return Path(DEFAULT_ATOMS_ROOT).expanduser().resolve()


def discover_audit_manifest_path() -> Path:
    """Return the canonical audit manifest path."""
    if _provider_discover_audit_manifest_path is not None:
        try:
            return Path(_provider_discover_audit_manifest_path()).resolve()
        except Exception:
            pass
    return _discover_shared_data_path("AUDIT_MANIFEST_PATH", _AUDIT_MANIFEST_RELATIVE)


def discover_references_registry_path() -> Path:
    """Return the canonical references registry path."""
    if _provider_discover_references_registry_path is not None:
        try:
            return Path(_provider_discover_references_registry_path()).resolve()
        except Exception:
            pass
    return _discover_shared_data_path(
        "REFERENCES_REGISTRY_PATH",
        _REFERENCES_REGISTRY_RELATIVE,
    )


def namespace_from_path(file_path: Path) -> str:
    """Derive the dotted namespace from a Phase 2D source file path."""
    parts = file_path.parent.parts
    clean: list[str] = []
    for part in parts:
        if part == "_artifacts":
            break
        clean.append(part)
    for anchor, prefix in _NAMESPACE_ANCHORS:
        index = _find_anchor(clean, anchor)
        if index is None:
            continue
        suffix = [part for part in clean[index + len(anchor) :] if part]
        return ".".join((*prefix, *suffix))
    return ".".join(clean)


def resolve_atom_id(supabase: "Client", namespace: str, short_name: str) -> str | None:
    """Resolve an atom short name + namespace into an atom_id via exact then suffix lookup."""
    fqdn = f"{namespace}.{short_name}"
    response = (
        supabase.table("atoms")
        .select("atom_id")
        .eq("fqdn", fqdn)
        .limit(1)
        .execute()
    )
    if response.data:
        return response.data[0]["atom_id"]

    response = (
        supabase.table("atoms")
        .select("atom_id")
        .like("fqdn", f"%.{short_name}")
        .limit(1)
        .execute()
    )
    if response.data:
        return response.data[0]["atom_id"]

    return None


def create_supabase_client_from_env() -> Any:
    """Create a service-role Supabase client from environment variables."""
    from supabase import create_client

    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )
