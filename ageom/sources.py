"""Multi-repo atom source management via sources.yml."""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from pathlib import Path
import types
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

_DEFAULT_CDG_GLOB = "**/*cdg*.json"
_CACHE_DIR = ".ageom_cache"


class AtomSource(BaseModel):
    """A single atom repository source."""

    name: str
    package: str
    path: str | None = None
    git: str | None = None
    ref: str = "main"
    cdg_glob: str = _DEFAULT_CDG_GLOB

    @model_validator(mode="after")
    def _require_path_or_git(self) -> "AtomSource":
        if not self.path and not self.git:
            raise ValueError(
                f"Source '{self.name}' must specify either 'path' or 'git'"
            )
        return self


class SourcesConfig(BaseModel):
    """Top-level sources.yml schema."""

    sources: list[AtomSource] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SourcesConfig":
        """Load from a YAML file.  Falls back to an empty config if missing."""
        if path is None:
            path = Path("sources.yml")
        path = Path(path)
        if not path.exists():
            logger.info("No sources.yml found at %s, using empty config", path)
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)


def load_sources(path: str | Path | None = None) -> SourcesConfig:
    """Convenience wrapper for ``SourcesConfig.load``."""
    return SourcesConfig.load(path)


def resolve_source(source: AtomSource, base_dir: Path | None = None) -> Path:
    """Return the local filesystem root for *source*.

    For ``path`` sources the path is resolved relative to *base_dir*
    (defaulting to cwd).  For ``git`` sources the repo is cloned /
    fetched into ``.ageom_cache/<name>``.
    """
    if source.path:
        base = base_dir or Path.cwd()
        return (base / Path(source.path).expanduser()).resolve()

    # Git source — clone or fetch
    assert source.git is not None
    cache_root = (base_dir or Path.cwd()) / _CACHE_DIR
    repo_dir = cache_root / source.name

    if repo_dir.exists():
        _git_fetch_checkout(repo_dir, source.ref)
    else:
        _git_clone(source.git, repo_dir, source.ref)

    return repo_dir


def sync_source(source: AtomSource, base_dir: Path | None = None) -> Path:
    """Fetch / update a git source.  No-op for local-path sources."""
    if source.path:
        base = base_dir or Path.cwd()
        resolved = (base / source.path).resolve()
        logger.info("Source '%s' is a local path: %s", source.name, resolved)
        return resolved
    return resolve_source(source, base_dir)


def discover_cdgs(source: AtomSource, base_dir: Path | None = None) -> list[Path]:
    """Return CDG JSON files found in *source* using its ``cdg_glob``."""
    root = resolve_source(source, base_dir)
    if not root.exists():
        logger.warning("Source root does not exist: %s", root)
        return []
    return sorted(root.glob(source.cdg_glob))


def find_cdg(name: str, config: SourcesConfig | None = None, base_dir: Path | None = None) -> Path | None:
    """Search all sources for a CDG file whose stem contains *name*.

    Returns the first match or ``None``.
    """
    if config is None:
        config = load_sources()
    for source in config.sources:
        for cdg_path in discover_cdgs(source, base_dir):
            if name in cdg_path.stem:
                return cdg_path
    return None


def import_atoms(source: AtomSource, base_dir: Path | None = None) -> None:
    """Import the Python package for *source*, triggering ``@register_atom``.

    For ``path`` sources the repo root is inserted into ``sys.path`` so that
    the package can be found even if it isn't installed.
    """
    if source.path:
        root = resolve_source(source, base_dir)
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    pkg = None
    pkg_path = None
    fallback_scan_root: Path | None = None
    try:
        pkg = importlib.import_module(source.package)
        pkg_path = getattr(pkg, "__path__", None)
    except Exception:
        if not source.path:
            logger.warning(
                "Could not import package '%s' for source '%s'",
                source.package,
                source.name,
            )
            return

        root = resolve_source(source, base_dir)
        package_root = root.joinpath(*source.package.split("."))
        if not package_root.exists():
            logger.warning(
                "Could not import package '%s' for source '%s'",
                source.package,
                source.name,
            )
            return

        # Fallback for broken top-level __init__.py files: treat the package as a
        # namespace package so submodules can still be imported and register atoms.
        pkg = types.ModuleType(source.package)
        pkg.__path__ = [str(package_root)]  # type: ignore[attr-defined]
        pkg.__package__ = source.package
        sys.modules[source.package] = pkg
        pkg_path = pkg.__path__
        fallback_scan_root = package_root

    # Walk submodules to trigger @register_atom decorators
    if fallback_scan_root is not None:
        for py_file in sorted(fallback_scan_root.rglob("*.py")):
            if py_file.name == "__init__.py":
                continue
            module_parts = py_file.relative_to(root).with_suffix("").parts
            modname = ".".join(module_parts)
            _import_module_from_file(modname, py_file)
        return

    if pkg_path is None:
        return

    import pkgutil

    for _importer, modname, _ispkg in pkgutil.walk_packages(
        pkg_path,
        prefix=source.package + ".",
        onerror=lambda name: logger.debug(
            "Failed to enumerate %s for source %s", name, source.name, exc_info=True
        ),
    ):
        try:
            importlib.import_module(modname)
        except Exception:
            logger.debug("Failed to import %s, skipping", modname, exc_info=True)


def _import_module_from_file(modname: str, path: Path) -> None:
    """Import *modname* directly from *path*, creating namespace parents as needed."""
    import importlib.util

    parent_parts = modname.split(".")[:-1]
    for idx in range(len(parent_parts)):
        pkg_name = ".".join(parent_parts[: idx + 1])
        if pkg_name in sys.modules:
            continue
        pkg_path = path.parents[len(parent_parts) - idx - 1]
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(pkg_path)]  # type: ignore[attr-defined]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg

    spec = importlib.util.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        logger.debug("Failed to import %s from %s, skipping", modname, path, exc_info=True)


def import_all_sources(
    config: SourcesConfig | None = None,
    base_dir: Path | None = None,
) -> None:
    """Import atom packages from all configured sources."""
    if config is None:
        config = load_sources()
    for source in config.sources:
        import_atoms(source, base_dir)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _git_clone(url: str, dest: Path, ref: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning %s -> %s (ref=%s)", url, dest, ref)
    subprocess.check_call(
        ["git", "clone", "--branch", ref, "--single-branch", url, str(dest)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _git_fetch_checkout(repo: Path, ref: str) -> None:
    logger.info("Fetching and checking out ref=%s in %s", ref, repo)
    subprocess.check_call(
        ["git", "fetch", "origin", ref],
        cwd=str(repo),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.check_call(
        ["git", "checkout", f"origin/{ref}"],
        cwd=str(repo),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
