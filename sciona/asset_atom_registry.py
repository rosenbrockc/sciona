"""Deterministic lookup of registered atom identifiers from provider repos."""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from sciona.atom_identity import candidate_atom_provider_roots

_ARTIFACT_ROOT_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("src", "sciona", "atoms"),
    ("sciona", "atoms"),
)
_PY_FILE_STEM_OMIT = {"__init__", "atoms"}


def _artifact_roots_for_repo(repo_root: Path) -> tuple[Path, ...]:
    roots: list[Path] = []
    for relative in _ARTIFACT_ROOT_CANDIDATES:
        candidate = repo_root.joinpath(*relative)
        if candidate.exists():
            roots.append(candidate.resolve())
    return tuple(roots)


def _iter_python_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def _callable_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _register_atom_name(decorator: ast.AST) -> str | None:
    if isinstance(decorator, ast.Name) and decorator.id == "register_atom":
        return ""
    if not isinstance(decorator, ast.Call):
        return None
    if not isinstance(decorator.func, ast.Name) or decorator.func.id != "register_atom":
        return None
    for keyword in decorator.keywords:
        if keyword.arg != "name":
            continue
        try:
            value = ast.literal_eval(keyword.value)
        except Exception:
            value = None
        if isinstance(value, str):
            return value.strip()
    return ""


def _namespace_prefix_for_artifact_root(artifact_root: Path) -> tuple[str, ...]:
    parts = artifact_root.parts
    if len(parts) >= 3 and parts[-3:] == ("src", "sciona", "atoms"):
        return ("sciona", "atoms")
    if len(parts) >= 2 and parts[-2:] == ("sciona", "atoms"):
        return ("sciona", "atoms")
    return tuple()


def _module_name_for_file(py_file: Path, artifact_root: Path) -> str:
    prefix = _namespace_prefix_for_artifact_root(artifact_root)
    rel_parts = list(py_file.relative_to(artifact_root).with_suffix("").parts)
    if rel_parts and rel_parts[-1] in _PY_FILE_STEM_OMIT:
        rel_parts = rel_parts[:-1]
    return ".".join((*prefix, *rel_parts))


def _registered_identifiers_for_file(py_file: Path, artifact_root: Path) -> set[str]:
    try:
        source_text = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source_text, filename=str(py_file))
    except Exception:
        return set()
    module_name = _module_name_for_file(py_file, artifact_root)
    identifiers: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        spec: str | None = None
        for decorator in node.decorator_list:
            spec = _register_atom_name(decorator)
            if spec is not None:
                break
        if spec is None:
            continue
        registration_name = spec or node.name
        for name in (registration_name, node.name):
            token = str(name or "").strip()
            if not token:
                continue
            identifiers.add(token)
            if "." in token:
                identifiers.add(token.rsplit(".", 1)[-1])
        if module_name and "." not in registration_name:
            identifiers.add(f"{module_name}.{registration_name}")
            identifiers.add(f"{module_name}.{node.name}")
    return identifiers


@lru_cache(maxsize=1)
def registered_atom_identifiers() -> frozenset[str]:
    """Return short and fully-qualified identifiers for registered atoms."""
    identifiers: set[str] = set()
    for repo_root in candidate_atom_provider_roots():
        resolved_root = Path(repo_root).expanduser().resolve()
        for artifact_root in _artifact_roots_for_repo(resolved_root):
            for py_file in _iter_python_files(artifact_root):
                identifiers.update(_registered_identifiers_for_file(py_file, artifact_root))
    return frozenset(sorted(identifiers))


def unknown_registered_atom_references(references: Iterable[str]) -> tuple[str, ...]:
    """Return sorted asset references that do not resolve to a registered atom."""
    available = registered_atom_identifiers()
    unknown = {
        token
        for raw in references
        if (token := str(raw or "").strip()) and token not in available
    }
    return tuple(sorted(unknown))


def clear_registered_atom_identifier_cache() -> None:
    """Clear the cached AST-scanned atom identifier set."""
    registered_atom_identifiers.cache_clear()
