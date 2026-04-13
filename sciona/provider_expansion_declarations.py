"""Load provider-owned expansion declaration maps from configured atom repos."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

from sciona.atom_identity import candidate_atom_provider_roots


def _module_candidates(family: str) -> tuple[tuple[str, ...], ...]:
    base = tuple(part for part in str(family or "").strip().split(".") if part)
    if not base:
        return tuple()
    module_name = base[-1]
    return (
        ("src", "sciona", "atoms", "expansion", *base[:-1], f"{module_name}.py"),
        ("src", "sciona", "atoms", "expansion", *base[:-1], f"{module_name}_registry.py"),
        ("sciona", "atoms", "expansion", *base[:-1], f"{module_name}.py"),
        ("sciona", "atoms", "expansion", *base[:-1], f"{module_name}_registry.py"),
    )


def _load_module(path: Path) -> object | None:
    spec = importlib.util.spec_from_file_location(
        "_sciona_provider_expansion_" + path.as_posix().replace("/", "_").replace(".", "_"),
        path,
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=32)
def load_provider_expansion_declarations(
    family: str,
    declaration_attr: str,
) -> dict[str, tuple[str, str, str]]:
    """Return provider-owned expansion declarations for a family.

    No matcher-local fallback is used here. If the configured atom providers do
    not expose the requested declaration map, an empty dict is returned.
    """
    family_text = str(family or "").strip()
    attr_text = str(declaration_attr or "").strip()
    if not family_text or not attr_text:
        return {}

    for provider_root in candidate_atom_provider_roots():
        root = Path(provider_root).expanduser().resolve()
        for relative in _module_candidates(family_text):
            module_path = root.joinpath(*relative)
            if not module_path.exists():
                continue
            module = _load_module(module_path)
            if module is None:
                continue
            declarations = getattr(module, attr_text, None)
            if isinstance(declarations, dict) and declarations:
                return dict(declarations)
    return {}


def clear_provider_expansion_declaration_caches() -> None:
    load_provider_expansion_declarations.cache_clear()

