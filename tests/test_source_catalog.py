"""Tests for deriving architect primitives from configured atom sources."""

from __future__ import annotations

import sys
from pathlib import Path

from ageom.architect.catalog import PrimitiveCatalog
from ageom.architect.source_catalog import seed_catalog_from_sources
from ageom.sources import AtomSource, SourcesConfig


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_seed_catalog_from_sources_uses_shared_registry_and_cdg_metadata(tmp_path: Path):
    repo = tmp_path / "repo"

    _write(repo / "sharedpkg" / "__init__.py", "")
    _write(repo / "sharedpkg" / "ghost" / "__init__.py", "")
    _write(
        repo / "sharedpkg" / "ghost" / "registry.py",
        """
from __future__ import annotations

REGISTRY = {}

def register_atom(witness, *, name=None):
    def decorator(func):
        atom_name = name or func.__name__
        REGISTRY[atom_name] = {"impl": func, "witness": witness}
        return func
    return decorator
""".strip()
        + "\n",
    )

    _write(repo / "mypkg" / "__init__.py", "")
    _write(
        repo / "mypkg" / "atoms.py",
        """
from __future__ import annotations

from sharedpkg.ghost.registry import register_atom

def witness_detect_peaks(signal: "AbstractSignal") -> "AbstractPeaks":
    return signal

@register_atom(witness_detect_peaks)
def detect_peaks(signal: "np.ndarray") -> "np.ndarray":
    return signal
""".strip()
        + "\n",
    )

    _write(
        repo / "pipeline_cdg.json",
        """
{
  "nodes": [
    {
      "node_id": "detect_peaks",
      "name": "DetectPeaks",
      "description": "Detect salient peaks from a filtered waveform.",
      "concept_type": "signal_transform",
      "inputs": [{"name": "signal", "type_desc": "np.ndarray", "constraints": "1D waveform"}],
      "outputs": [{"name": "peaks", "type_desc": "np.ndarray", "constraints": "peak indices"}],
      "status": "atomic",
      "type_signature": "(signal: np.ndarray) -> np.ndarray"
    }
  ]
}
""".strip()
        + "\n",
    )

    config = SourcesConfig(
        sources=[AtomSource(name="demo-source", package="mypkg", path="repo")]
    )
    catalog = PrimitiveCatalog()

    before_modules = set(sys.modules)
    try:
        added = seed_catalog_from_sources(catalog, config=config, base_dir=tmp_path)
        assert added == 1

        prim = catalog.get("detect_peaks")
        assert prim is not None
        assert prim.source == "demo-source"
        assert prim.category.value == "signal_transform"
        assert prim.description == "Detect salient peaks from a filtered waveform."
        assert prim.inputs[0].name == "signal"
        assert prim.outputs[0].name == "peaks"
        assert prim.type_signature == "(signal: np.ndarray) -> np.ndarray"

        aliased = catalog.get("DetectPeaks")
        assert aliased is not None
        assert aliased.name == "detect_peaks"
    finally:
        for name in set(sys.modules) - before_modules:
            if name == "mypkg" or name.startswith("mypkg.") or name == "sharedpkg" or name.startswith("sharedpkg."):
                sys.modules.pop(name, None)


def test_seed_catalog_marks_defaulted_parameters_optional(tmp_path: Path):
    repo = tmp_path / "repo"

    _write(repo / "sharedpkg" / "__init__.py", "")
    _write(repo / "sharedpkg" / "ghost" / "__init__.py", "")
    _write(
        repo / "sharedpkg" / "ghost" / "registry.py",
        """
from __future__ import annotations

REGISTRY = {}

def register_atom(witness, *, name=None):
    def decorator(func):
        atom_name = name or func.__name__
        REGISTRY[atom_name] = {"impl": func, "witness": witness}
        return func
    return decorator
""".strip()
        + "\n",
    )

    _write(repo / "mypkg" / "__init__.py", "")
    _write(
        repo / "mypkg" / "atoms.py",
        """
from __future__ import annotations

from sharedpkg.ghost.registry import register_atom

def witness_smooth(signal: "AbstractSignal", window: int = 5, method: str = "median") -> "AbstractSignal":
    return signal

@register_atom(witness_smooth)
def smooth(signal: "np.ndarray", window: int = 5, method: str = "median") -> "np.ndarray":
    return signal
""".strip()
        + "\n",
    )

    config = SourcesConfig(
        sources=[AtomSource(name="demo-source", package="mypkg", path="repo")]
    )
    catalog = PrimitiveCatalog()

    before_modules = set(sys.modules)
    try:
        added = seed_catalog_from_sources(catalog, config=config, base_dir=tmp_path)
        assert added == 1
        prim = catalog.get("smooth")
        assert prim is not None
        assert [port.name for port in prim.inputs] == ["signal", "window", "method"]
        assert prim.inputs[0].required is True
        assert prim.inputs[1].required is False
        assert prim.inputs[1].default_value_repr == "5"
        assert prim.inputs[2].required is False
        assert prim.inputs[2].default_value_repr == "'median'"
    finally:
        for name in set(sys.modules) - before_modules:
            if name == "mypkg" or name.startswith("mypkg.") or name == "sharedpkg" or name.startswith("sharedpkg."):
                sys.modules.pop(name, None)


def test_seed_catalog_from_sources_falls_back_to_ast_when_imports_are_broken(tmp_path: Path):
    repo = tmp_path / "repo"
    _write(repo / "brokenpkg" / "__init__.py", "from brokenpkg import missing_dependency\n")
    _write(
        repo / "brokenpkg" / "atoms.py",
        """
from __future__ import annotations

from ageoa.ghost.registry import register_atom

def witness_derive_feature(signal: "AbstractSignal", threshold: float = 0.5) -> "AbstractSignal":
    return signal

@register_atom(witness_derive_feature)
def derive_feature(signal: "np.ndarray", threshold: float = 0.5) -> "np.ndarray":
    \"\"\"Compute a derived feature with an optional threshold.\"\"\"
    return signal
""".strip()
        + "\n",
    )

    config = SourcesConfig(
        sources=[AtomSource(name="broken-source", package="brokenpkg", path="repo")]
    )
    catalog = PrimitiveCatalog()

    before_modules = set(sys.modules)
    try:
        added = seed_catalog_from_sources(catalog, config=config, base_dir=tmp_path)
        assert added == 1
        prim = catalog.get("derive_feature")
        assert prim is not None
        assert prim.description == "Compute a derived feature with an optional threshold."
        assert prim.inputs[0].name == "signal"
        assert prim.inputs[0].required is True
        assert prim.inputs[1].name == "threshold"
        assert prim.inputs[1].required is False
        assert prim.inputs[1].default_value_repr == "0.5"
    finally:
        for name in set(sys.modules) - before_modules:
            if name == "brokenpkg" or name.startswith("brokenpkg."):
                sys.modules.pop(name, None)
