"""Tests for deriving architect primitives from configured atom sources."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ageom.architect.catalog import CatalogReport, PrimitiveCatalog
from ageom.architect.source_catalog import seed_catalog_from_sources
from ageom.sources import AtomSource, SourcesConfig
from ageom.types import Declaration, Prover


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
    report = CatalogReport()

    before_modules = set(sys.modules)
    try:
        added = seed_catalog_from_sources(catalog, config=config, base_dir=tmp_path, report=report)
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
        assert report.source_live_registry_candidates == 1
        assert report.source_cdg_metadata_matches == 1
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


def test_seed_catalog_prefers_witness_doc_for_live_registry_metadata(tmp_path: Path):
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

    _write(repo / "richpkg" / "__init__.py", "")
    _write(
        repo / "richpkg" / "atoms.py",
        """
from __future__ import annotations

from sharedpkg.ghost.registry import register_atom

def witness_bandpass(signal: "AbstractSignal", lowcut: float, highcut: float) -> "AbstractSignal":
    \"\"\"Apply a stable bandpass stage to the signal.\"\"\"
    return signal

@register_atom(witness_bandpass)
def bandpass_filter(signal: "np.ndarray", lowcut: float, highcut: float) -> "np.ndarray":
    return signal
""".strip()
        + "\n",
    )

    config = SourcesConfig(
        sources=[AtomSource(name="rich-source", package="richpkg", path="repo")]
    )
    catalog = PrimitiveCatalog()
    report = CatalogReport()

    before_modules = set(sys.modules)
    try:
        added = seed_catalog_from_sources(catalog, config=config, base_dir=tmp_path, report=report)
        assert added == 1
        prim = catalog.get("bandpass_filter")
        assert prim is not None
        assert prim.description == "Apply a stable bandpass stage to the signal."
        assert [port.name for port in prim.inputs] == ["signal", "lowcut", "highcut"]
        assert report.source_live_registry_candidates == 1
        assert report.source_witness_doc_fallbacks == 1
    finally:
        for name in set(sys.modules) - before_modules:
            if name == "richpkg" or name.startswith("richpkg.") or name == "sharedpkg" or name.startswith("sharedpkg."):
                sys.modules.pop(name, None)


def test_seed_catalog_uses_witness_signature_for_ast_fallback_when_wrapper_is_generic(tmp_path: Path):
    repo = tmp_path / "repo"
    _write(repo / "brokenpkg" / "__init__.py", "from brokenpkg import missing_dependency\n")
    _write(
        repo / "brokenpkg" / "atoms.py",
        """
from __future__ import annotations

from ageoa.ghost.registry import register_atom

def witness_fft_transform(signal: "AbstractSignal", axis: int = -1) -> "AbstractSpectrum":
    \"\"\"Compute a forward spectral transform on the signal.\"\"\"
    return signal

@register_atom(witness_fft_transform)
def fft_transform(*args, **kwargs) -> object:
    return None
""".strip()
        + "\n",
    )

    config = SourcesConfig(
        sources=[AtomSource(name="broken-source", package="brokenpkg", path="repo")]
    )
    catalog = PrimitiveCatalog()
    report = CatalogReport()

    before_modules = set(sys.modules)
    try:
        added = seed_catalog_from_sources(
            catalog,
            config=config,
            base_dir=tmp_path,
            include_live_registries=False,
            report=report,
        )
        assert added == 1
        prim = catalog.get("fft_transform")
        assert prim is not None
        assert prim.description == "Compute a forward spectral transform on the signal."
        assert [port.name for port in prim.inputs] == ["signal", "axis"]
        assert prim.inputs[1].required is False
        assert prim.inputs[1].default_value_repr == "-1"
        assert prim.outputs[0].type_desc == "AbstractSpectrum"
        assert report.source_ast_candidates == 1
        assert report.source_witness_doc_fallbacks == 1
        assert report.source_witness_signature_fallbacks == 1
    finally:
        for name in set(sys.modules) - before_modules:
            if name == "brokenpkg" or name.startswith("brokenpkg."):
                sys.modules.pop(name, None)


def test_seed_catalog_adds_suffix_aliases_for_dotted_registration_names(tmp_path: Path):
    repo = tmp_path / "repo"
    _write(repo / "brokenpkg" / "__init__.py", "from brokenpkg import missing_dependency\n")
    _write(
        repo / "brokenpkg" / "atoms.py",
        """
from __future__ import annotations

from ageoa.ghost.registry import register_atom

def witness_linear_solve(matrix: "AbstractMatrix", rhs: "AbstractVector") -> "AbstractVector":
    return rhs

@register_atom(witness_linear_solve, name="scipy.linalg.solve")
def solve_impl(matrix: object, rhs: object) -> object:
    return rhs
""".strip()
        + "\n",
    )

    config = SourcesConfig(
        sources=[AtomSource(name="broken-source", package="brokenpkg", path="repo")]
    )
    catalog = PrimitiveCatalog()

    before_modules = set(sys.modules)
    try:
        added = seed_catalog_from_sources(
            catalog,
            config=config,
            base_dir=tmp_path,
            include_live_registries=False,
        )
        assert added == 1
        assert catalog.get("scipy.linalg.solve") is not None
        assert catalog.get("solve") is not None
        assert catalog.get("linalg.solve") is not None
        assert catalog.get("linalg solve") is not None
        assert catalog.get("linalg_solve") is not None
        assert catalog.get("solve").name == "scipy.linalg.solve"
    finally:
        for name in set(sys.modules) - before_modules:
            if name == "brokenpkg" or name.startswith("brokenpkg."):
                sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# Mock SkillIndex for dedup integration tests
# ---------------------------------------------------------------------------


class _MockSkillIndex:
    """Returns high similarity for a specific incumbent name."""

    def __init__(self, incumbent_name: str, score: float):
        self._name = incumbent_name
        self._score = score

    def search_by_embedding(self, query_text: str, k: int = 1):
        decl = Declaration(name=self._name, type_signature="", prover=Prover.LEAN4)
        return [(decl, self._score)]

    def _primitive_to_text(self, prim):
        return f"{prim.name}: {prim.description}"

    def add_primitive(self, primitive):
        pass


# ---------------------------------------------------------------------------
# Task 4: dedup integration in seed_catalog_from_sources
# ---------------------------------------------------------------------------


def test_seed_catalog_dedup_merges_similar_primitives(tmp_path: Path):
    """Two sources with semantically identical atoms under different names."""
    repo = tmp_path / "repo"

    # Source A: atom named "detect_peaks"
    _write(repo / "pkga" / "__init__.py", "")
    _write(
        repo / "pkga" / "atoms.py",
        (
            "from __future__ import annotations\n"
            "from ageoa.ghost.registry import register_atom\n"
            "@register_atom(None)\n"
            "def detect_peaks(signal: 'np.ndarray') -> 'np.ndarray':\n"
            "    '''Detect salient peaks from a waveform.'''\n"
            "    return signal\n"
        ),
    )

    # Source B: atom named "find_peaks" — semantically same
    _write(repo / "pkgb" / "__init__.py", "")
    _write(
        repo / "pkgb" / "atoms.py",
        (
            "from __future__ import annotations\n"
            "from ageoa.ghost.registry import register_atom\n"
            "@register_atom(None)\n"
            "def find_peaks(signal: 'np.ndarray') -> 'np.ndarray':\n"
            "    '''Find peaks.'''\n"
            "    return signal\n"
        ),
    )

    config = SourcesConfig(
        sources=[
            AtomSource(name="source-a", package="pkga", path="repo"),
            AtomSource(name="source-b", package="pkgb", path="repo"),
        ]
    )
    catalog = PrimitiveCatalog()
    report = CatalogReport()

    # Pre-add detect_peaks so the mock index can find it as incumbent
    from ageom.architect.models import AlgorithmicPrimitive, ConceptType, IOSpec

    catalog.add(
        AlgorithmicPrimitive(
            name="detect_peaks",
            source="source-a",
            category=ConceptType.CUSTOM,
            description="Detect salient peaks from a waveform.",
            inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
            outputs=[IOSpec(name="result", type_desc="np.ndarray")],
        )
    )

    # Mock index returns detect_peaks as the incumbent with high similarity
    mock_idx = _MockSkillIndex("detect_peaks", 0.92)

    before_modules = set(sys.modules)
    try:
        added = seed_catalog_from_sources(
            catalog,
            config=config,
            base_dir=tmp_path,
            include_live_registries=False,
            skill_index=mock_idx,
            report=report,
        )
        # find_peaks should be merged into detect_peaks
        assert catalog.get("find_peaks") is not None
        assert catalog.get("find_peaks").name == catalog.get("detect_peaks").name
        assert report.merged >= 1
    finally:
        for name in set(sys.modules) - before_modules:
            for prefix in ("pkga", "pkgb"):
                if name == prefix or name.startswith(prefix + "."):
                    sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# Task 5: structural dedup via topo_hash
# ---------------------------------------------------------------------------


def test_seed_catalog_reports_structural_duplicates(tmp_path: Path):
    """Two sources with CDGs that have identical structure produce structural_skips."""
    repo = tmp_path / "repo"

    # Build two CDGs with identical topology: parent -> child_a, child_b
    cdg_template = {
        "nodes": [
            {
                "node_id": "root",
                "name": "Root",
                "description": "Root node",
                "status": "decomposed",
                "children": ["c1", "c2"],
                "parent_id": None,
            },
            {
                "node_id": "c1",
                "name": "Step1",
                "description": "First step",
                "status": "atomic",
                "parent_id": "root",
                "concept_type": "sorting",
                "inputs": [{"name": "x", "type_desc": "int"}],
                "outputs": [{"name": "y", "type_desc": "int"}],
                "type_signature": "int -> int",
            },
            {
                "node_id": "c2",
                "name": "Step2",
                "description": "Second step",
                "status": "atomic",
                "parent_id": "root",
                "concept_type": "sorting",
                "inputs": [{"name": "y", "type_desc": "int"}],
                "outputs": [{"name": "z", "type_desc": "int"}],
                "type_signature": "int -> int",
            },
        ],
        "edges": [
            {"source_id": "c1", "target_id": "c2"},
        ],
    }

    # Source A
    _write(repo / "pkga" / "__init__.py", "")
    _write(repo / "pkga" / "atoms.py", "")
    src_a_cdg = tmp_path / "repo" / "pkga" / "pipeline_cdg.json"
    _write(src_a_cdg, json.dumps(cdg_template))

    # Source B — identical structure
    _write(repo / "pkgb" / "__init__.py", "")
    _write(repo / "pkgb" / "atoms.py", "")
    src_b_cdg = tmp_path / "repo" / "pkgb" / "pipeline_cdg.json"
    _write(src_b_cdg, json.dumps(cdg_template))

    config = SourcesConfig(
        sources=[
            AtomSource(name="source-a", package="pkga", path="repo/pkga", cdg_glob="*.json"),
            AtomSource(name="source-b", package="pkgb", path="repo/pkgb", cdg_glob="*.json"),
        ]
    )
    catalog = PrimitiveCatalog()
    report = CatalogReport()

    before_modules = set(sys.modules)
    try:
        seed_catalog_from_sources(
            catalog,
            config=config,
            base_dir=tmp_path,
            include_live_registries=False,
            report=report,
        )
        # Both sources have the same topo_hash, so the second should be flagged
        assert report.structural_skips >= 1
    finally:
        for name in set(sys.modules) - before_modules:
            for prefix in ("pkga", "pkgb"):
                if name == prefix or name.startswith(prefix + "."):
                    sys.modules.pop(name, None)
