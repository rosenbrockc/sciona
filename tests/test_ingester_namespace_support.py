from __future__ import annotations

from pathlib import Path

from sciona.ingester import emitter
from sciona.sources import AtomSource, SourcesConfig


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_resolve_ghost_abstract_path_prefers_configured_src_layout_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "sciona-atoms"
    abstract_path = (
        repo
        / "src"
        / "sciona"
        / "atoms"
        / "signal_processing"
        / "biosppy"
        / "ghost"
        / "abstract.py"
    )
    _write(
        abstract_path,
        """
class AbstractArray: ...
class AbstractSignal: ...
""".strip()
        + "\n",
    )

    monkeypatch.setattr(
        emitter,
        "load_sources",
        lambda: SourcesConfig(
            sources=[
                AtomSource(
                    name="pilot",
                    package="sciona.atoms.signal_processing.biosppy",
                    path=str(repo),
                    python_path="src",
                )
            ]
        ),
    )
    monkeypatch.setattr(emitter, "candidate_atom_provider_roots", lambda: tuple())

    resolved = emitter._resolve_ghost_abstract_path(
        "sciona.atoms.signal_processing.biosppy"
    )

    assert resolved == abstract_path


def test_resolve_ghost_abstract_path_falls_back_to_src_provider_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "sciona-atoms"
    abstract_path = (
        repo_root
        / "src"
        / "sciona"
        / "atoms"
        / "signal_processing"
        / "biosppy"
        / "ghost"
        / "abstract.py"
    )
    _write(
        abstract_path,
        """
class AbstractArray: ...
class AbstractDistribution: ...
""".strip()
        + "\n",
    )

    monkeypatch.setattr(emitter, "load_sources", lambda: SourcesConfig())
    monkeypatch.setattr(
        emitter,
        "candidate_atom_provider_roots",
        lambda: (repo_root,),
    )

    resolved = emitter._resolve_ghost_abstract_path(
        "sciona.atoms.signal_processing.biosppy"
    )

    assert resolved == abstract_path


def test_resolve_ghost_abstract_path_keeps_legacy_ageoa_compatibility(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "ageo-atoms"
    abstract_path = repo_root / "ageoa" / "ghost" / "abstract.py"
    _write(
        abstract_path,
        """
class AbstractArray: ...
class AbstractScalar: ...
""".strip()
        + "\n",
    )

    monkeypatch.setattr(emitter, "load_sources", lambda: SourcesConfig())
    monkeypatch.setattr(
        emitter,
        "candidate_atom_provider_roots",
        lambda: (repo_root,),
    )

    resolved = emitter._resolve_ghost_abstract_path("ageoa")

    assert resolved == abstract_path


def test_available_ghost_abstract_names_parses_fallback_file_when_import_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "sciona-atoms"
    abstract_path = (
        repo
        / "src"
        / "sciona"
        / "atoms"
        / "signal_processing"
        / "biosppy"
        / "ghost"
        / "abstract.py"
    )
    _write(
        abstract_path,
        """
class AbstractArray: ...
class AbstractSignal: ...
class Helper: ...
""".strip()
        + "\n",
    )

    monkeypatch.setattr(emitter, "load_sources", lambda: SourcesConfig())
    monkeypatch.setattr(
        emitter,
        "candidate_atom_provider_roots",
        lambda: (repo,),
    )

    def fake_import_module(name: str, package: str | None = None):
        if name == "sciona.atoms.signal_processing.biosppy.ghost.abstract":
            raise ImportError(name)
        return original_import_module(name, package)

    original_import_module = emitter.importlib.import_module
    monkeypatch.setattr(emitter.importlib, "import_module", fake_import_module)

    names = emitter._available_ghost_abstract_names(
        ghost_package_root="sciona.atoms.signal_processing.biosppy"
    )

    assert names == {"AbstractArray", "AbstractSignal"}
