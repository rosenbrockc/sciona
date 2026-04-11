from __future__ import annotations

import argparse
from pathlib import Path

from sciona.architect.catalog import PrimitiveCatalog
from sciona.architect.models import (
    AlgorithmicPrimitive,
    ConceptType,
    IOSpec,
    PrimitiveParamSpec,
)
from sciona.commands._helpers import _load_architect_catalog
from sciona.sources import AtomSource, SourcesConfig


def _primitive(name: str, source: str) -> AlgorithmicPrimitive:
    return AlgorithmicPrimitive(
        name=name,
        source=source,
        category=ConceptType.SIGNAL_TRANSFORM,
        description=f"{name} description",
        inputs=[IOSpec(name="x", type_desc="float")],
        outputs=[IOSpec(name="y", type_desc="float")],
    )


def test_load_architect_catalog_sources_only_ignores_saved_catalogs(
    monkeypatch, tmp_path: Path
) -> None:
    skill_index_dir = tmp_path / "skill_index"
    skill_index_dir.mkdir()

    stale_catalog = PrimitiveCatalog()
    stale_catalog.add(_primitive("stale_catalog_primitive", "stale"))
    stale_catalog.save(skill_index_dir / "catalog_stale.json")

    def _seed(catalog, **kwargs):
        catalog.add(_primitive("source_primitive", "source"))
        return 1

    monkeypatch.setattr("sciona.sources.load_sources", lambda path=None: object())
    monkeypatch.setattr("sciona.architect.source_catalog.seed_catalog_from_sources", _seed)

    config = argparse.Namespace(skill_index_dir=skill_index_dir, sources_file=tmp_path / "sources.yml")
    args = argparse.Namespace(catalog=None, sources_only=True)

    catalog, _alignment = _load_architect_catalog(args, config)

    assert catalog.get("source_primitive") is not None
    assert catalog.get("stale_catalog_primitive") is None


def test_load_architect_catalog_default_includes_saved_catalogs(
    monkeypatch, tmp_path: Path
) -> None:
    skill_index_dir = tmp_path / "skill_index"
    skill_index_dir.mkdir()

    stale_catalog = PrimitiveCatalog()
    stale_catalog.add(_primitive("stale_catalog_primitive", "stale"))
    stale_catalog.save(skill_index_dir / "catalog_stale.json")

    monkeypatch.setattr("sciona.sources.load_sources", lambda path=None: object())
    monkeypatch.setattr(
        "sciona.architect.source_catalog.seed_catalog_from_sources",
        lambda catalog, **kwargs: 0,
    )

    config = argparse.Namespace(skill_index_dir=skill_index_dir, sources_file=tmp_path / "sources.yml")
    args = argparse.Namespace(catalog=None, sources_only=False)

    catalog, _alignment = _load_architect_catalog(args, config)

    assert catalog.get("stale_catalog_primitive") is not None


def test_load_architect_catalog_attaches_tunables_from_non_ageoa_source(
    monkeypatch, tmp_path: Path
) -> None:
    skill_index_dir = tmp_path / "skill_index"
    skill_index_dir.mkdir()

    provider_root = tmp_path / "provider"
    manifest_path = provider_root / "data" / "hyperparams" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")

    def _seed(catalog, **kwargs):
        catalog.add(_primitive("source_primitive", "source"))
        return 1

    monkeypatch.setattr(
        "sciona.sources.load_sources",
        lambda path=None: SourcesConfig(
            sources=[
                AtomSource(
                    name="federated-provider",
                    package="sciona.atoms.demo",
                    path="./provider",
                )
            ]
        ),
    )
    monkeypatch.setattr(
        "sciona.architect.source_catalog.seed_catalog_from_sources",
        _seed,
    )
    monkeypatch.setattr(
        "sciona.sources.resolve_source",
        lambda source, base_dir=None: provider_root,
    )
    monkeypatch.setattr(
        "sciona.architect.hyperparams.load_manifest",
        lambda path: {
            "source_primitive": [
                PrimitiveParamSpec(name="window", kind="int", default=5),
            ]
        },
    )
    monkeypatch.setattr(
        "sciona.architect.hyperparams.get_runtime_signal_event_rate_params",
        lambda: {},
    )

    config = argparse.Namespace(
        skill_index_dir=skill_index_dir,
        sources_file=tmp_path / "sources.yml",
    )
    args = argparse.Namespace(catalog=None, sources_only=True)

    catalog, _alignment = _load_architect_catalog(args, config)

    primitive = catalog.get("source_primitive")
    assert primitive is not None
    assert [param.name for param in primitive.tunable_params] == ["window"]


def test_load_architect_catalog_merges_tunables_from_multiple_sources(
    monkeypatch, tmp_path: Path
) -> None:
    skill_index_dir = tmp_path / "skill_index"
    skill_index_dir.mkdir()

    first_root = tmp_path / "provider-one"
    second_root = tmp_path / "provider-two"
    first_manifest = first_root / "data" / "hyperparams" / "manifest.json"
    second_manifest = second_root / "data" / "hyperparams" / "manifest.json"
    first_manifest.parent.mkdir(parents=True, exist_ok=True)
    second_manifest.parent.mkdir(parents=True, exist_ok=True)
    first_manifest.write_text("{}", encoding="utf-8")
    second_manifest.write_text("{}", encoding="utf-8")

    def _seed(catalog, **kwargs):
        catalog.add(_primitive("first_primitive", "source-one"))
        catalog.add(_primitive("second_primitive", "source-two"))
        return 2

    monkeypatch.setattr(
        "sciona.sources.load_sources",
        lambda path=None: SourcesConfig(
            sources=[
                AtomSource(
                    name="provider-one",
                    package="sciona.atoms.demo.one",
                    path="./provider-one",
                ),
                AtomSource(
                    name="provider-two",
                    package="ageoa.demo.two",
                    path="./provider-two",
                ),
            ]
        ),
    )
    monkeypatch.setattr(
        "sciona.architect.source_catalog.seed_catalog_from_sources",
        _seed,
    )
    monkeypatch.setattr(
        "sciona.sources.resolve_source",
        lambda source, base_dir=None: (
            first_root if source.name == "provider-one" else second_root
        ),
    )

    def _load_manifest(path: Path):
        if path == first_manifest:
            return {
                "first_primitive": [
                    PrimitiveParamSpec(name="alpha", kind="float", default=0.1),
                ]
            }
        if path == second_manifest:
            return {
                "second_primitive": [
                    PrimitiveParamSpec(name="beta", kind="int", default=2),
                ]
            }
        raise AssertionError(f"Unexpected manifest path: {path}")

    monkeypatch.setattr("sciona.architect.hyperparams.load_manifest", _load_manifest)
    monkeypatch.setattr(
        "sciona.architect.hyperparams.get_runtime_signal_event_rate_params",
        lambda: {},
    )

    config = argparse.Namespace(
        skill_index_dir=skill_index_dir,
        sources_file=tmp_path / "sources.yml",
    )
    args = argparse.Namespace(catalog=None, sources_only=True)

    catalog, _alignment = _load_architect_catalog(args, config)

    first = catalog.get("first_primitive")
    second = catalog.get("second_primitive")
    assert first is not None
    assert second is not None
    assert [param.name for param in first.tunable_params] == ["alpha"]
    assert [param.name for param in second.tunable_params] == ["beta"]
