from __future__ import annotations

import argparse
from pathlib import Path

from ageom.architect.catalog import PrimitiveCatalog
from ageom.architect.models import AlgorithmicPrimitive, ConceptType, IOSpec
from ageom.commands._helpers import _load_architect_catalog


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

    monkeypatch.setattr("ageom.sources.load_sources", lambda path=None: object())
    monkeypatch.setattr("ageom.architect.source_catalog.seed_catalog_from_sources", _seed)

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

    monkeypatch.setattr("ageom.sources.load_sources", lambda path=None: object())
    monkeypatch.setattr(
        "ageom.architect.source_catalog.seed_catalog_from_sources",
        lambda catalog, **kwargs: 0,
    )

    config = argparse.Namespace(skill_index_dir=skill_index_dir, sources_file=tmp_path / "sources.yml")
    args = argparse.Namespace(catalog=None, sources_only=False)

    catalog, _alignment = _load_architect_catalog(args, config)

    assert catalog.get("stale_catalog_primitive") is not None
