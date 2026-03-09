from __future__ import annotations

import json
from pathlib import Path

import pytest

from ageom.architect.catalog import CatalogReport
from ageom.catalog_validation import run_catalog_validation
from ageom.sources import AtomSource, SourcesConfig


@pytest.mark.asyncio
async def test_run_catalog_validation_writes_report(monkeypatch, tmp_path: Path):
    sources = SourcesConfig(
        sources=[
            AtomSource(name="ageo-atoms", package="ageoa", path="../ageo-atoms"),
            AtomSource(name="hpy-atoms", package="hpyatoms", path="~/codes/hpy-atoms"),
        ]
    )

    monkeypatch.setattr("ageom.catalog_validation.load_sources", lambda path=None: sources)

    def _resolve(source, base_dir=None):
        return tmp_path / source.name

    def _seed(catalog, **kwargs):
        report = kwargs["report"]
        report.total_candidates = 11
        report.added = 7
        report.merged = 2
        report.source_breakdown = {
            "ageo-atoms": {"ast_candidates": 8, "added": 5},
            "hpy-atoms": {"ast_candidates": 3, "added": 2},
        }
        return 7

    for name in ("ageo-atoms", "hpy-atoms"):
        (tmp_path / name).mkdir()

    monkeypatch.setattr("ageom.catalog_validation.resolve_source", _resolve)
    monkeypatch.setattr("ageom.catalog_validation.seed_catalog_from_sources", _seed)
    monkeypatch.setattr(
        "ageom.catalog_validation.audit_source_registration_alignment",
        lambda **kwargs: {
            "source_count": 2,
            "matched_total": 7,
            "registry_only_total": 1,
            "ast_only_total": 0,
            "drift_sources": ["hpy-atoms"],
            "registry_error_sources": [],
            "rows": [],
        },
    )

    summary = await run_catalog_validation(tmp_path)

    assert summary["status"] == "passed"
    assert summary["configured_sources"] == 2
    assert summary["resolved_sources"] == 2
    assert summary["source_candidates"] == 11
    assert summary["source_added"] == 7
    assert summary["violations"] == []
    assert "resolved=2/2" in summary["coverage_summary"]
    assert "registry_only=1" in summary["alignment_summary"]
    assert Path(summary["report"]).exists()
    payload = json.loads(Path(summary["report"]).read_text(encoding="utf-8"))
    assert payload["source_breakdown"]["ageo-atoms"]["ast_candidates"] == 8
    assert payload["alignment"]["registry_only_total"] == 1


@pytest.mark.asyncio
async def test_run_catalog_validation_flags_missing_and_zero_candidate_sources(
    monkeypatch, tmp_path: Path
):
    sources = SourcesConfig(
        sources=[
            AtomSource(name="ageo-atoms", package="ageoa", path="../ageo-atoms"),
            AtomSource(name="missing-atoms", package="missingatoms", path="~/codes/missing-atoms"),
        ]
    )

    monkeypatch.setattr("ageom.catalog_validation.load_sources", lambda path=None: sources)

    def _resolve(source, base_dir=None):
        return tmp_path / source.name

    def _seed(catalog, **kwargs):
        report: CatalogReport = kwargs["report"]
        report.total_candidates = 3
        report.added = 3
        report.source_breakdown = {
            "ageo-atoms": {"ast_candidates": 3, "added": 3},
            "missing-atoms": {"ast_candidates": 0, "added": 0},
        }
        return 3

    (tmp_path / "ageo-atoms").mkdir()
    monkeypatch.setattr("ageom.catalog_validation.resolve_source", _resolve)
    monkeypatch.setattr("ageom.catalog_validation.seed_catalog_from_sources", _seed)
    monkeypatch.setattr(
        "ageom.catalog_validation.audit_source_registration_alignment",
        lambda **kwargs: {
            "source_count": 2,
            "matched_total": 3,
            "registry_only_total": 0,
            "ast_only_total": 1,
            "drift_sources": ["missing-atoms"],
            "registry_error_sources": ["missing-atoms"],
            "rows": [],
        },
    )

    summary = await run_catalog_validation(tmp_path)

    assert summary["status"] == "failed"
    assert "missing_source:missing-atoms" in summary["violations"]
    assert "source_no_candidates:missing-atoms" in summary["violations"]
    assert summary["missing_sources"] == ["missing-atoms"]
    assert summary["zero_candidate_sources"] == ["missing-atoms"]
    assert summary["alignment"]["registry_error_sources"] == ["missing-atoms"]
    assert "missing=1" in summary["coverage_summary"]
    assert "drift=1" in summary["alignment_summary"]
