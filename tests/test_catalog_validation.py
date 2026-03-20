from __future__ import annotations

import json
from pathlib import Path

import pytest

from sciona.architect.catalog import CatalogReport
from sciona.catalog_validation import run_catalog_validation
from sciona.sources import AtomSource, SourcesConfig


@pytest.mark.asyncio
async def test_run_catalog_validation_writes_report(monkeypatch, tmp_path: Path):
    sources = SourcesConfig(
        sources=[
            AtomSource(name="ageo-atoms", package="ageoa", path="../ageo-atoms"),
            AtomSource(name="hpy-atoms", package="hpyatoms", path="~/codes/hpy-atoms"),
        ]
    )

    monkeypatch.setattr("sciona.catalog_validation.load_sources", lambda path=None: sources)

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

    monkeypatch.setattr("sciona.catalog_validation.resolve_source", _resolve)
    monkeypatch.setattr("sciona.catalog_validation.seed_catalog_from_sources", _seed)
    monkeypatch.setattr(
        "sciona.catalog_validation.audit_source_registration_alignment",
        lambda **kwargs: {
            "source_count": 2,
            "matched_total": 7,
            "registry_only_total": 1,
            "ast_only_total": 0,
            "highest_severity": "high",
            "severity_counts": {"healthy": 1, "medium": 0, "high": 1, "critical": 0},
            "drift_sources": ["hpy-atoms"],
            "registry_error_sources": [],
            "rows": [
                {
                    "source": "hpy-atoms",
                    "severity": "high",
                    "registry_only_count": 1,
                    "ast_only_count": 0,
                }
            ],
        },
    )

    summary = await run_catalog_validation(tmp_path)

    assert summary["status"] == "passed"
    assert summary["configured_sources"] == 2
    assert summary["resolved_sources"] == 2
    assert summary["source_candidates"] == 11
    assert summary["source_added"] == 7
    assert summary["violations"] == []
    assert summary["warnings"] == ["high_alignment_drift:hpy-atoms"]
    assert summary["high_severity_sources"] == ["hpy-atoms"]
    assert summary["medium_severity_sources"] == []
    assert "resolved=2/2" in summary["coverage_summary"]
    assert "severity=high" in summary["alignment_summary"]
    assert "warnings=1 high=1 medium=0" == summary["warning_summary"]
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

    monkeypatch.setattr("sciona.catalog_validation.load_sources", lambda path=None: sources)

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
    monkeypatch.setattr("sciona.catalog_validation.resolve_source", _resolve)
    monkeypatch.setattr("sciona.catalog_validation.seed_catalog_from_sources", _seed)
    monkeypatch.setattr(
        "sciona.catalog_validation.audit_source_registration_alignment",
        lambda **kwargs: {
            "source_count": 2,
            "matched_total": 3,
            "registry_only_total": 0,
            "ast_only_total": 1,
            "highest_severity": "critical",
            "severity_counts": {"healthy": 1, "medium": 0, "high": 0, "critical": 1},
            "drift_sources": ["missing-atoms"],
            "registry_error_sources": ["missing-atoms"],
            "rows": [
                {
                    "source": "missing-atoms",
                    "severity": "critical",
                    "registry_only_count": 0,
                    "ast_only_count": 1,
                }
            ],
        },
    )

    summary = await run_catalog_validation(tmp_path)

    assert summary["status"] == "failed"
    assert "missing_source:missing-atoms" in summary["violations"]
    assert "source_no_candidates:missing-atoms" in summary["violations"]
    assert summary["missing_sources"] == ["missing-atoms"]
    assert summary["zero_candidate_sources"] == ["missing-atoms"]
    assert summary["alignment"]["registry_error_sources"] == ["missing-atoms"]
    assert summary["warnings"] == []
    assert "severity=critical" in summary["alignment_summary"]
    assert "missing=1" in summary["coverage_summary"]
    assert "drift=1" in summary["alignment_summary"]


@pytest.mark.asyncio
async def test_run_catalog_validation_fails_on_critical_alignment_drift_only(
    monkeypatch, tmp_path: Path
):
    sources = SourcesConfig(
        sources=[
            AtomSource(name="ageo-atoms", package="ageoa", path="../ageo-atoms"),
        ]
    )

    monkeypatch.setattr("sciona.catalog_validation.load_sources", lambda path=None: sources)

    def _resolve(source, base_dir=None):
        return tmp_path / source.name

    def _seed(catalog, **kwargs):
        report: CatalogReport = kwargs["report"]
        report.total_candidates = 5
        report.added = 5
        report.source_breakdown = {
            "ageo-atoms": {"ast_candidates": 5, "added": 5},
        }
        return 5

    (tmp_path / "ageo-atoms").mkdir()
    monkeypatch.setattr("sciona.catalog_validation.resolve_source", _resolve)
    monkeypatch.setattr("sciona.catalog_validation.seed_catalog_from_sources", _seed)
    monkeypatch.setattr(
        "sciona.catalog_validation.audit_source_registration_alignment",
        lambda **kwargs: {
            "source_count": 1,
            "matched_total": 4,
            "registry_only_total": 0,
            "ast_only_total": 1,
            "highest_severity": "critical",
            "severity_counts": {"healthy": 0, "medium": 0, "high": 0, "critical": 1},
            "drift_sources": ["ageo-atoms"],
            "registry_error_sources": ["ageo-atoms"],
            "rows": [
                {
                    "source": "ageo-atoms",
                    "severity": "critical",
                    "registry_only_count": 0,
                    "ast_only_count": 1,
                }
            ],
        },
    )

    summary = await run_catalog_validation(tmp_path)

    assert summary["status"] == "failed"
    assert summary["violations"] == ["critical_alignment_drift"]
    assert summary["warnings"] == []
    assert summary["missing_sources"] == []
    assert summary["zero_candidate_sources"] == []
    assert "severity=critical" in summary["alignment_summary"]
