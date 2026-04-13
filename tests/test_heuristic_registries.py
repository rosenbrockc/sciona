from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sciona.heuristic_registries import (
    EXTERNAL_ASSET_DIR_CANDIDATES,
    HeuristicFamilyRegistry,
    HeuristicRegistryAudit,
    HeuristicRegistryEntry,
    clear_heuristic_registry_caches,
    heuristic_registry_summary,
    heuristic_registry_compatibility_report,
    load_heuristic_registries,
    load_local_heuristic_registries,
    load_local_heuristic_registries_by_family,
    resolve_heuristic_registry,
    resolve_local_heuristic_registry,
)
from sciona.heuristics import HeuristicActionClass, HeuristicProducerKind


def test_loads_reference_registries_for_signal_and_non_signal_families() -> None:
    by_family = load_local_heuristic_registries_by_family()
    assert "signal_event_rate" in by_family
    assert "divide_and_conquer" in by_family


def test_signal_registry_uses_canonical_heuristics_and_generic_actions() -> None:
    registry = resolve_heuristic_registry("signal_event_rate")
    assert registry is not None
    entry_ids = {entry.heuristic_id for entry in registry.entries}
    assert "interval_instability" in entry_ids
    interval_entry = next(
        entry for entry in registry.entries if entry.heuristic_id == "interval_instability"
    )
    assert interval_entry.action_priority[0] == HeuristicActionClass.INSERT_CORRECTION


def test_signal_registry_prefers_namespace_pilot_asset_when_available() -> None:
    registry = resolve_heuristic_registry("signal_event_rate")

    assert registry is not None
    summary = heuristic_registry_summary(registry)
    assert summary["source_kind"] == "shared_asset"
    assert summary["source_repository"] == "../sciona-atoms-signal"
    assert summary["source_path"].endswith(
        "data/heuristics/families/signal_event_rate.json"
    )


def test_signal_registry_resolves_from_skeleton_family_alias() -> None:
    registry = resolve_heuristic_registry("signal_detect_measure")

    assert registry is not None
    assert registry.family == "signal_event_rate"
    assert "signal_detect_measure" in registry.family_aliases


def test_non_signal_registry_proves_interface_stays_generic() -> None:
    registry = resolve_heuristic_registry("divide_and_conquer")
    assert registry is not None
    entry_ids = {entry.heuristic_id for entry in registry.entries}
    assert "coverage_fragmentation" in entry_ids
    assert "merge_cost_pressure" in entry_ids


def test_divide_and_conquer_registry_prefers_namespace_pilot_asset_when_available() -> None:
    registry = resolve_heuristic_registry("divide_and_conquer")

    assert registry is not None
    summary = heuristic_registry_summary(registry)
    assert summary["source_kind"] == "shared_asset"
    assert summary["source_repository"] == "../sciona-atoms"
    assert summary["source_path"].endswith(
        "data/heuristics/families/divide_and_conquer.json"
    )


def test_sequential_filter_registry_resolves_from_shared_assets() -> None:
    registry = resolve_heuristic_registry("kalman_filter")

    assert registry is not None
    assert registry.family == "sequential_filter"
    entry_ids = {entry.heuristic_id for entry in registry.entries}
    assert "residual_structure_after_transform" in entry_ids
    assert "alignment_error" in entry_ids


def test_sequential_filter_registry_reports_shared_provider_asset_path() -> None:
    registry = resolve_heuristic_registry("sequential_filter")

    assert registry is not None
    summary = heuristic_registry_summary(registry)
    assert summary["source_kind"] == "shared_asset"
    assert summary["source_repository"] == "../sciona-atoms"
    assert summary["source_path"].endswith(
        "data/heuristics/families/sequential_filter.json"
    )


def test_registry_rejects_unknown_heuristic_ids() -> None:
    with pytest.raises(ValueError):
        HeuristicFamilyRegistry(
            asset_id="bad.registry",
            asset_version="v1",
            family="demo",
            name="Bad Registry",
            summary="Bad registry.",
            dejargonized_summary="Bad registry.",
            entries=[
                HeuristicRegistryEntry(
                    heuristic_id="ecg_rr_irregularity",
                    sanctioned_producer_kinds=[HeuristicProducerKind.ATOM_OUTPUT],
                    supported_action_classes=[HeuristicActionClass.GATE_OR_VALIDATE],
                )
            ],
            audit=HeuristicRegistryAudit(references=[{"title": "Reference"}]),
        )


def test_registry_rejects_duplicate_entries() -> None:
    with pytest.raises(ValueError):
        HeuristicFamilyRegistry(
            asset_id="dup.registry",
            asset_version="v1",
            family="demo",
            name="Dup Registry",
            summary="Duplicate registry.",
            dejargonized_summary="Duplicate registry.",
            entries=[
                HeuristicRegistryEntry(
                    heuristic_id="interval_instability",
                    sanctioned_producer_kinds=[HeuristicProducerKind.ATOM_OUTPUT],
                    supported_action_classes=[HeuristicActionClass.GATE_OR_VALIDATE],
                ),
                HeuristicRegistryEntry(
                    heuristic_id="interval_instability",
                    sanctioned_producer_kinds=[HeuristicProducerKind.RUNTIME_TRANSFORM],
                    supported_action_classes=[HeuristicActionClass.INSERT_CORRECTION],
                ),
            ],
            audit=HeuristicRegistryAudit(references=[{"title": "Reference"}]),
        )


def test_registry_summary_reports_family_and_count() -> None:
    registry = next(iter(load_heuristic_registries()))
    summary = heuristic_registry_summary(registry)
    assert summary["family"]
    assert summary["heuristic_count"] > 0
    assert summary["source_kind"]
    assert summary["compatibility_status"]


def _clear_registry_caches() -> None:
    clear_heuristic_registry_caches()


def _write_registry(
    asset_dir: Path,
    *,
    asset_id: str,
    asset_version: str,
    family: str,
    heuristic_id: str = "interval_instability",
    family_aliases: list[str] | None = None,
) -> None:
    asset_dir.mkdir(parents=True, exist_ok=True)
    path = asset_dir / f"{family}.json"
    path.write_text(
        json.dumps(
            {
                "asset_id": asset_id,
                "asset_version": asset_version,
                "family": family,
                "family_aliases": family_aliases or [],
                "name": f"{family} registry",
                "summary": "Cross-family heuristic registry fixture.",
                "dejargonized_summary": "Fixture registry for loader compatibility tests.",
                "entries": [
                    {
                        "heuristic_id": heuristic_id,
                        "sanctioned_producer_kinds": ["atom_output"],
                        "supported_action_classes": ["gate_or_validate"],
                        "action_priority": ["gate_or_validate"],
                    }
                ],
                "audit": {
                    "references": [
                        {
                            "title": "Fixture Reference",
                            "citation": "Synthetic fixture registry.",
                        }
                    ]
                },
            }
        )
    )


@pytest.fixture
def isolated_registry_layout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    local_dir = tmp_path / "local" / "heuristic_registries"
    atoms_root = tmp_path / "ageo-atoms"
    external_dir = atoms_root.joinpath(*EXTERNAL_ASSET_DIR_CANDIDATES[0])
    monkeypatch.setattr("sciona.heuristic_registries.ASSET_DIR", local_dir)
    monkeypatch.setattr("sciona.heuristic_registries.DEFAULT_AGEO_ATOMS_ROOT", atoms_root)
    monkeypatch.setattr(
        "sciona.heuristic_registries.candidate_atom_provider_roots",
        lambda: (atoms_root,),
    )
    monkeypatch.setenv("SCIONA_AGEO_ATOMS_ROOT", str(atoms_root))
    _clear_registry_caches()
    try:
        yield {
            "local_dir": local_dir,
            "atoms_root": atoms_root,
            "external_dir": external_dir,
        }
    finally:
        _clear_registry_caches()


def test_external_registry_is_preferred_when_present(
    isolated_registry_layout: dict[str, Path],
) -> None:
    _write_registry(
        isolated_registry_layout["local_dir"],
        asset_id="family.demo.heuristics.local",
        asset_version="v1",
        family="demo_family",
        family_aliases=["demo_alias"],
    )
    _write_registry(
        isolated_registry_layout["external_dir"],
        asset_id="family.demo.heuristics.shared",
        asset_version="v1",
        family="demo_family",
        family_aliases=["demo_alias"],
    )

    registry = resolve_heuristic_registry("demo_alias")

    assert registry is not None
    summary = heuristic_registry_summary(registry)
    assert registry.asset_id == "family.demo.heuristics.shared"
    assert summary["source_kind"] == "shared_asset"
    assert summary["source_repository"] == "../ageo-atoms"
    assert summary["compatibility_status"] == "external_preferred"


def test_external_registry_prefers_first_configured_provider_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    local_dir = tmp_path / "local" / "heuristic_registries"
    first_root = tmp_path / "provider-one"
    second_root = tmp_path / "provider-two"
    first_external_dir = first_root.joinpath(*EXTERNAL_ASSET_DIR_CANDIDATES[0])
    second_external_dir = second_root.joinpath(*EXTERNAL_ASSET_DIR_CANDIDATES[0])

    monkeypatch.setattr("sciona.heuristic_registries.ASSET_DIR", local_dir)
    monkeypatch.setenv(
        "SCIONA_ATOM_PROVIDER_ROOTS",
        os.pathsep.join((str(first_root), str(second_root))),
    )
    monkeypatch.delenv("SCIONA_AGEO_ATOMS_ROOT", raising=False)
    _clear_registry_caches()
    try:
        _write_registry(
            first_external_dir,
            asset_id="family.demo.heuristics.first",
            asset_version="v1",
            family="demo_family",
        )
        _write_registry(
            second_external_dir,
            asset_id="family.demo.heuristics.second",
            asset_version="v1",
            family="demo_family",
        )

        registry = resolve_heuristic_registry("demo_family")

        assert registry is not None
        summary = heuristic_registry_summary(registry)
        assert registry.asset_id == "family.demo.heuristics.first"
        assert summary["source_path"] == str(first_external_dir / "demo_family.json")
    finally:
        _clear_registry_caches()


def test_local_registry_falls_back_when_external_dir_missing(
    isolated_registry_layout: dict[str, Path],
) -> None:
    _write_registry(
        isolated_registry_layout["local_dir"],
        asset_id="family.demo.heuristics.local",
        asset_version="v1",
        family="demo_family",
    )

    registry = resolve_heuristic_registry("demo_family")

    assert registry is not None
    summary = heuristic_registry_summary(registry)
    assert summary["source_kind"] == "local_asset"
    assert summary["compatibility_status"] == "local_fallback"
    assert "shared heuristic registry directory not found" in " ".join(
        summary["compatibility_notes"]
    )


def test_version_mismatch_is_reported_when_both_sources_exist(
    isolated_registry_layout: dict[str, Path],
) -> None:
    _write_registry(
        isolated_registry_layout["local_dir"],
        asset_id="family.demo.heuristics.local",
        asset_version="v1",
        family="demo_family",
    )
    _write_registry(
        isolated_registry_layout["external_dir"],
        asset_id="family.demo.heuristics.shared",
        asset_version="v2",
        family="demo_family",
    )

    registry = resolve_local_heuristic_registry("demo_family")

    assert registry is not None
    summary = heuristic_registry_summary(registry)
    assert registry.asset_id == "family.demo.heuristics.shared"
    assert summary["compatibility_status"] == "version_mismatch"
    assert "asset_version mismatch" in " ".join(summary["compatibility_notes"])


def test_compatibility_report_surfaces_selected_source_and_missing_external_dir(
    isolated_registry_layout: dict[str, Path],
) -> None:
    _write_registry(
        isolated_registry_layout["local_dir"],
        asset_id="family.demo.heuristics.local",
        asset_version="v1",
        family="demo_family",
    )

    report = heuristic_registry_compatibility_report()

    assert report["external_asset_dir_exists"] is False
    assert report["record_count"] == 1
    assert report["warnings"]
    assert report["records"][0]["family"] == "demo_family"
    assert report["records"][0]["compatibility_status"] == "local_fallback"
