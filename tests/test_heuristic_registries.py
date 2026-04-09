from __future__ import annotations

import pytest

from sciona.heuristic_registries import (
    HeuristicFamilyRegistry,
    HeuristicRegistryAudit,
    HeuristicRegistryEntry,
    heuristic_registry_summary,
    load_local_heuristic_registries,
    load_local_heuristic_registries_by_family,
    resolve_local_heuristic_registry,
)
from sciona.heuristics import HeuristicActionClass, HeuristicProducerKind


def test_loads_reference_registries_for_signal_and_non_signal_families() -> None:
    by_family = load_local_heuristic_registries_by_family()
    assert "signal_event_rate" in by_family
    assert "divide_and_conquer" in by_family


def test_signal_registry_uses_canonical_heuristics_and_generic_actions() -> None:
    registry = resolve_local_heuristic_registry("signal_event_rate")
    assert registry is not None
    entry_ids = {entry.heuristic_id for entry in registry.entries}
    assert "interval_instability" in entry_ids
    interval_entry = next(
        entry for entry in registry.entries if entry.heuristic_id == "interval_instability"
    )
    assert interval_entry.action_priority[0] == HeuristicActionClass.INSERT_CORRECTION


def test_signal_registry_resolves_from_skeleton_family_alias() -> None:
    registry = resolve_local_heuristic_registry("signal_detect_measure")

    assert registry is not None
    assert registry.family == "signal_event_rate"
    assert "signal_detect_measure" in registry.family_aliases


def test_non_signal_registry_proves_interface_stays_generic() -> None:
    registry = resolve_local_heuristic_registry("divide_and_conquer")
    assert registry is not None
    entry_ids = {entry.heuristic_id for entry in registry.entries}
    assert "coverage_fragmentation" in entry_ids
    assert "resource_growth_instability" in entry_ids


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
    registry = next(iter(load_local_heuristic_registries()))
    summary = heuristic_registry_summary(registry)
    assert summary["family"]
    assert summary["heuristic_count"] > 0
