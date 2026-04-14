from __future__ import annotations

import json
from pathlib import Path

import pytest

from sciona.asset_atom_registry import clear_registered_atom_identifier_cache
from sciona.principal.expansion_assets import (
    clear_local_expansion_asset_caches,
    load_local_expansion_assets_by_family,
    load_local_expansion_assets,
    resolve_local_expansion_asset,
)


def _write_expansion_asset(
    asset_dir: Path,
    *,
    asset_id: str,
    asset_version: str,
    family: str,
    family_aliases: list[str] | None = None,
    source_kind: str = "local_asset",
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
                "domain": "signal_processing",
                "name": f"{family} expansion inventory",
                "summary": "Cross-family expansion fixture.",
                "operations": [
                    {
                        "rule_name": "insert_jump_removal_before_filter",
                        "name": "Insert Jump Removal Before Filter",
                        "dejargonized_summary": "Fixture expansion operation.",
                        "trigger": {
                            "metric_name": "jump_discontinuity_count",
                            "comparison": "gt",
                            "threshold": 3.0,
                            "required_runtime_keys": ["signal"],
                        },
                        "action_classes": ["precondition"],
                        "rewrite": {
                            "before_summary": "before",
                            "after_summary": "after",
                            "information_flow_effect": "fixture",
                        },
                    }
                ],
                "audit": {
                    "provenance": "fixture",
                    "source_kind": source_kind,
                    "review_status": "transitional",
                    "rationale": "Fixture expansion asset for loader preference tests.",
                    "dejargonized_summary": "Fixture expansion asset.",
                    "references": [{"title": "Fixture Reference"}],
                },
            }
        )
    )


def _write_registered_atom(provider_root: Path, *, module_name: str, atom_name: str) -> None:
    module_path = provider_root / "src" / "sciona" / "atoms" / f"{module_name}.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(
        "\n".join(
            [
                "from sciona.ghost.abstract import AbstractArray",
                "from sciona.ghost.registry import register_atom",
                "",
                f"def witness_{atom_name}(x: AbstractArray) -> AbstractArray:",
                "    return x",
                "",
                f"@register_atom(witness_{atom_name})",
                f"def {atom_name}(x):",
                "    return x",
                "",
            ]
        )
    )


@pytest.fixture
def isolated_provider_expansion_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, Path]:
    local_dir = tmp_path / "local" / "expansions"
    provider_root = tmp_path / "sciona-atoms-signal"
    provider_dir = provider_root / "data" / "expansions"
    monkeypatch.setattr("sciona.principal.expansion_assets.ASSET_DIR", local_dir)
    monkeypatch.setattr(
        "sciona.principal.expansion_assets.candidate_atom_provider_roots",
        lambda: (provider_root,),
    )
    monkeypatch.setattr(
        "sciona.asset_atom_registry.candidate_atom_provider_roots",
        lambda: (provider_root,),
    )
    clear_registered_atom_identifier_cache()
    clear_local_expansion_asset_caches()
    try:
        yield {"local_dir": local_dir, "provider_dir": provider_dir}
    finally:
        clear_registered_atom_identifier_cache()
        clear_local_expansion_asset_caches()


def test_provider_expansion_asset_prefers_external_copy(
    isolated_provider_expansion_layout: dict[str, Path],
) -> None:
    _write_expansion_asset(
        isolated_provider_expansion_layout["local_dir"],
        asset_id="family.signal_event_rate.expansions.local",
        asset_version="local.v1",
        family="signal_event_rate",
        family_aliases=["signal_detect_measure"],
        source_kind="local_asset",
    )
    _write_expansion_asset(
        isolated_provider_expansion_layout["provider_dir"],
        asset_id="family.signal_event_rate.expansions.provider",
        asset_version="provider.v1",
        family="signal_event_rate",
        family_aliases=["signal_detect_measure"],
        source_kind="shared_asset",
    )

    by_family = load_local_expansion_assets_by_family()
    asset = by_family["signal_event_rate"]

    assert asset.asset_id == "family.signal_event_rate.expansions.provider"
    assert asset.audit.source_kind == "shared_asset"


def test_provider_expansion_asset_resolves_alias_from_external_copy(
    isolated_provider_expansion_layout: dict[str, Path],
) -> None:
    _write_expansion_asset(
        isolated_provider_expansion_layout["local_dir"],
        asset_id="family.signal_event_rate.expansions.local",
        asset_version="local.v1",
        family="signal_event_rate",
        family_aliases=["signal_detect_measure"],
    )
    _write_expansion_asset(
        isolated_provider_expansion_layout["provider_dir"],
        asset_id="family.signal_event_rate.expansions.provider",
        asset_version="provider.v1",
        family="signal_event_rate",
        family_aliases=["signal_detect_measure"],
        source_kind="shared_asset",
    )

    asset = resolve_local_expansion_asset("signal_detect_measure")

    assert asset is not None
    assert asset.asset_id == "family.signal_event_rate.expansions.provider"
    assert asset.audit.source_kind == "shared_asset"


def test_provider_expansion_asset_rejects_unknown_matched_primitive(
    isolated_provider_expansion_layout: dict[str, Path],
) -> None:
    provider_dir = isolated_provider_expansion_layout["provider_dir"]
    provider_dir.mkdir(parents=True, exist_ok=True)
    (provider_dir / "signal_event_rate.json").write_text(
        json.dumps(
            {
                "asset_id": "family.signal_event_rate.expansions.provider",
                "asset_version": "provider.v1",
                "family": "signal_event_rate",
                "family_aliases": ["signal_detect_measure"],
                "domain": "signal_processing",
                "name": "Signal Event Rate Expansion Inventory",
                "summary": "Provider asset with invalid primitive reference.",
                "operations": [
                    {
                        "rule_name": "insert_something",
                        "name": "Insert Something",
                        "dejargonized_summary": "Fixture operation.",
                        "trigger": {
                            "metric_name": "quality",
                            "comparison": "gt",
                            "threshold": 1.0,
                            "required_boundary_requirements": [
                                {
                                    "boundary_kind": "root_input",
                                    "port_name": "signal",
                                    "matched_primitives": ["does_not_exist"],
                                }
                            ],
                        },
                    }
                ],
                "audit": {
                    "source_kind": "shared_asset",
                    "review_status": "transitional",
                    "dejargonized_summary": "Fixture expansion asset.",
                    "references": [{"title": "Fixture Reference"}],
                },
            }
        )
    )

    with pytest.raises(ValueError, match="unknown registered atoms: does_not_exist"):
        load_local_expansion_assets()


def test_provider_expansion_asset_accepts_registered_matched_primitive(
    isolated_provider_expansion_layout: dict[str, Path],
) -> None:
    _write_registered_atom(
        isolated_provider_expansion_layout["provider_dir"].parents[1],
        module_name="fixture_atoms",
        atom_name="normalize_records",
    )
    provider_dir = isolated_provider_expansion_layout["provider_dir"]
    provider_dir.mkdir(parents=True, exist_ok=True)
    (provider_dir / "generic_records.json").write_text(
        json.dumps(
            {
                "asset_id": "family.generic_records.expansions.provider",
                "asset_version": "provider.v1",
                "family": "generic_records",
                "domain": "tabular_processing",
                "name": "Generic Records Expansion Inventory",
                "summary": "Provider asset with valid primitive reference.",
                "operations": [
                    {
                        "rule_name": "insert_normalization_gate",
                        "name": "Insert Normalization Gate",
                        "dejargonized_summary": "Fixture operation.",
                        "trigger": {
                            "metric_name": "quality",
                            "comparison": "gt",
                            "threshold": 1.0,
                            "required_boundary_requirements": [
                                {
                                    "boundary_kind": "root_input",
                                    "port_name": "records",
                                    "matched_primitives": ["normalize_records"],
                                }
                            ],
                        },
                    }
                ],
                "audit": {
                    "source_kind": "shared_asset",
                    "review_status": "transitional",
                    "dejargonized_summary": "Fixture expansion asset.",
                    "references": [{"title": "Fixture Reference"}],
                },
            }
        )
    )
    clear_registered_atom_identifier_cache()
    clear_local_expansion_asset_caches()

    by_family = load_local_expansion_assets_by_family()

    assert "generic_records" in by_family
