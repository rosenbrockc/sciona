from __future__ import annotations

import json
from pathlib import Path

import pytest

from sciona.principal.expansion_assets import (
    clear_local_expansion_asset_caches,
    load_local_expansion_assets_by_family,
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
    clear_local_expansion_asset_caches()
    try:
        yield {"local_dir": local_dir, "provider_dir": provider_dir}
    finally:
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
