from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from sciona.principal.expansion_assets import (
    AssetBackedExpansionRuleSet,
    ExpansionFamilyAsset,
)
from sciona.principal.expansion_manifest import (
    MANIFEST_SINKS,
    build_expansion_inventory_manifest,
    check_expansion_manifest_closure,
)


class _RuleSet:
    name = "ml_model_selection"
    domain = "machine_learning"

    def diagnose(self, cdg, context):
        return []

    def rules(self):
        return [
            SimpleNamespace(name="apply_kfold_ensemble"),
            SimpleNamespace(name="apply_stacking_ensemble"),
        ]


def _asset() -> ExpansionFamilyAsset:
    return ExpansionFamilyAsset.model_validate(
        {
            "asset_id": "family.ml_model_selection.expansions.v1",
            "asset_version": "v1",
            "family": "ml_model_selection",
            "domain": "machine_learning",
            "name": "ML Model Selection Expansions",
            "summary": "Expansion inventory for ML model selection.",
            "operations": [
                {
                    "rule_name": "apply_kfold_ensemble",
                    "operation_type": "replace",
                    "name": "Apply k-fold ensemble",
                    "dejargonized_summary": "Use k-fold ensemble predictions.",
                },
                {
                    "rule_name": "apply_stacking_ensemble",
                    "operation_type": "replace",
                    "name": "Apply stacking ensemble",
                    "dejargonized_summary": "Use a stacking meta learner.",
                    "prerequisite_operations": ["apply_kfold_ensemble"],
                },
            ],
            "audit": {
                "source_kind": "shared_asset",
                "review_status": "transitional",
                "dejargonized_summary": "ML expansion inventory.",
                "references": [{"title": "Fixture"}],
            },
        }
    )


def test_expansion_inventory_manifest_uses_identical_sink_operations() -> None:
    manifest = build_expansion_inventory_manifest([_asset()])

    expected_keys = manifest["operation_keys"]
    assert manifest["asset_count"] == 1
    assert manifest["operation_count"] == 2
    assert set(manifest["sinks"]) == set(MANIFEST_SINKS)
    for sink in MANIFEST_SINKS:
        operations = manifest["sinks"][sink]["operations"]
        assert [operation["operation_key"] for operation in operations] == expected_keys
        assert {operation["artifact_kind"] for operation in operations} == {
            "expansion_operation"
        }


def test_expansion_manifest_closure_accepts_assets_and_empty_markers(tmp_path: Path) -> None:
    asset = _asset()
    rule_set = AssetBackedExpansionRuleSet(_RuleSet(), asset)
    provider_with_assets = tmp_path / "sciona-atoms-ml"
    provider_with_assets.joinpath("data", "expansions").mkdir(parents=True)
    provider_with_assets.joinpath("data", "expansions", "ml_model_selection.json").write_text("{}")
    provider_empty = tmp_path / "sciona-atoms-geo"
    provider_empty.joinpath("data", "expansions").mkdir(parents=True)
    provider_empty.joinpath("data", "expansions", "README.md").write_text(
        "# Empty Expansion Inventory\n"
    )

    report = check_expansion_manifest_closure(
        provider_roots=[provider_with_assets, provider_empty],
        assets=[asset],
        rule_sets=[rule_set],
    )

    assert report.ok
    assert report.asset_count == 1
    assert report.operation_count == 2
    assert report.missing_provider_inventory_roots == ()
    assert report.missing_asset_backed_rule_sets == ()
    assert report.missing_runtime_rules == ()
    assert report.manifest_sink_mismatches == ()


def test_expansion_manifest_closure_reports_missing_runtime_rules(tmp_path: Path) -> None:
    provider = tmp_path / "sciona-atoms-ml"
    provider.joinpath("data", "expansions").mkdir(parents=True)
    provider.joinpath("data", "expansions", "ml_model_selection.json").write_text("{}")
    incomplete_rule_set = AssetBackedExpansionRuleSet(
        SimpleNamespace(
            name="ml_model_selection",
            domain="machine_learning",
            diagnose=lambda cdg, context: [],
            rules=lambda: [SimpleNamespace(name="apply_kfold_ensemble")],
        ),
        _asset(),
    )

    report = check_expansion_manifest_closure(
        provider_roots=[provider],
        assets=[_asset()],
        rule_sets=[incomplete_rule_set],
    )

    assert not report.ok
    assert report.missing_runtime_rules == (
        "ml_model_selection:apply_stacking_ensemble",
    )

