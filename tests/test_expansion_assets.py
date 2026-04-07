"""Tests for local auditable expansion-family assets."""

from __future__ import annotations

import numpy as np

from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.principal.expansion import (
    ExpansionContext,
    ExpansionDiagnostic,
    ExpansionEngine,
)
from sciona.principal.expansion_assets import (
    AssetBackedExpansionRuleSet,
    ExpansionFamilyAsset,
    asset_backed_rule_sets,
    load_local_expansion_assets_by_family,
)
from sciona.principal.expansion_rules import default_rule_sets
from sciona.principal.expansion_rules.signal_event_rate import (
    _build_insert_jump_removal_before_filter,
)


def _signal_rate_cdg() -> CDGExport:
    source = AlgorithmicNode(
        node_id="src",
        name="Source",
        description="signal source",
        concept_type=ConceptType.CUSTOM,
        status=NodeStatus.ATOMIC,
        outputs=[IOSpec(name="signal", type_desc="np.ndarray")],
    )
    filt = AlgorithmicNode(
        node_id="filt",
        name="Filter",
        description="filter signal",
        concept_type=ConceptType.SIGNAL_FILTER,
        status=NodeStatus.ATOMIC,
        matched_primitive="filter_signal_for_detection",
        inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
        outputs=[IOSpec(name="signal", type_desc="np.ndarray")],
    )
    return CDGExport(
        nodes=[source, filt],
        edges=[
            DependencyEdge(
                source_id="src",
                target_id="filt",
                output_name="signal",
                input_name="signal",
                source_type="np.ndarray",
                target_type="np.ndarray",
            )
        ],
    )


def _root_boundary_signal_rate_cdg() -> CDGExport:
    root = AlgorithmicNode(
        node_id="root",
        name="Root",
        description="top level signal pipeline",
        concept_type=ConceptType.ANALYSIS,
        status=NodeStatus.DECOMPOSED,
        children=["filt", "det"],
        outputs=[IOSpec(name="rate", type_desc="np.ndarray")],
        inputs=[
            IOSpec(name="signal", type_desc="np.ndarray"),
            IOSpec(name="sampling_rate", type_desc="float"),
        ],
    )
    filt = AlgorithmicNode(
        node_id="filt",
        parent_id="root",
        name="Filter",
        description="filter signal",
        concept_type=ConceptType.SIGNAL_FILTER,
        status=NodeStatus.ATOMIC,
        matched_primitive="filter_signal_for_detection",
        inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
        outputs=[IOSpec(name="signal", type_desc="np.ndarray")],
    )
    detect = AlgorithmicNode(
        node_id="det",
        parent_id="root",
        name="Detect",
        description="detect events",
        concept_type=ConceptType.DATA_EXTRACTION,
        status=NodeStatus.ATOMIC,
        matched_primitive="detect_peaks_in_signal",
        inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
        outputs=[IOSpec(name="events", type_desc="np.ndarray")],
    )
    return CDGExport(
        nodes=[root, filt, detect],
        edges=[
            DependencyEdge(
                source_id="filt",
                target_id="det",
                output_name="signal",
                input_name="signal",
                source_type="np.ndarray",
                target_type="np.ndarray",
            )
        ],
    )


class TestExpansionAssets:
    def test_loads_local_expansion_assets(self):
        by_family = load_local_expansion_assets_by_family()
        asset = by_family["signal_event_rate"]

        assert asset.asset_id == "family.signal_event_rate.expansions.v1"
        assert asset.audit.review_status == "transitional"
        assert len(asset.operations) == 4

    def test_asset_backed_rule_set_attaches_provenance(self):
        class _StubRuleSet:
            name = "signal_event_rate"
            domain = "signal_processing"

            def diagnose(self, cdg, context):
                return [
                    ExpansionDiagnostic(
                        rule_name="insert_jump_removal_before_filter",
                        severity=0.9,
                        evidence="synthetic discontinuity evidence",
                        metric_name="jump_discontinuity_count",
                        metric_value=7.0,
                        threshold=3.0,
                        source_domain="signal_processing",
                    )
                ]

            def rules(self):
                return [_build_insert_jump_removal_before_filter()]

        rule_set = asset_backed_rule_sets([_StubRuleSet()])[0]
        context = ExpansionContext(
            signal_data={
                "signal": np.zeros(32, dtype=float),
            },
            planning_artifact={
                "planning_constraints": [
                    {"category": "loss"},
                    {"category": "provenance"},
                ]
            },
        )

        diagnostics = rule_set.diagnose(_signal_rate_cdg(), context)

        assert diagnostics
        assert diagnostics[0].asset_id == "family.signal_event_rate.expansions.v1"
        assert diagnostics[0].asset_operation == "insert_jump_removal_before_filter"
        assert diagnostics[0].asset_source_kind == "local_asset"

    def test_engine_reports_applied_asset_summary(self):
        class _StubRuleSet:
            name = "signal_event_rate"
            domain = "signal_processing"

            def diagnose(self, cdg, context):
                return [
                    ExpansionDiagnostic(
                        rule_name="insert_jump_removal_before_filter",
                        severity=0.9,
                        evidence="synthetic discontinuity evidence",
                        metric_name="jump_discontinuity_count",
                        metric_value=7.0,
                        threshold=3.0,
                        source_domain="signal_processing",
                    )
                ]

            def rules(self):
                return [_build_insert_jump_removal_before_filter()]

        engine = ExpansionEngine(asset_backed_rule_sets([_StubRuleSet()]))
        context = ExpansionContext(
            signal_data={
                "signal": np.zeros(32, dtype=float),
            },
            planning_artifact={
                "planning_constraints": [
                    {"category": "loss"},
                    {"category": "provenance"},
                ]
            },
        )

        result = engine.expand(_signal_rate_cdg(), context)

        assert result.expanded is True
        assert result.applied_assets[0]["asset_id"] == "family.signal_event_rate.expansions.v1"
        assert result.applied_assets[0]["asset_operation"] == "insert_jump_removal_before_filter"

    def test_default_rule_sets_expose_asset_backed_provenance(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(5000)
        for idx in range(500, signal.size, 500):
            signal[idx:] += 25.0

        rule_set = next(rs for rs in default_rule_sets() if rs.name == "signal_event_rate")
        diagnostics = rule_set.diagnose(
            _signal_rate_cdg(),
            ExpansionContext(
                signal_data={"signal": signal},
                planning_artifact={"planning_constraints": [{"category": "loss"}]},
            ),
        )

        assert diagnostics
        assert diagnostics[0].asset_id == "family.signal_event_rate.expansions.v1"
        assert diagnostics[0].asset_operation == "insert_jump_removal_before_filter"

    def test_asset_wrapper_can_require_root_boundaries_and_adjacencies(self):
        class _StubRuleSet:
            name = "synthetic_boundary"
            domain = "signal_processing"

            def diagnose(self, cdg, context):
                return [
                    ExpansionDiagnostic(
                        rule_name="boundary_rule",
                        severity=0.9,
                        evidence="synthetic boundary evidence",
                        metric_name="boundary_metric",
                        metric_value=1.0,
                        threshold=0.1,
                    )
                ]

            def rules(self):
                return []

        asset = ExpansionFamilyAsset.model_validate(
            {
                "asset_id": "family.synthetic_boundary.expansions.v1",
                "asset_version": "phase4.v1",
                "family": "synthetic_boundary",
                "domain": "signal_processing",
                "name": "Synthetic Boundary Expansion Inventory",
                "summary": "Synthetic asset for boundary-aware applicability tests.",
                "operations": [
                    {
                        "rule_name": "boundary_rule",
                        "name": "Boundary Rule",
                        "intent": "Require semantic boundary applicability.",
                        "dejargonized_summary": "Only apply when a root signal boundary feeds a filter that feeds a detector.",
                        "trigger": {
                            "metric_name": "boundary_metric",
                            "comparison": "gt",
                            "threshold": 0.1,
                            "required_root_inputs": ["signal"],
                            "required_primitives": [
                                "filter_signal_for_detection",
                                "detect_peaks_in_signal",
                            ],
                            "required_adjacencies": [
                                ["filter_signal_for_detection", "detect_peaks_in_signal"]
                            ],
                        },
                    }
                ],
                "audit": {
                    "review_status": "transitional",
                    "source_kind": "local_asset",
                    "dejargonized_summary": "Synthetic boundary asset.",
                    "references": [{"title": "Synthetic boundary reference"}],
                },
            }
        )
        rule_set = AssetBackedExpansionRuleSet(_StubRuleSet(), asset)

        diagnostics = rule_set.diagnose(
            _root_boundary_signal_rate_cdg(),
            ExpansionContext(
                signal_data={"signal": np.zeros(32, dtype=float)},
                planning_artifact={"planning_constraints": [{"category": "loss"}]},
            ),
        )

        assert diagnostics
        assert diagnostics[0].asset_id == "family.synthetic_boundary.expansions.v1"
        assert diagnostics[0].asset_operation == "boundary_rule"
