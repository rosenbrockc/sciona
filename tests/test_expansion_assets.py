"""Tests for local auditable expansion-family assets."""

from __future__ import annotations

import numpy as np

from sciona.asset_migration import MigrationReadinessAsset
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
    ExpansionTriggerAsset,
    ExpansionFamilyAsset,
    expansion_asset_summary,
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


def _generic_records_cdg() -> CDGExport:
    root = AlgorithmicNode(
        node_id="root",
        name="Records Pipeline",
        description="top level records pipeline",
        concept_type=ConceptType.DATA_ASSEMBLY,
        status=NodeStatus.DECOMPOSED,
        children=["norm"],
        inputs=[IOSpec(name="records", type_desc="list[dict]")],
        outputs=[IOSpec(name="result", type_desc="list[dict]")],
    )
    norm = AlgorithmicNode(
        node_id="norm",
        parent_id="root",
        name="Normalize Records",
        description="normalize input records",
        concept_type=ConceptType.CUSTOM,
        status=NodeStatus.ATOMIC,
        matched_primitive="normalize_records",
        depth=1,
        inputs=[IOSpec(name="records", type_desc="list[dict]")],
        outputs=[IOSpec(name="records", type_desc="list[dict]")],
    )
    return CDGExport(nodes=[root, norm], edges=[], metadata={"goal": "records"})


class TestExpansionAssets:
    def test_loads_local_expansion_assets(self):
        by_family = load_local_expansion_assets_by_family()
        asset = by_family["signal_event_rate"]
        jump = asset.operation("insert_jump_removal_before_filter")
        correction = asset.operation("insert_peak_correction_after_detection")

        assert asset.asset_id == "family.signal_event_rate.expansions.v1"
        assert asset.audit.review_status == "transitional"
        assert len(asset.operations) == 5
        assert jump is not None
        assert correction is not None
        assert jump.trigger.required_runtime_keys == ["signal"]
        assert [action.value for action in jump.action_classes] == [
            "precondition",
            "insert_correction",
        ]
        assert jump.trigger.required_boundary_requirements[0].boundary_kind == "root_input"
        assert jump.trigger.required_boundary_requirements[0].port_name == "signal"
        assert asset.audit.migration_readiness.status == "in_progress"
        assert asset.audit.migration_readiness.required_check_count() == 3
        assert asset.audit.migration_readiness.completed_required_check_count() == 2

    def test_expansion_asset_summary_includes_migration_readiness(self):
        by_family = load_local_expansion_assets_by_family()
        asset = by_family["signal_event_rate"]
        jump = asset.operation("insert_jump_removal_before_filter")

        summary = expansion_asset_summary(asset, jump)

        assert summary["migration_readiness_status"] == "in_progress"
        assert summary["migration_readiness_ready"] is False
        assert summary["migration_readiness_check_count"] == 3
        assert "runtime_independence" in summary["migration_readiness_check_ids"]

    def test_trigger_asset_accepts_legacy_signal_and_boundary_aliases(self):
        trigger = ExpansionTriggerAsset.model_validate(
            {
                "metric_name": "coverage_gap",
                "threshold": 0.5,
                "required_signal_keys": ["records.payload"],
                "required_root_inputs": ["records"],
                "required_planning_terms": ["loss", "telemetry"],
            }
        )

        assert trigger.required_runtime_keys == ["records.payload"]
        assert trigger.required_boundary_requirements[0].boundary_kind == "root_input"
        assert trigger.required_boundary_requirements[0].port_name == "records"
        assert trigger.required_planning_constraint_categories == ["loss", "telemetry"]

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

        diagnostics = rule_set.diagnose(_root_boundary_signal_rate_cdg(), context)

        assert diagnostics
        assert diagnostics[0].asset_id == "family.signal_event_rate.expansions.v1"
        assert diagnostics[0].asset_operation == "insert_jump_removal_before_filter"
        assert diagnostics[0].asset_source_kind == "local_asset"
        assert diagnostics[0].asset_migration_readiness_status == "in_progress"
        assert diagnostics[0].asset_migration_readiness_ready is False

    def test_asset_backed_rule_set_accepts_summary_only_evidence(self):
        class _StubRuleSet:
            name = "signal_event_rate"
            domain = "signal_processing"

            def diagnose(self, cdg, context):
                return [
                    ExpansionDiagnostic(
                        rule_name="insert_outlier_rejection_after_detection",
                        severity=0.9,
                        evidence="summary-only event telemetry",
                        metric_name="interval_outlier_fraction",
                        metric_value=0.22,
                        threshold=0.15,
                        source_domain="signal_processing",
                    )
                ]

            def rules(self):
                return []

        rule_set = asset_backed_rule_sets([_StubRuleSet()])[0]
        context = ExpansionContext(
            runtime_evidence={
                "telemetry_summary": {
                    "intermediates": {
                        "events": {
                            "count": 438.0,
                            "outlier_fraction": 0.22,
                        }
                    }
                }
            }
        )

        diagnostics = rule_set.diagnose(_root_boundary_signal_rate_cdg(), context)

        assert diagnostics
        assert diagnostics[0].asset_operation == "insert_outlier_rejection_after_detection"

    def test_asset_backed_rule_set_accepts_peak_correction_summary_evidence(self):
        class _StubRuleSet:
            name = "signal_event_rate"
            domain = "signal_processing"

            def diagnose(self, cdg, context):
                return [
                    ExpansionDiagnostic(
                        rule_name="insert_peak_correction_after_detection",
                        severity=0.7,
                        evidence="moderate event drift",
                        metric_name="interval_outlier_fraction",
                        metric_value=0.09,
                        threshold=0.05,
                        source_domain="signal_processing",
                    )
                ]

            def rules(self):
                return []

        rule_set = asset_backed_rule_sets([_StubRuleSet()])[0]
        context = ExpansionContext(
            runtime_evidence={
                "telemetry_summary": {
                    "signal": {"count": 38943.0},
                    "events": {"count": 438.0, "outlier_fraction": 0.09},
                },
                "canonical_runtime_context": {
                    "canonical_inputs": {
                        "signal": {"raw_key": "h10_ecg_value"},
                        "sampling_rate": {"raw_key": "ecg_sampling_rate"},
                    }
                },
            }
        )

        diagnostics = rule_set.diagnose(_root_boundary_signal_rate_cdg(), context)

        assert diagnostics
        assert diagnostics[0].asset_operation == "insert_peak_correction_after_detection"

    def test_asset_wrapper_supports_generic_runtime_keys_and_boundaries(self):
        class _StubRuleSet:
            name = "generic_records"
            domain = "tabular_processing"

            def diagnose(self, cdg, context):
                return [
                    ExpansionDiagnostic(
                        rule_name="insert_normalization_gate",
                        severity=0.9,
                        evidence="records are inconsistent",
                        metric_name="record_shape_variance",
                        metric_value=2.0,
                        threshold=1.0,
                    )
                ]

            def rules(self):
                return []

        asset = ExpansionFamilyAsset.model_validate(
            {
                "asset_id": "family.generic_records.expansions.v1",
                "asset_version": "phase2.v1",
                "family": "generic_records",
                "domain": "tabular_processing",
                "name": "Generic Records Expansion Inventory",
                "summary": "Generic asset for family-neutral runtime key and boundary checks.",
                "operations": [
                    {
                        "rule_name": "insert_normalization_gate",
                        "name": "Insert Normalization Gate",
                        "intent": "Protect downstream processing by inserting a normalization stage before the first consumer.",
                        "dejargonized_summary": "Add a normalization step when the incoming records need cleanup before the main analysis.",
                        "trigger": {
                            "metric_name": "record_shape_variance",
                            "comparison": "gt",
                            "threshold": 1.0,
                            "required_runtime_keys": ["records.payload"],
                            "required_runtime_namespaces": ["records"],
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
                    "review_status": "transitional",
                    "source_kind": "local_asset",
                    "dejargonized_summary": "Generic family-neutral records asset.",
                    "references": [{"title": "Generic records reference"}],
                },
            }
        )
        rule_set = AssetBackedExpansionRuleSet(_StubRuleSet(), asset)

        diagnostics = rule_set.diagnose(
            _generic_records_cdg(),
            ExpansionContext(
                runtime_inputs={"records.payload": [1, 2, 3]},
                planning_artifact={"planning_constraints": [{"category": "loss"}]},
            ),
        )

        assert diagnostics
        assert diagnostics[0].asset_id == "family.generic_records.expansions.v1"
        assert diagnostics[0].asset_operation == "insert_normalization_gate"

    def test_migration_readiness_asset_accepts_ready_checklist(self):
        readiness = MigrationReadinessAsset.model_validate(
            {
                "status": "migrated",
                "target_repository": "../ageo-atoms",
                "checklist": [
                    {
                        "check_id": "schema",
                        "description": "Schema stable",
                        "required": True,
                        "satisfied": True,
                    }
                ],
            }
        )

        assert readiness.is_ready_for_migration() is True

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

        result = engine.expand(_root_boundary_signal_rate_cdg(), context)

        assert result.expanded is True
        assert result.applied_assets[0]["asset_id"] == "family.signal_event_rate.expansions.v1"
        assert result.applied_assets[0]["asset_operation"] == "insert_jump_removal_before_filter"
        assert result.applied_assets[0]["asset_migration_readiness_status"] == "in_progress"
        assert result.applied_assets[0]["asset_migration_readiness_ready"] is False

    def test_default_rule_sets_expose_asset_backed_provenance(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(5000)
        for idx in range(500, signal.size, 500):
            signal[idx:] += 25.0

        rule_set = next(rs for rs in default_rule_sets() if rs.name == "signal_event_rate")
        diagnostics = rule_set.diagnose(
            _root_boundary_signal_rate_cdg(),
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
