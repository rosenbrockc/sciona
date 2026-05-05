"""Tests for local auditable expansion-family assets (generic / machinery)."""

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
)
from sciona.principal.expansion_assets import (
    AssetBackedExpansionRuleSet,
    ExpansionTriggerAsset,
    ExpansionFamilyAsset,
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


def _synthetic_filter_detect_cdg() -> CDGExport:
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
                        "operation_type": "insert",
                        "applies_to": "generic_records.records->normalize_records",
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
                        "rewrite": {
                            "before_summary": "Records flow into normalization without a cleanup gate.",
                            "after_summary": "A normalization gate is inserted before downstream processing.",
                            "information_flow_effect": "Adds input-shape evidence before records are consumed.",
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
                "target_repository": "../sciona-atoms",
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
                        "operation_type": "insert",
                        "applies_to": "synthetic_boundary.signal->filter_signal_for_detection",
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
                        "rewrite": {
                            "before_summary": "The root signal boundary feeds the filter directly.",
                            "after_summary": "A boundary-aware validation stage is inserted before the filter.",
                            "information_flow_effect": "Keeps raw signal access explicit before detection.",
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
            _synthetic_filter_detect_cdg(),
            ExpansionContext(
                signal_data={"signal": np.zeros(32, dtype=float)},
                planning_artifact={"planning_constraints": [{"category": "loss"}]},
            ),
        )

        assert diagnostics
        assert diagnostics[0].asset_id == "family.synthetic_boundary.expansions.v1"
        assert diagnostics[0].asset_operation == "boundary_rule"

    def test_asset_wrapper_supports_any_requirements_and_contraindications(self):
        class _StubRuleSet:
            name = "synthetic_any"
            domain = "tabular_processing"

            def diagnose(self, cdg, context):
                return [
                    ExpansionDiagnostic(
                        rule_name="any_rule",
                        severity=0.9,
                        evidence="synthetic any evidence",
                        metric_name="quality",
                        metric_value=1.0,
                        threshold=0.0,
                    )
                ]

            def rules(self):
                return []

        asset = ExpansionFamilyAsset.model_validate(
            {
                "asset_id": "family.synthetic_any.expansions.v1",
                "asset_version": "phase4.v1",
                "family": "synthetic_any",
                "domain": "tabular_processing",
                "name": "Synthetic Any Expansion Inventory",
                "summary": "Synthetic asset for any-key and contraindication tests.",
                "operations": [
                    {
                        "rule_name": "any_rule",
                        "operation_type": "validation",
                        "applies_to": "synthetic_any.records",
                        "name": "Any Rule",
                        "intent": "Require one of several evidence keys and reject a known contraindication.",
                        "dejargonized_summary": "Apply when either quality signal is present unless cleanup already ran.",
                        "trigger": {
                            "metric_name": "quality",
                            "comparison": "gt",
                            "threshold": 0.0,
                            "required_any_intermediate_keys": [
                                "quality_score",
                                "quality_flag",
                            ],
                            "contraindicated_intermediate_keys": [
                                "cleanup_already_applied"
                            ],
                        },
                        "rewrite": {
                            "before_summary": "Records flow onward with only raw quality evidence.",
                            "after_summary": "A validation stage records the quality decision.",
                            "information_flow_effect": "Adds quality evidence without changing the output contract.",
                        },
                    }
                ],
                "audit": {
                    "review_status": "transitional",
                    "source_kind": "local_asset",
                    "dejargonized_summary": "Synthetic any-key asset.",
                    "references": [{"title": "Synthetic any-key reference"}],
                },
            }
        )
        rule_set = AssetBackedExpansionRuleSet(_StubRuleSet(), asset)

        assert rule_set.diagnose(
            _generic_records_cdg(),
            ExpansionContext(intermediates={"quality_flag": True}),
        )
        assert not rule_set.diagnose(
            _generic_records_cdg(),
            ExpansionContext(intermediates={"other_quality": True}),
        )
        assert not rule_set.diagnose(
            _generic_records_cdg(),
            ExpansionContext(
                intermediates={
                    "quality_flag": True,
                    "cleanup_already_applied": True,
                }
            ),
        )
