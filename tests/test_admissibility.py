"""Focused tests for the deterministic admissibility layer."""

from __future__ import annotations

from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.architect.semantic_graph import (
    SemanticLossClass,
    project_semantic_cdg,
)
from sciona.principal.admissibility import (
    AdmissibilityContext,
    AdmissibilityDisposition,
    AdmissibilityEvaluator,
    MinimumCountPerDurationRule,
    RequiredRuntimeKeysRule,
    RootBoundaryLossRule,
    ThresholdMetricRule,
)


def _signal_rate_cdg() -> CDGExport:
    root = AlgorithmicNode(
        node_id="root",
        parent_id=None,
        name="Root",
        description="Root waveform-to-rate pipeline",
        concept_type=ConceptType.ANALYSIS,
        status=NodeStatus.DECOMPOSED,
        children=["filt", "rate"],
        inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
        outputs=[IOSpec(name="rate", type_desc="np.ndarray")],
    )
    filt = AlgorithmicNode(
        node_id="filt",
        parent_id="root",
        name="Filter",
        description="Filter signal",
        concept_type=ConceptType.SIGNAL_FILTER,
        status=NodeStatus.ATOMIC,
        matched_primitive="filter_signal_for_detection",
        inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
        outputs=[IOSpec(name="signal", type_desc="np.ndarray")],
    )
    rate = AlgorithmicNode(
        node_id="rate",
        parent_id="root",
        name="Rate",
        description="Compute rate",
        concept_type=ConceptType.ANALYSIS,
        status=NodeStatus.ATOMIC,
        matched_primitive="compute_event_rate",
        inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
        outputs=[IOSpec(name="rate", type_desc="np.ndarray")],
    )
    return CDGExport(
        nodes=[root, filt, rate],
        edges=[
            DependencyEdge(
                source_id="filt",
                target_id="rate",
                output_name="signal",
                input_name="signal",
                source_type="np.ndarray",
                target_type="np.ndarray",
            )
        ],
    )


def test_required_runtime_keys_rule_hard_rejects_missing_context() -> None:
    evaluator = AdmissibilityEvaluator(
        [RequiredRuntimeKeysRule(["sampling_rate", "signal"])]
    )

    report = evaluator.evaluate(
        AdmissibilityContext(
            runtime_context={"signal": [1.0, 2.0, 3.0]},
            family="signal_event_rate",
        )
    )

    assert report.hard_rejected is True
    assert report.decisions[0].disposition == AdmissibilityDisposition.HARD_REJECT
    assert "sampling_rate" in report.decisions[0].evidence


def test_minimum_count_per_duration_rule_rejects_catastrophic_sparsity() -> None:
    evaluator = AdmissibilityEvaluator(
        [
            MinimumCountPerDurationRule(
                count_metric="events.count",
                duration_metric="events.duration_seconds",
                min_per_minute=20.0,
            )
        ]
    )

    report = evaluator.evaluate(
        AdmissibilityContext(
            telemetry={"events": {"count": 3, "duration_seconds": 600.0}},
            family="signal_event_rate",
        )
    )

    assert report.hard_rejected is True
    assert report.decisions[0].metric_name == "events.count"
    assert report.decisions[0].observed_value == 0.3


def test_threshold_metric_rule_can_route_to_refinement() -> None:
    evaluator = AdmissibilityEvaluator(
        [
            ThresholdMetricRule(
                rule_id="interval_outlier_fraction",
                metric_name="events.outlier_fraction",
                threshold=0.15,
                disposition=AdmissibilityDisposition.ROUTE_TO_REFINEMENT,
                summary="Detected intervals are unstable enough to require cleanup.",
                suggested_refinement="insert_outlier_rejection_after_detection",
            )
        ]
    )

    report = evaluator.evaluate(
        AdmissibilityContext(
            telemetry={"events": {"outlier_fraction": 0.32}},
            family="signal_event_rate",
        )
    )

    assert report.routed_to_refinement is True
    assert report.decisions[0].suggested_refinement == (
        "insert_outlier_rejection_after_detection"
    )


def test_root_boundary_loss_rule_rejects_early_loss() -> None:
    semantic = project_semantic_cdg(_signal_rate_cdg())
    lossy_edges = []
    for edge in semantic.edges:
        if edge.source_id.startswith("boundary:in:"):
            lossy_edges.append(
                edge.model_copy(update={"loss_class": SemanticLossClass.LOSSY})
            )
        else:
            lossy_edges.append(edge)
    semantic = semantic.model_copy(update={"edges": lossy_edges})

    evaluator = AdmissibilityEvaluator([RootBoundaryLossRule()])
    report = evaluator.evaluate(
        AdmissibilityContext(semantic_cdg=semantic, family="signal_event_rate")
    )

    assert report.hard_rejected is True
    assert report.decisions[0].rule_id == "root_boundary_loss"


def test_admissibility_report_summary_is_structured_and_deterministic() -> None:
    evaluator = AdmissibilityEvaluator(
        [
            ThresholdMetricRule(
                rule_id="warn_density",
                metric_name="events.density",
                threshold=1.0,
                disposition=AdmissibilityDisposition.SOFT_WARN,
                summary="Event density is suspicious.",
            ),
            RequiredRuntimeKeysRule(["sampling_rate"], rule_id="needs_sampling_rate"),
        ]
    )

    report = evaluator.evaluate(
        AdmissibilityContext(telemetry={"events": {"density": 2.0}}, family="signal_event_rate")
    )
    summary = report.summary()

    assert report.hard_rejected is True
    assert summary["decision_count"] == 2
    assert summary["hard_reject_rule_ids"] == ["needs_sampling_rate"]
    assert summary["warning_rule_ids"] == ["warn_density"]
