"""Phase 6 integration tests for canonical runtime evidence contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.architect.planning_contract import build_planning_artifact
from sciona.principal.admissibility import (
    AdmissibilityEvaluator,
    MinimumCountPerDurationRule,
    RequiredRuntimeKeysRule,
    build_admissibility_context,
)
from sciona.principal.evaluator import _build_runtime_artifacts


def _multi_stream_cdg() -> CDGExport:
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
        inputs=[IOSpec(name="events", type_desc="np.ndarray")],
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


def test_build_admissibility_context_supports_multistream_rules_and_summaries() -> None:
    planning_artifact = build_planning_artifact(
        goal="Estimate event rate from a signal",
        thread_id="thread-1",
        paradigm="signal_event_rate",
        family_hint="signal_event_rate",
        root_inputs=[
            IOSpec(name="signal", type_desc="np.ndarray"),
            IOSpec(name="sampling_rate", type_desc="float"),
        ],
        root_outputs=[IOSpec(name="rate", type_desc="np.ndarray")],
    ).model_dump(mode="json")

    signal = np.sin(np.linspace(0.0, 100.0, 12996))
    context = build_admissibility_context(
        cdg=_multi_stream_cdg(),
        planning_artifact=planning_artifact,
        runtime_artifacts={
            "signal_data": {
                "capnostream_value": np.linspace(0.0, 1.0, 30),
                "capnostream_sampling_rate": 21.0,
                "h10_ecg_value": signal,
                "h10_ecg_sampling_rate": 129.96,
                "h10_ecg_t": np.linspace(0.0, 100.0, 12996),
            },
            "intermediates": {"events": np.array([12.0, 140.0, 280.0])},
        },
        family="signal_event_rate",
    )

    evaluator = AdmissibilityEvaluator(
        [
            RequiredRuntimeKeysRule(["sampling_rate"]),
            MinimumCountPerDurationRule(
                count_metric="events.count",
                duration_metric="events.duration_seconds",
                min_per_minute=20.0,
            ),
        ]
    )
    report = evaluator.evaluate(context)

    assert context.semantic_cdg is not None
    assert context.runtime_context["stream_count"] == 2
    assert context.runtime_context["canonical_inputs"]["signal"] == "h10_ecg_value"
    assert context.runtime_context["canonical_inputs"]["sampling_rate"] == "h10_ecg_sampling_rate"
    assert context.runtime_context["sampling_rate"] == pytest.approx(129.96)
    assert context.runtime_context["signal"] == "h10_ecg_value"
    assert context.telemetry["signal"]["count"] == 12996.0
    assert context.telemetry["signal"]["max_abs"] <= 1.0
    assert context.telemetry["events"]["count"] == 3.0
    assert context.telemetry["events"]["duration_seconds"] == pytest.approx(100.0)
    assert context.metric("events.count") == 3.0
    assert context.metric("events.duration_seconds") == pytest.approx(100.0)
    assert report.hard_rejected is True
    assert report.decisions[0].rule_id == "minimum_count_per_duration"


def test_runtime_artifacts_emit_phase6_canonical_evidence_contract(tmp_path) -> None:
    artifacts = _build_runtime_artifacts(
        trace_path=tmp_path / "trace.jsonl",
        stdout_payload={
            "intermediates": {"events": np.array([12.0, 140.0, 280.0])},
            "outputs": {"rate": np.array([70.0, 71.0, 69.5])},
        },
        signal_data={
            "capnostream_value": np.linspace(0.0, 1.0, 30),
            "capnostream_sampling_rate": 21.0,
            "h10_ecg_value": np.sin(np.linspace(0.0, 100.0, 12996)),
            "h10_ecg_sampling_rate": 129.96,
            "h10_ecg_t": np.linspace(0.0, 100.0, 12996),
        },
    )

    assert artifacts["runtime_context"]["canonical_inputs"]["signal"] == "h10_ecg_value"
    assert (
        artifacts["canonical_runtime_context"]["canonical_inputs"]["sampling_rate"]["raw_key"]
        == "h10_ecg_sampling_rate"
    )
    assert artifacts["telemetry_summary"]["streams"]["ecg"]["signal"]["duration_seconds"] == pytest.approx(100.0)
    assert artifacts["telemetry_summary"]["events"]["count"] == 3.0
    assert artifacts["telemetry_summary"]["outputs"]["rate"]["mean"] > 0.0


def test_build_admissibility_context_prefers_phase6_artifact_contract() -> None:
    planning_artifact = build_planning_artifact(
        goal="Estimate event rate from a signal",
        thread_id="thread-1",
        paradigm="signal_event_rate",
        family_hint="signal_event_rate",
        root_inputs=[
            IOSpec(name="signal", type_desc="np.ndarray"),
            IOSpec(name="sampling_rate", type_desc="float"),
        ],
        root_outputs=[IOSpec(name="rate", type_desc="np.ndarray")],
    ).model_dump(mode="json")

    runtime_artifacts = _build_runtime_artifacts(
        trace_path=Path("trace.jsonl"),
        stdout_payload={
            "intermediates": {"events": np.array([12.0, 140.0, 280.0])},
            "outputs": {"rate": np.array([70.0, 71.0, 69.5])},
        },
        signal_data={
            "h10_ecg_value": np.sin(np.linspace(0.0, 100.0, 12996)),
            "h10_ecg_sampling_rate": 129.96,
            "h10_ecg_t": np.linspace(0.0, 100.0, 12996),
        },
    )

    context = build_admissibility_context(
        cdg=_multi_stream_cdg(),
        planning_artifact=planning_artifact,
        runtime_artifacts=runtime_artifacts,
        family="signal_event_rate",
    )

    assert context.runtime_context["sampling_rate"] == pytest.approx(129.96)
    assert context.runtime_context["signal"] == "h10_ecg_value"
    assert context.telemetry["events"]["duration_seconds"] == pytest.approx(100.0)
