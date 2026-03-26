"""Tests for the curated ingest regression harness."""

from __future__ import annotations

import json
import textwrap
from unittest.mock import AsyncMock

import pytest

from sciona.ingester.graph import IngesterAgent
from sciona.ingester.monitor import STATUS_FILE, IngestMonitor
from sciona.ingester.regression_harness import (
    IngestRegressionCase,
    IngestRegressionResult,
    SemanticExpectation,
    default_ingest_regression_cases,
    run_ingest_regression_suite,
    summarize_ingest_regression_results,
    summarize_monitor_trace,
)

_ROLLING_SOURCE = textwrap.dedent("""\
    class RollingAverager:
        def __init__(self, window_size: int = 5):
            self.window_size = window_size
            self.buffer: list = []
            self.count: int = 0
            self.result: float = 0.0

        def add_sample(self, value: float) -> None:
            self.buffer.append(value)
            if len(self.buffer) > self.window_size:
                self.buffer = self.buffer[-self.window_size:]
            self.count += 1

        def compute_average(self) -> float:
            if not self.buffer:
                self.result = 0.0
            else:
                self.result = sum(self.buffer) / len(self.buffer)
            return self.result
""")

_PROCEDURAL_SOURCE = textwrap.dedent("""\
    def remove_baseline(signal):
        baseline = sum(signal) / len(signal)
        return [value - baseline for value in signal]

    def compute_snr(signal):
        return max(signal) - min(signal)

    raw = [1.0, 2.0, 3.0, 4.0]
    clean = remove_baseline(raw)
    snr = compute_snr(clean)
""")

_CHUNK_RESPONSE = json.dumps(
    {
        "macro_atoms": [
            {
                "name": "Sample Accumulator",
                "description": "Accumulate sample values into a rolling buffer",
                "method_names": ["add_sample"],
                "inputs": [
                    {
                        "name": "value",
                        "type_desc": "float",
                        "constraints": "numeric sample value",
                    },
                ],
                "outputs": [],
                "config_params": ["window_size"],
                "concept_type": "custom",
                "is_optional": False,
            },
            {
                "name": "Average Computer",
                "description": "Compute the rolling average from the buffer",
                "method_names": ["compute_average"],
                "inputs": [],
                "outputs": [
                    {
                        "name": "result",
                        "type_desc": "float",
                        "constraints": "rolling average value",
                    },
                ],
                "config_params": [],
                "concept_type": "custom",
                "is_optional": False,
            },
        ],
        "edges": [
            {
                "source_id": "sample_accumulator",
                "target_id": "average_computer",
                "output_name": "buffer",
                "input_name": "buffer",
                "source_type": "list",
                "target_type": "list",
            },
        ],
    }
)

_HOIST_RESPONSE = json.dumps(
    {
        "state_models": [
            {
                "model_name": "RollingAveragerState",
                "fields": [["buffer", "list"], ["count", "int"]],
                "source_attrs": ["buffer", "count"],
                "docstring": "Cross-window state for the rolling averager.",
            },
        ],
    }
)

_ABSTRACT_RESPONSE = json.dumps(
    {
        "abstract_name": "Temporal Aggregator",
        "conceptual_transform": "Aggregates a time-local buffer into a summary.",
        "abstract_inputs": ["numeric sample"],
        "abstract_outputs": ["rolling summary"],
        "algorithmic_properties": ["stateful"],
        "cross_disciplinary_applications": ["signal processing"],
    }
)


def test_default_case_matrix_covers_required_families():
    cases = default_ingest_regression_cases()

    assert len(cases) == 6
    assert {case.case_id for case in cases} == {
        "sklearn_style_estimator",
        "flat_scientific_function",
        "rolling_stateful_class",
        "bayesian_or_message_passing",
        "non_python_ffi",
        "procedural_ingest",
    }


def test_summarize_monitor_trace_counts_prompt_keys_and_stalled(tmp_path):
    mon = IngestMonitor(tmp_path, enable_trace=True)
    mon.start(
        source_path="src/example.py",
        class_name="Example",
        procedural=False,
        llm_provider="tests",
        llm_model="fixture",
        max_depth=1,
    )
    mon.llm_start("ingester_chunk")
    mon.llm_end("ingester_chunk", ok=True)
    mon.llm_start("ingester_hoist_state")
    mon.llm_end("ingester_hoist_state", ok=True)

    stalled_status = {
        "state": "running",
        "phase": "phase2_chunk",
        "last_heartbeat_at": 1.0,
        "llm_call_inflight": None,
    }
    (tmp_path / STATUS_FILE).write_text(json.dumps(stalled_status))

    summary = summarize_monitor_trace(tmp_path, stale_seconds=1)

    assert summary.llm_call_count == 2
    assert summary.llm_prompt_counts == {
        "ingester_chunk": 1,
        "ingester_hoist_state": 1,
    }
    assert summary.timed_out_or_stalled is True
    assert summary.classified_state == "stalled"


def test_summary_aggregation_computes_rates():
    results = [
        IngestRegressionResult(
            case_id="case_a",
            family="stateful",
            completed=True,
            mypy_passed=True,
            ghost_passed=False,
            llm_call_count=2,
            semantic_checks=[],
        ),
        IngestRegressionResult(
            case_id="case_b",
            family="procedural",
            completed=False,
            timed_out_or_stalled=True,
            mypy_passed=False,
            ghost_passed=False,
            llm_call_count=0,
            error="boom",
            semantic_checks=[],
        ),
    ]

    summary = summarize_ingest_regression_results(results)

    assert summary.total_cases == 2
    assert summary.completed_cases == 1
    assert summary.completion_rate == 0.5
    assert summary.mypy_pass_rate == 0.5
    assert summary.timeout_or_stall_count == 1
    assert summary.llm_call_total == 2
    assert summary.failures == ["case_b"]
    assert summary.family_breakdown["stateful"].completed_cases == 1


@pytest.mark.asyncio
async def test_run_curated_suite_over_stateful_and_procedural_cases(tmp_path):
    cases = [
        IngestRegressionCase(
            case_id="rolling_stateful",
            family="rolling_stateful",
            class_name="RollingAverager",
            inline_source=_ROLLING_SOURCE,
            semantic_expectations=[
                SemanticExpectation(check="has_canonical_ir"),
                SemanticExpectation(
                    check="generated_state_models_contains",
                    value="RollingAveragerState",
                ),
            ],
        ),
        IngestRegressionCase(
            case_id="procedural_pipeline",
            family="procedural_ingest",
            class_name="PulsarFold",
            procedural=True,
            inline_source=_PROCEDURAL_SOURCE,
            semantic_expectations=[
                SemanticExpectation(check="cdg_node_count_at_least", minimum=3),
            ],
        ),
    ]

    def build_agent(case_dir, monitor, case):
        if case.procedural:
            return IngesterAgent(llm=AsyncMock(), monitor=monitor)
        llm = AsyncMock()
        llm.complete.side_effect = [
            _CHUNK_RESPONSE,
            _HOIST_RESPONSE,
            _ABSTRACT_RESPONSE,
            _ABSTRACT_RESPONSE,
        ]
        return IngesterAgent(llm=llm, monitor=monitor)

    results, summary = await run_ingest_regression_suite(
        cases,
        output_root=tmp_path,
        agent_factory=build_agent,
        stale_seconds=30,
    )

    assert len(results) == 2
    assert summary.total_cases == 2
    assert summary.completed_cases == 2
    assert summary.family_breakdown["rolling_stateful"].completed_cases == 1
    assert summary.family_breakdown["procedural_ingest"].completed_cases == 1

    rolling = next(item for item in results if item.case_id == "rolling_stateful")
    assert rolling.completed is True
    assert rolling.llm_call_count == 4
    assert rolling.llm_prompt_counts["ingester_chunk"] == 1
    assert rolling.llm_prompt_counts["ingester_hoist_state"] == 1
    assert rolling.llm_prompt_counts["ingester_abstract"] == 2
    assert rolling.has_canonical_ir is True
    assert "state_models.py" in rolling.published_artifacts
    assert all(check.passed for check in rolling.semantic_checks)

    procedural = next(item for item in results if item.case_id == "procedural_pipeline")
    assert procedural.completed is True
    assert procedural.llm_call_count == 0
    assert "atoms.py" in procedural.published_artifacts
    assert all(check.passed for check in procedural.semantic_checks)
