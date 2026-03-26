"""Tests for the curated ingest regression harness."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sciona.ingester.graph import IngesterAgent
from sciona.ingester.models import IngestionBundle, ValidatedMacroPlan
from sciona.ingester.monitor import STATUS_FILE, IngestMonitor
from sciona.ingester.regression_harness import (
    IngestRegressionCase,
    IngestRegressionResult,
    NormalizedArtifactBundle,
    SemanticExpectation,
    compare_case_artifacts_to_goldens,
    default_ingest_regression_cases,
    normalize_snapshot_payload,
    run_ingest_regression_suite,
    summarize_ingest_regression_results,
    summarize_monitor_trace,
)

_FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "ingest_regression"
_GOLDEN_ROOT = Path(__file__).resolve().parent / "golden"

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


def _golden_case_dir(case_id: str) -> Path:
    return _GOLDEN_ROOT / "ingest_regression" / case_id


def _read_golden_json(case_id: str, name: str) -> dict:
    return json.loads((_golden_case_dir(case_id) / name).read_text())


def _read_golden_text(case_id: str, name: str) -> str:
    return (_golden_case_dir(case_id) / name).read_text()


def _bundle_from_golden(case_id: str, case_dir: Path) -> IngestionBundle:
    cdg = _read_golden_json(case_id, "cdg.json")
    cdg.setdefault("metadata", {})
    cdg["metadata"]["timestamp"] = 1712345678.0
    payload = {
        "cdg": cdg,
        "generated_atoms": _read_golden_text(case_id, "atoms.py"),
        "generated_state_models": (
            _read_golden_text(case_id, "state_models.py")
            if (_golden_case_dir(case_id) / "state_models.py").exists()
            else ""
        ),
        "generated_witnesses": _read_golden_text(case_id, "witnesses.py"),
        "mypy_passed": True,
        "ghost_sim_passed": True,
    }
    return IngestionBundle.model_validate(payload)


def _validated_plan_from_golden(case_id: str, case_dir: Path) -> ValidatedMacroPlan:
    canonical_ir = _read_golden_json(case_id, "canonical_ir.json")
    canonical_ir["started_at"] = 1.0
    canonical_ir["operations"] = list(reversed(canonical_ir.get("operations", [])))

    planning_graph = _read_golden_json(case_id, "planning_graph.json")
    planning_graph["timestamp"] = 2.0
    planning_graph["planned_groups"] = list(
        reversed(planning_graph.get("planned_groups", []))
    )

    return ValidatedMacroPlan.model_validate(
        {
            "plan": {
                "canonical_ir": canonical_ir,
                "planning_graph": planning_graph,
            },
            "all_attrs_accounted": True,
            "coverage_report": "fixture",
            "ir_validated": True,
        }
    )


class _GoldenFixtureAgent:
    def __init__(self, case_dir: Path, case: IngestRegressionCase):
        self.case_dir = case_dir
        self.case = case
        self._deps = SimpleNamespace(proof_env=None)

    async def ingest_state(self, source_path: str, class_name: str) -> dict:
        return {
            "bundle": _bundle_from_golden(self.case.case_id, self.case_dir),
            "validated_plan": _validated_plan_from_golden(self.case.case_id, self.case_dir),
            "error": "",
            "type_failure_classification": {},
            "ghost_failure_classification": {},
        }

    async def ingest_procedural(self, source_path: str, class_name: str) -> IngestionBundle:
        return _bundle_from_golden(self.case.case_id, self.case_dir)


class _FailureFixtureAgent:
    def __init__(self):
        self._deps = SimpleNamespace(proof_env=None)

    async def ingest_state(self, source_path: str, class_name: str) -> dict:
        return {
            "bundle": IngestionBundle.model_validate(
                {
                    "cdg": {
                        "nodes": [],
                        "edges": [],
                        "metadata": {"timestamp": 1234.0},
                    },
                    "generated_atoms": "class BrokenAtom:\n    pass\n",
                    "generated_witnesses": "def witness_broken_atom():\n    return None\n",
                    "mypy_passed": False,
                    "ghost_sim_passed": False,
                }
            ),
            "error": "type check failed",
            "phase": "verify_types",
            "type_failure_classification": {"reason_code": "missing_annotation"},
            "ghost_failure_classification": {},
        }

    async def ingest_procedural(self, source_path: str, class_name: str) -> IngestionBundle:
        raise AssertionError("procedural path should not be used for failure case")


def test_default_case_matrix_covers_required_families():
    cases = default_ingest_regression_cases(fixture_root=_FIXTURE_ROOT)

    assert len(cases) == 6
    assert {case.case_id for case in cases} == {
        "sklearn_style_estimator",
        "rolling_stateful_class",
        "bayesian_or_message_passing",
        "dsp_biosignal_pipeline",
        "non_python_ffi",
        "procedural_ingest",
    }
    assert {case.family for case in cases} == {
        "sklearn_estimator",
        "rolling_stateful",
        "bayesian_or_message_passing",
        "dsp_biosignal",
        "non_python_ffi",
        "procedural_ingest",
    }
    assert all(Path(case.source_path).exists() for case in cases)
    assert all(case.expected_artifacts for case in cases)


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
    assert summary.cache_state == "unknown"
    assert summary.cache_state_source == ""


def test_summary_aggregation_computes_rates():
    results = [
        IngestRegressionResult(
            case_id="case_a",
            family="stateful",
            completed=True,
            mypy_passed=True,
            ghost_passed=False,
            llm_call_count=2,
            cache_state="hit",
            runtime_ms=100.0,
            semantic_checks=[],
            golden_match=True,
        ),
        IngestRegressionResult(
            case_id="case_b",
            family="procedural",
            completed=False,
            timed_out_or_stalled=True,
            mypy_passed=False,
            ghost_passed=False,
            llm_call_count=0,
            cache_state="miss",
            runtime_ms=300.0,
            error="boom",
            semantic_checks=[],
            golden_match=False,
        ),
    ]

    summary = summarize_ingest_regression_results(results)

    assert summary.total_cases == 2
    assert summary.completed_cases == 1
    assert summary.completion_rate == 0.5
    assert summary.mypy_pass_rate == 0.5
    assert summary.timeout_or_stall_count == 1
    assert summary.llm_call_total == 2
    assert summary.cache_hit_cases == 1
    assert summary.cache_miss_cases == 1
    assert summary.cache_unknown_cases == 0
    assert summary.cache_observed_cases == 2
    assert summary.cache_hit_rate == 0.5
    assert summary.runtime_ms_total == 400.0
    assert summary.runtime_ms_avg == 200.0
    assert summary.runtime_ms_p50 == 200.0
    assert summary.runtime_ms_max == 300.0
    assert summary.golden_compared_cases == 2
    assert summary.golden_matched_cases == 1
    assert summary.golden_match_rate == 0.5
    assert summary.failures == ["case_b"]
    assert summary.family_breakdown["stateful"].completed_cases == 1
    assert summary.family_breakdown["stateful"].cache_hit_cases == 1
    assert summary.family_breakdown["procedural"].cache_miss_cases == 1
    assert summary.family_breakdown["stateful"].runtime_ms_total == 100.0
    assert summary.family_breakdown["procedural"].runtime_ms_max == 300.0


def test_summarize_monitor_trace_reads_cache_state_from_completed_marker(tmp_path):
    mon = IngestMonitor(tmp_path, enable_trace=True)
    mon.start(
        source_path="src/example.py",
        class_name="Example",
        procedural=False,
        llm_provider="tests",
        llm_model="fixture",
        max_depth=1,
    )
    mon.complete(summary={"cache": {"state": "hit"}})

    summary = summarize_monitor_trace(tmp_path, stale_seconds=30)

    assert summary.classified_state == "completed"
    assert summary.marker_state == "completed"
    assert summary.cache_state == "hit"
    assert summary.cache_state_source == "surface.status.summary.cache.state"


def test_normalize_snapshot_payload_strips_path_and_transient_noise(tmp_path):
    payload = {
        "run_id": "abc",
        "started_at": 123.0,
        "metadata": {
            "timestamp": 456.0,
            "source_path": str(tmp_path / "sources" / "example.py"),
        },
        "ops": [
            {"operation_id": "b", "started_at": 99.0},
            {"operation_id": "a"},
        ],
        "values": [3, 1, 2],
    }

    normalized = normalize_snapshot_payload(payload, output_dir=tmp_path)

    assert "run_id" not in normalized
    assert "started_at" not in normalized
    assert "timestamp" not in normalized["metadata"]
    assert normalized["metadata"]["source_path"].startswith("<case_output_dir>")
    assert [item["operation_id"] for item in normalized["ops"]] == ["a", "b"]
    assert normalized["values"] == [1, 2, 3]


def test_compare_case_artifacts_to_goldens_handles_missing_optional_artifact(tmp_path):
    case = IngestRegressionCase(
        case_id="optional_failure_artifact",
        family="test",
        class_name="Dummy",
        expected_artifacts=["verification_failure"],
        optional_artifacts=["verification_failure"],
    )
    observed = NormalizedArtifactBundle(case_id=case.case_id, artifacts={})
    comparison = compare_case_artifacts_to_goldens(
        case,
        observed=observed,
        golden_root=tmp_path,
        output_dir=tmp_path / "out",
    )

    assert comparison.matched is True
    assert comparison.compared_artifacts == ["verification_failure"]
    assert comparison.mismatched_artifacts == []


def test_compare_case_artifacts_to_goldens_reports_content_mismatch(tmp_path):
    case = IngestRegressionCase(
        case_id="mismatch_case",
        family="test",
        class_name="Dummy",
        expected_artifacts=["canonical_ir", "atoms"],
    )
    golden_dir = tmp_path / "ingest_regression" / case.case_id
    golden_dir.mkdir(parents=True)
    (golden_dir / "canonical_ir.json").write_text(json.dumps({"subject_name": "X"}))
    (golden_dir / "atoms.py").write_text("class Atom:\n    pass\n")

    observed = NormalizedArtifactBundle(
        case_id=case.case_id,
        artifacts={
            "canonical_ir": json.dumps({"subject_name": "Y"}, indent=2, sort_keys=True),
            "atoms": "class Atom:\n    pass\n",
        },
    )

    comparison = compare_case_artifacts_to_goldens(
        case,
        observed=observed,
        golden_root=tmp_path,
        output_dir=tmp_path / "run",
    )

    assert comparison.matched is False
    assert comparison.mismatched_artifacts == ["canonical_ir"]
    assert comparison.mismatch_details["canonical_ir"] == "content_mismatch"


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


@pytest.mark.asyncio
async def test_run_default_real_world_corpus_with_goldens(tmp_path):
    cases = default_ingest_regression_cases(fixture_root=_FIXTURE_ROOT)

    def build_agent(case_dir, monitor, case):
        return _GoldenFixtureAgent(case_dir, case)

    results, summary = await run_ingest_regression_suite(
        cases,
        output_root=tmp_path,
        agent_factory=build_agent,
        golden_root=_GOLDEN_ROOT,
        stale_seconds=30,
    )

    assert len(results) == 6
    assert summary.total_cases == 6
    assert summary.completed_cases == 6
    assert summary.golden_compared_cases == 6
    assert summary.golden_matched_cases == 6
    assert summary.golden_match_rate == 1.0
    assert set(summary.family_breakdown.keys()) == {
        "sklearn_estimator",
        "rolling_stateful",
        "bayesian_or_message_passing",
        "dsp_biosignal",
        "non_python_ffi",
        "procedural_ingest",
    }
    assert all(result.golden_match is True for result in results)
    assert all(result.mismatched_artifacts == [] for result in results)

    rust_case = next(result for result in results if result.case_id == "non_python_ffi")
    assert rust_case.source_language == "rust"


@pytest.mark.asyncio
async def test_failure_artifact_snapshot_participates_in_golden_compare(tmp_path):
    case = IngestRegressionCase(
        case_id="verification_failure_case",
        family="failure_case",
        class_name="BrokenEstimator",
        expected_language="python",
        source_path=str((_FIXTURE_ROOT / "verification_failure_case" / "source.py").resolve()),
        expected_artifacts=["verification_failure"],
        semantic_expectations=[],
    )

    def build_agent(case_dir, monitor, built_case):
        return _FailureFixtureAgent()

    results, summary = await run_ingest_regression_suite(
        [case],
        output_root=tmp_path,
        agent_factory=build_agent,
        golden_root=_GOLDEN_ROOT,
        stale_seconds=30,
    )

    assert summary.total_cases == 1
    assert summary.completed_cases == 0
    assert summary.golden_matched_cases == 1
    assert summary.golden_match_rate == 1.0
    assert results[0].golden_match is True
    assert results[0].has_verification_failure_artifact is True
    assert results[0].mismatched_artifacts == []
