"""Lightweight regression harness for curated ingest coverage."""

from __future__ import annotations

import json
import inspect
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from sciona.ingester.graph import IngesterAgent
from sciona.ingester.models import IngestionBundle
from sciona.ingester.monitor import IngestMonitor, TRACE_FILE

AgentFactory = Callable[[Path, IngestMonitor, "IngestRegressionCase"], IngesterAgent]


class SemanticExpectation(BaseModel):
    """Deterministic semantic expectation for one curated case."""

    check: str
    field: str = ""
    value: str = ""
    minimum: int | None = None


class SemanticCheckResult(BaseModel):
    """Observed result of a deterministic semantic check."""

    check: str
    passed: bool
    detail: str = ""


class IngestRegressionCase(BaseModel):
    """One curated ingest regression case."""

    case_id: str
    family: str
    class_name: str
    procedural: bool = False
    expected_language: str = "python"
    source_path: str = ""
    inline_source: str = ""
    semantic_expectations: list[SemanticExpectation] = Field(default_factory=list)


class IngestRegressionResult(BaseModel):
    """JSON-friendly result row for one harness run."""

    case_id: str
    family: str
    completed: bool = False
    failed_phase: str = ""
    timed_out_or_stalled: bool = False
    mypy_passed: bool = False
    ghost_passed: bool = False
    type_failure_reason: str = ""
    ghost_failure_reason: str = ""
    llm_call_count: int = 0
    llm_prompt_counts: dict[str, int] = Field(default_factory=dict)
    runtime_ms: float = 0.0
    published_artifacts: list[str] = Field(default_factory=list)
    semantic_checks: list[SemanticCheckResult] = Field(default_factory=list)
    error: str = ""
    output_dir: str = ""
    source_language: str = ""
    has_canonical_ir: bool = False
    has_planning_graph: bool = False


class FamilyBreakdown(BaseModel):
    """Per-family summary row."""

    total_cases: int = 0
    completed_cases: int = 0
    mypy_passed_cases: int = 0
    ghost_passed_cases: int = 0
    llm_call_total: int = 0
    stalled_cases: int = 0


class IngestRegressionSummary(BaseModel):
    """Aggregated summary across a curated suite."""

    total_cases: int = 0
    completed_cases: int = 0
    completion_rate: float = 0.0
    mypy_pass_rate: float = 0.0
    ghost_pass_rate: float = 0.0
    timeout_or_stall_count: int = 0
    llm_call_total: int = 0
    semantic_check_pass_rate: float = 0.0
    family_breakdown: dict[str, FamilyBreakdown] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)


class MonitorTraceSummary(BaseModel):
    """Normalized monitor/trace metrics for one ingest run."""

    llm_call_count: int = 0
    llm_prompt_counts: dict[str, int] = Field(default_factory=dict)
    timed_out_or_stalled: bool = False
    classified_state: str = "missing"


_SOURCE_EXTENSIONS: dict[str, str] = {
    "python": ".py",
    "rust": ".rs",
    "cpp": ".cpp",
    "c++": ".cpp",
    "julia": ".jl",
}


def default_ingest_regression_cases() -> list[IngestRegressionCase]:
    """Return a curated matrix spanning protected ingest families."""

    return [
        IngestRegressionCase(
            case_id="sklearn_style_estimator",
            family="sklearn_estimator",
            class_name="CalibratedStyleClassifier",
            expected_language="python",
            inline_source="""
class CalibratedStyleClassifier:
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.classes_ = []
        self.scale_ = 1.0

    def fit(self, x, y):
        self.classes_ = sorted(set(y))
        self.scale_ = len(x) or 1.0
        return self

    def predict(self, x):
        if self.normalize:
            return [self.classes_[0] for _ in x]
        return [self.classes_[-1] for _ in x]

    def get_metadata_routing(self):
        return {"normalize": self.normalize}
""".strip(),
            semantic_expectations=[
                SemanticExpectation(check="has_canonical_ir"),
                SemanticExpectation(check="has_planning_graph"),
            ],
        ),
        IngestRegressionCase(
            case_id="flat_scientific_function",
            family="flat_scientific_function",
            class_name="detrend_signal",
            expected_language="python",
            inline_source="""
def detrend_signal(signal):
    baseline = sum(signal) / len(signal)
    return [value - baseline for value in signal]
""".strip(),
            semantic_expectations=[
                SemanticExpectation(check="source_language_equals", value="python"),
            ],
        ),
        IngestRegressionCase(
            case_id="rolling_stateful_class",
            family="rolling_stateful",
            class_name="RollingWindowAccumulator",
            expected_language="python",
            inline_source="""
class RollingWindowAccumulator:
    def __init__(self, window_size: int = 4):
        self.window_size = window_size
        self.buffer = []
        self.total = 0.0

    def add(self, value: float):
        self.buffer.append(value)
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size:]
        self.total = float(sum(self.buffer))

    def average(self) -> float:
        if not self.buffer:
            return 0.0
        return self.total / len(self.buffer)
""".strip(),
            semantic_expectations=[
                SemanticExpectation(check="has_canonical_ir"),
            ],
        ),
        IngestRegressionCase(
            case_id="bayesian_or_message_passing",
            family="bayesian_or_message_passing",
            class_name="PosteriorAccumulator",
            expected_language="python",
            inline_source="""
class PosteriorAccumulator:
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta

    def update(self, successes: int, failures: int):
        self.alpha += successes
        self.beta += failures
        return self

    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)
""".strip(),
            semantic_expectations=[
                SemanticExpectation(check="has_canonical_ir"),
            ],
        ),
        IngestRegressionCase(
            case_id="non_python_ffi",
            family="non_python_ffi",
            class_name="integrate_step",
            expected_language="rust",
            inline_source="""
pub fn integrate_step(position: f64, velocity: f64, dt: f64) -> f64 {
    position + velocity * dt
}
""".strip(),
            semantic_expectations=[
                SemanticExpectation(check="source_language_equals", value="rust"),
            ],
        ),
        IngestRegressionCase(
            case_id="procedural_ingest",
            family="procedural_ingest",
            class_name="ProceduralPipeline",
            procedural=True,
            expected_language="python",
            inline_source="""
def normalize(xs):
    baseline = sum(xs) / len(xs)
    return [value - baseline for value in xs]


def score(xs):
    return max(xs) - min(xs)


raw = [1.0, 2.0, 3.0, 4.0]
clean = normalize(raw)
spread = score(clean)
""".strip(),
            semantic_expectations=[
                SemanticExpectation(check="cdg_node_count_at_least", minimum=3),
            ],
        ),
    ]


def read_monitor_trace_events(output_dir: str | Path) -> list[dict[str, Any]]:
    """Read structured trace events written by :class:`IngestMonitor`."""

    path = Path(output_dir) / TRACE_FILE
    if not path.exists():
        return []

    events: list[dict[str, Any]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            events.append(decoded)
    return events


def summarize_monitor_trace(
    output_dir: str | Path, *, stale_seconds: int = 120
) -> MonitorTraceSummary:
    """Summarize trace and status artifacts for one ingest run."""

    events = read_monitor_trace_events(output_dir)
    prompt_counts: Counter[str] = Counter()
    for event in events:
        if str(event.get("event_type")) != "LLM_CALL_START":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        prompt_key = str(payload.get("prompt_key") or "").strip()
        if prompt_key:
            prompt_counts[prompt_key] += 1

    status = IngestMonitor.read_status(output_dir)
    classified = IngestMonitor.classify_state(status, stale_seconds=stale_seconds)
    return MonitorTraceSummary(
        llm_call_count=sum(prompt_counts.values()),
        llm_prompt_counts=dict(sorted(prompt_counts.items())),
        timed_out_or_stalled=classified == "stalled",
        classified_state=classified,
    )


def summarize_ingest_regression_results(
    results: list[IngestRegressionResult],
) -> IngestRegressionSummary:
    """Aggregate case results into a lightweight regression summary."""

    total_cases = len(results)
    completed_cases = sum(1 for item in results if item.completed)
    mypy_passed_cases = sum(1 for item in results if item.mypy_passed)
    ghost_passed_cases = sum(1 for item in results if item.ghost_passed)
    timeout_or_stall_count = sum(1 for item in results if item.timed_out_or_stalled)
    llm_call_total = sum(item.llm_call_count for item in results)

    semantic_total = sum(len(item.semantic_checks) for item in results)
    semantic_passed = sum(
        1
        for item in results
        for check in item.semantic_checks
        if check.passed
    )

    family_breakdown: dict[str, FamilyBreakdown] = {}
    for item in results:
        family_row = family_breakdown.setdefault(item.family, FamilyBreakdown())
        family_row.total_cases += 1
        family_row.completed_cases += int(item.completed)
        family_row.mypy_passed_cases += int(item.mypy_passed)
        family_row.ghost_passed_cases += int(item.ghost_passed)
        family_row.llm_call_total += item.llm_call_count
        family_row.stalled_cases += int(item.timed_out_or_stalled)

    failures = [item.case_id for item in results if not item.completed or item.error]
    return IngestRegressionSummary(
        total_cases=total_cases,
        completed_cases=completed_cases,
        completion_rate=(completed_cases / total_cases) if total_cases else 0.0,
        mypy_pass_rate=(mypy_passed_cases / total_cases) if total_cases else 0.0,
        ghost_pass_rate=(ghost_passed_cases / total_cases) if total_cases else 0.0,
        timeout_or_stall_count=timeout_or_stall_count,
        llm_call_total=llm_call_total,
        semantic_check_pass_rate=(
            semantic_passed / semantic_total if semantic_total else 0.0
        ),
        family_breakdown=family_breakdown,
        failures=failures,
    )


def _materialize_case_source(case: IngestRegressionCase, case_dir: Path) -> Path:
    if case.source_path:
        return Path(case.source_path)

    suffix = _SOURCE_EXTENSIONS.get(case.expected_language.lower(), ".txt")
    source_dir = case_dir / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    path = source_dir / f"{case.case_id}{suffix}"
    path.write_text(case.inline_source)
    return path


def _empty_bundle() -> IngestionBundle:
    return IngestionBundle.model_validate({"cdg": {"nodes": [], "edges": []}})


def _stage_success_artifacts(
    monitor: IngestMonitor,
    *,
    bundle: IngestionBundle,
) -> list[str]:
    if bundle.generated_atoms:
        monitor.stage_file("atoms.py", bundle.generated_atoms)
    if bundle.generated_state_models:
        monitor.stage_file("state_models.py", bundle.generated_state_models)
    if bundle.generated_witnesses:
        monitor.stage_file("witnesses.py", bundle.generated_witnesses)
    monitor.stage_json("cdg.json", bundle.cdg.model_dump())
    if bundle.match_results:
        monitor.stage_json("matches.json", [item.to_dict() for item in bundle.match_results])
    return monitor.publish_staged()


def _stage_failure_artifacts(
    monitor: IngestMonitor,
    *,
    state: dict[str, Any] | None,
) -> list[str]:
    if not state:
        return []
    bundle = state.get("bundle")
    if bundle is not None:
        if getattr(bundle, "generated_atoms", ""):
            monitor.stage_file("atoms.py", bundle.generated_atoms)
        if getattr(bundle, "generated_state_models", ""):
            monitor.stage_file("state_models.py", bundle.generated_state_models)
        if getattr(bundle, "generated_witnesses", ""):
            monitor.stage_file("witnesses.py", bundle.generated_witnesses)
        if getattr(bundle, "cdg", None) is not None:
            monitor.stage_json("cdg.json", bundle.cdg.model_dump())
        if getattr(bundle, "match_results", None):
            monitor.stage_json("matches.json", [item.to_dict() for item in bundle.match_results])
    raw_dfg = state.get("raw_dfg")
    if raw_dfg is not None:
        monitor.stage_json("raw_dfg.json", raw_dfg.model_dump(mode="json"))
    validated_plan = state.get("validated_plan")
    if validated_plan is not None:
        monitor.stage_json("validated_plan.json", validated_plan.model_dump(mode="json"))
    monitor.stage_json(
        "ingest_failure_state.json",
        {
            "error": state.get("error", ""),
            "mypy_passed": bool(state.get("mypy_passed", False)),
            "ghost_passed": bool(state.get("ghost_passed", False)),
            "mypy_errors": state.get("mypy_errors", ""),
            "ghost_errors": state.get("ghost_errors", ""),
            "type_repair_count": int(state.get("type_repair_count", 0) or 0),
            "ghost_repair_count": int(state.get("ghost_repair_count", 0) or 0),
        },
    )
    return monitor.publish_staged()


def _evaluate_semantic_expectations(
    case: IngestRegressionCase,
    *,
    bundle: IngestionBundle,
    final_state: dict[str, Any] | None,
    source_language: str,
) -> list[SemanticCheckResult]:
    checks: list[SemanticCheckResult] = []

    validated_plan = (final_state or {}).get("validated_plan")
    canonical_ir = None
    planning_graph = None
    if validated_plan is not None and getattr(validated_plan, "plan", None) is not None:
        canonical_ir = validated_plan.plan.canonical_ir
        planning_graph = validated_plan.plan.planning_graph

    for expectation in case.semantic_expectations:
        kind = expectation.check
        passed = False
        detail = ""

        if kind == "generated_atoms_contains":
            passed = expectation.value in bundle.generated_atoms
            detail = expectation.value
        elif kind == "generated_atoms_not_contains":
            passed = expectation.value not in bundle.generated_atoms
            detail = expectation.value
        elif kind == "generated_state_models_contains":
            passed = expectation.value in bundle.generated_state_models
            detail = expectation.value
        elif kind == "generated_witnesses_contains":
            passed = expectation.value in bundle.generated_witnesses
            detail = expectation.value
        elif kind == "has_canonical_ir":
            passed = canonical_ir is not None
            detail = "canonical_ir present" if passed else "canonical_ir missing"
        elif kind == "has_planning_graph":
            passed = planning_graph is not None
            detail = "planning_graph present" if passed else "planning_graph missing"
        elif kind == "source_language_equals":
            passed = source_language == expectation.value
            detail = f"observed={source_language}"
        elif kind == "cdg_node_count_at_least":
            minimum = int(expectation.minimum or 0)
            observed = len(bundle.cdg.nodes)
            passed = observed >= minimum
            detail = f"observed={observed} minimum={minimum}"
        elif kind == "mypy_passed":
            passed = bundle.mypy_passed
            detail = f"observed={bundle.mypy_passed}"
        elif kind == "ghost_passed":
            passed = bundle.ghost_sim_passed
            detail = f"observed={bundle.ghost_sim_passed}"
        else:
            detail = f"unsupported expectation: {kind}"

        checks.append(SemanticCheckResult(check=kind, passed=passed, detail=detail))

    return checks


async def run_ingest_regression_case(
    case: IngestRegressionCase,
    *,
    output_root: str | Path,
    agent_factory: AgentFactory,
    stale_seconds: int = 120,
) -> IngestRegressionResult:
    """Run one curated ingest case via the public ingester entrypoints."""

    case_dir = Path(output_root) / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    source_path = _materialize_case_source(case, case_dir)
    monitor = IngestMonitor(case_dir, enable_trace=True, stale_seconds=stale_seconds)
    monitor.start(
        source_path=str(source_path),
        class_name=case.class_name,
        procedural=case.procedural,
        llm_provider="regression_harness",
        llm_model="deterministic_fixture",
        max_depth=1,
    )

    started = time.perf_counter()
    final_state: dict[str, Any] | None = None
    bundle = _empty_bundle()
    published_artifacts: list[str] = []
    error = ""
    failed_phase = ""
    source_language = case.expected_language

    agent = agent_factory(case_dir, monitor, case)
    try:
        if case.procedural:
            bundle = await agent.ingest_procedural(str(source_path), case.class_name)
            published_artifacts = _stage_success_artifacts(monitor, bundle=bundle)
            source_language = case.expected_language
            monitor.complete(
                summary={
                    "cdg_nodes": len(bundle.cdg.nodes),
                    "cdg_edges": len(bundle.cdg.edges),
                    "matches": len(bundle.match_results),
                    "mypy_passed": bool(bundle.mypy_passed),
                    "ghost_sim_passed": bool(bundle.ghost_sim_passed),
                    "published_files": published_artifacts,
                }
            )
        else:
            final_state = await agent.ingest_state(str(source_path), case.class_name)
            if final_state.get("raw_dfg") is not None:
                source_language = str(final_state["raw_dfg"].source_language or source_language)
            if final_state.get("error"):
                error = str(final_state.get("error") or "")
                failed_phase = str(IngestMonitor.read_status(case_dir).get("phase") or "")
                published_artifacts = _stage_failure_artifacts(
                    monitor,
                    state=final_state,
                )
                monitor.fail(error=error)
                maybe_bundle = final_state.get("bundle")
                if maybe_bundle is not None:
                    bundle = maybe_bundle
            else:
                bundle = final_state["bundle"]
                published_artifacts = _stage_success_artifacts(monitor, bundle=bundle)
                monitor.complete(
                    summary={
                        "cdg_nodes": len(bundle.cdg.nodes),
                        "cdg_edges": len(bundle.cdg.edges),
                        "matches": len(bundle.match_results),
                        "mypy_passed": bool(bundle.mypy_passed),
                        "ghost_sim_passed": bool(bundle.ghost_sim_passed),
                        "published_files": published_artifacts,
                    }
                )
    except Exception as exc:
        error = str(exc)
        if final_state is not None:
            published_artifacts = _stage_failure_artifacts(monitor, state=final_state)
        monitor.fail(error=error)
    finally:
        proof_env = getattr(getattr(agent, "_deps", None), "proof_env", None)
        close = getattr(proof_env, "close", None)
        if callable(close):
            maybe_awaitable = close()
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable

    runtime_ms = (time.perf_counter() - started) * 1000.0
    trace_summary = summarize_monitor_trace(case_dir, stale_seconds=stale_seconds)
    status = IngestMonitor.read_status(case_dir)
    failed_phase = failed_phase or str(status.get("phase") or "")

    semantic_checks = _evaluate_semantic_expectations(
        case,
        bundle=bundle,
        final_state=final_state,
        source_language=source_language,
    )
    validated_plan = (final_state or {}).get("validated_plan")
    has_canonical_ir = bool(
        validated_plan is not None
        and getattr(validated_plan, "plan", None) is not None
        and validated_plan.plan.canonical_ir is not None
    )
    has_planning_graph = bool(
        validated_plan is not None
        and getattr(validated_plan, "plan", None) is not None
        and validated_plan.plan.planning_graph is not None
    )
    completed = trace_summary.classified_state == "completed" and not error
    type_reason = ""
    ghost_reason = ""
    if final_state:
        type_reason = str(
            (final_state.get("type_failure_classification") or {}).get("reason_code") or ""
        )
        ghost_reason = str(
            (final_state.get("ghost_failure_classification") or {}).get("reason_code") or ""
        )
        error = error or str(final_state.get("error") or "")

    return IngestRegressionResult(
        case_id=case.case_id,
        family=case.family,
        completed=completed,
        failed_phase=failed_phase,
        timed_out_or_stalled=trace_summary.timed_out_or_stalled,
        mypy_passed=bool(bundle.mypy_passed),
        ghost_passed=bool(bundle.ghost_sim_passed),
        type_failure_reason=type_reason,
        ghost_failure_reason=ghost_reason,
        llm_call_count=trace_summary.llm_call_count,
        llm_prompt_counts=trace_summary.llm_prompt_counts,
        runtime_ms=runtime_ms,
        published_artifacts=published_artifacts,
        semantic_checks=semantic_checks,
        error=error,
        output_dir=str(case_dir),
        source_language=source_language,
        has_canonical_ir=has_canonical_ir,
        has_planning_graph=has_planning_graph,
    )


async def run_ingest_regression_suite(
    cases: list[IngestRegressionCase],
    *,
    output_root: str | Path,
    agent_factory: AgentFactory,
    stale_seconds: int = 120,
) -> tuple[list[IngestRegressionResult], IngestRegressionSummary]:
    """Run a curated suite sequentially and summarize the results."""

    results: list[IngestRegressionResult] = []
    for case in cases:
        results.append(
            await run_ingest_regression_case(
                case,
                output_root=output_root,
                agent_factory=agent_factory,
                stale_seconds=stale_seconds,
            )
        )
    return results, summarize_ingest_regression_results(results)
