"""Lightweight regression harness for curated ingest coverage."""

from __future__ import annotations

import json
import inspect
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from sciona.architect.handoff import CDGExport
from sciona.ingester.graph import IngesterAgent
from sciona.ingester.models import IngestIRPlan, IngestPlanGraph, IngestionBundle
from sciona.ingester.monitor import (
    IngestMonitor,
    OUTPUT_SCOPE_SYMBOL,
    TRACE_FILE,
)

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
    output_scope: str = OUTPUT_SCOPE_SYMBOL
    procedural: bool = False
    expected_language: str = "python"
    source_path: str = ""
    inline_source: str = ""
    fixture_origin: str = ""
    golden_case_id: str = ""
    expected_artifacts: list[str] = Field(default_factory=list)
    optional_artifacts: list[str] = Field(default_factory=list)
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
    cache_state: str = "unknown"
    cache_state_source: str = ""
    runtime_ms: float = 0.0
    published_artifacts: list[str] = Field(default_factory=list)
    semantic_checks: list[SemanticCheckResult] = Field(default_factory=list)
    error: str = ""
    output_dir: str = ""
    source_language: str = ""
    has_canonical_ir: bool = False
    has_planning_graph: bool = False
    compared_artifacts: list[str] = Field(default_factory=list)
    mismatched_artifacts: list[str] = Field(default_factory=list)
    golden_mismatch_details: dict[str, str] = Field(default_factory=dict)
    golden_match: bool | None = None
    has_verification_failure_artifact: bool = False


class FamilyBreakdown(BaseModel):
    """Per-family summary row."""

    total_cases: int = 0
    completed_cases: int = 0
    mypy_passed_cases: int = 0
    ghost_passed_cases: int = 0
    llm_call_total: int = 0
    stalled_cases: int = 0
    cache_hit_cases: int = 0
    cache_miss_cases: int = 0
    cache_unknown_cases: int = 0
    runtime_ms_total: float = 0.0
    runtime_ms_avg: float = 0.0
    runtime_ms_p50: float = 0.0
    runtime_ms_max: float = 0.0
    golden_compared_cases: int = 0
    golden_mismatched_cases: int = 0


class IngestRegressionSummary(BaseModel):
    """Aggregated summary across a curated suite."""

    total_cases: int = 0
    completed_cases: int = 0
    completion_rate: float = 0.0
    mypy_pass_rate: float = 0.0
    ghost_pass_rate: float = 0.0
    timeout_or_stall_count: int = 0
    llm_call_total: int = 0
    cache_hit_cases: int = 0
    cache_miss_cases: int = 0
    cache_unknown_cases: int = 0
    cache_observed_cases: int = 0
    cache_hit_rate: float = 0.0
    runtime_ms_total: float = 0.0
    runtime_ms_avg: float = 0.0
    runtime_ms_p50: float = 0.0
    runtime_ms_max: float = 0.0
    semantic_check_pass_rate: float = 0.0
    golden_compared_cases: int = 0
    golden_matched_cases: int = 0
    golden_match_rate: float = 0.0
    family_breakdown: dict[str, FamilyBreakdown] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)


class MonitorTraceSummary(BaseModel):
    """Normalized monitor/trace metrics for one ingest run."""

    llm_call_count: int = 0
    llm_prompt_counts: dict[str, int] = Field(default_factory=dict)
    timed_out_or_stalled: bool = False
    classified_state: str = "missing"
    status_phase: str = ""
    marker_state: str = ""
    cache_state: str = "unknown"
    cache_state_source: str = ""


class NormalizedArtifactBundle(BaseModel):
    """Normalized reviewable snapshot surfaces for one case."""

    case_id: str
    artifacts: dict[str, str] = Field(default_factory=dict)


class GoldenArtifactComparison(BaseModel):
    """Golden comparison result for one case."""

    matched: bool = True
    compared_artifacts: list[str] = Field(default_factory=list)
    mismatched_artifacts: list[str] = Field(default_factory=list)
    mismatch_details: dict[str, str] = Field(default_factory=dict)


_SOURCE_EXTENSIONS: dict[str, str] = {
    "python": ".py",
    "rust": ".rs",
    "cpp": ".cpp",
    "c++": ".cpp",
    "julia": ".jl",
}

_ARTIFACT_FILE_NAMES: dict[str, str] = {
    "canonical_ir": "canonical_ir.json",
    "planning_graph": "planning_graph.json",
    "atoms": "atoms.py",
    "state_models": "state_models.py",
    "witnesses": "witnesses.py",
    "cdg": "cdg.json",
    "verification_failure": "verification_failure.json",
}

_JSON_ARTIFACTS: set[str] = {
    "canonical_ir",
    "planning_graph",
    "cdg",
    "verification_failure",
}

_DEFAULT_REQUIRED_ARTIFACTS: list[str] = [
    "canonical_ir",
    "planning_graph",
    "atoms",
    "witnesses",
    "cdg",
]

_TRANSIENT_JSON_KEYS: set[str] = {
    "timestamp",
    "run_id",
    "started_at",
    "ended_at",
    "last_heartbeat_at",
    "llm_call_inflight",
}

_ORDER_HINT_KEYS: tuple[str, ...] = (
    "operation_id",
    "group_id",
    "node_id",
    "slot_name",
    "source_id",
    "target_id",
    "name",
    "method_name",
    "output_name",
    "input_name",
)

_ABS_WIN_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_fixture_root() -> Path:
    return _repo_root() / "tests" / "fixtures" / "ingest_regression"


def _fixture_source_path(
    fixture_root: Path,
    *,
    case_id: str,
    expected_language: str,
) -> str:
    suffix = _SOURCE_EXTENSIONS.get(expected_language.lower(), ".txt")
    return str((fixture_root / case_id / f"source{suffix}").resolve())


def default_ingest_regression_cases(
    *,
    fixture_root: str | Path | None = None,
) -> list[IngestRegressionCase]:
    """Return a curated real-world matrix spanning protected families."""

    root = Path(fixture_root) if fixture_root is not None else _default_fixture_root()
    return [
        IngestRegressionCase(
            case_id="sklearn_style_estimator",
            family="sklearn_estimator",
            class_name="CalibratedStyleClassifier",
            expected_language="python",
            source_path=_fixture_source_path(
                root,
                case_id="sklearn_style_estimator",
                expected_language="python",
            ),
            fixture_origin="tests/test_ingest.py",
            expected_artifacts=list(_DEFAULT_REQUIRED_ARTIFACTS),
            semantic_expectations=[
                SemanticExpectation(check="has_canonical_ir"),
                SemanticExpectation(check="has_planning_graph"),
                SemanticExpectation(check="source_language_equals", value="python"),
            ],
        ),
        IngestRegressionCase(
            case_id="rolling_stateful_class",
            family="rolling_stateful",
            class_name="RollingAverager",
            expected_language="python",
            source_path=_fixture_source_path(
                root,
                case_id="rolling_stateful_class",
                expected_language="python",
            ),
            fixture_origin="tests/test_ingest_stateful.py",
            expected_artifacts=[*list(_DEFAULT_REQUIRED_ARTIFACTS), "state_models"],
            semantic_expectations=[
                SemanticExpectation(check="has_canonical_ir"),
                SemanticExpectation(check="source_language_equals", value="python"),
            ],
        ),
        IngestRegressionCase(
            case_id="sklearn_grouped_images",
            family="sklearn_grouped_images",
            class_name="grid_to_graph",
            output_scope="family",
            expected_language="python",
            source_path=_fixture_source_path(
                root,
                case_id="sklearn_grouped_images",
                expected_language="python",
            ),
            fixture_origin="INGESTER_ROLLOUT_WORKSTREAM2_PLAN.md",
            expected_artifacts=list(_DEFAULT_REQUIRED_ARTIFACTS),
            semantic_expectations=[
                SemanticExpectation(check="has_canonical_ir"),
                SemanticExpectation(check="source_language_equals", value="python"),
                SemanticExpectation(check="generated_atoms_contains", value="def grid_to_graph"),
                SemanticExpectation(
                    check="generated_atoms_contains",
                    value="def extract_patches_2d",
                ),
            ],
        ),
        IngestRegressionCase(
            case_id="bayesian_or_message_passing",
            family="bayesian_or_message_passing",
            class_name="PosteriorAccumulator",
            expected_language="python",
            source_path=_fixture_source_path(
                root,
                case_id="bayesian_or_message_passing",
                expected_language="python",
            ),
            fixture_origin="tests/test_bayesian_ingester.py",
            expected_artifacts=list(_DEFAULT_REQUIRED_ARTIFACTS),
            semantic_expectations=[
                SemanticExpectation(check="has_canonical_ir"),
                SemanticExpectation(check="source_language_equals", value="python"),
            ],
        ),
        IngestRegressionCase(
            case_id="detector_structured_output",
            family="detector_structured_output",
            class_name="PeakDetector",
            expected_language="python",
            source_path=_fixture_source_path(
                root,
                case_id="detector_structured_output",
                expected_language="python",
            ),
            fixture_origin="INGESTER_ROLLOUT_WORKSTREAM2_PLAN.md",
            expected_artifacts=list(_DEFAULT_REQUIRED_ARTIFACTS),
            semantic_expectations=[
                SemanticExpectation(check="has_canonical_ir"),
                SemanticExpectation(check="source_language_equals", value="python"),
                SemanticExpectation(check="generated_atoms_contains", value='"rpeaks"'),
                SemanticExpectation(check="generated_atoms_contains", value='"quality"'),
            ],
        ),
        IngestRegressionCase(
            case_id="dsp_biosignal_pipeline",
            family="dsp_biosignal",
            class_name="ECGProcessor",
            expected_language="python",
            source_path=_fixture_source_path(
                root,
                case_id="dsp_biosignal_pipeline",
                expected_language="python",
            ),
            fixture_origin="tests/test_ingest_biosppy_ecg.py",
            expected_artifacts=[*list(_DEFAULT_REQUIRED_ARTIFACTS), "state_models"],
            semantic_expectations=[
                SemanticExpectation(check="has_canonical_ir"),
                SemanticExpectation(check="source_language_equals", value="python"),
            ],
        ),
        IngestRegressionCase(
            case_id="non_python_ffi",
            family="non_python_ffi",
            class_name="Integrator",
            expected_language="rust",
            source_path=_fixture_source_path(
                root,
                case_id="non_python_ffi",
                expected_language="rust",
            ),
            fixture_origin="tests/test_treesitter_rust.py",
            expected_artifacts=list(_DEFAULT_REQUIRED_ARTIFACTS),
            semantic_expectations=[
                SemanticExpectation(check="source_language_equals", value="rust"),
                SemanticExpectation(check="has_canonical_ir"),
            ],
        ),
        IngestRegressionCase(
            case_id="procedural_ingest",
            family="procedural_ingest",
            class_name="PulsarFold",
            procedural=True,
            expected_language="python",
            source_path=_fixture_source_path(
                root,
                case_id="procedural_ingest",
                expected_language="python",
            ),
            fixture_origin="tests/test_ingest_procedural.py",
            expected_artifacts=["atoms", "witnesses", "cdg"],
            semantic_expectations=[
                SemanticExpectation(check="cdg_node_count_at_least", minimum=3),
            ],
        ),
    ]


def _is_transient_json_key(key: str) -> bool:
    return key in _TRANSIENT_JSON_KEYS or key.endswith("_at")


def _normalize_source_text(source: str) -> str:
    lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized = "\n".join(line.rstrip() for line in lines).strip()
    if not normalized:
        return ""
    return normalized + "\n"


def _normalize_path_like_text(value: str, *, output_dir: Path) -> str:
    normalized = value.replace("\\", "/")
    case_dir = output_dir.as_posix()
    output_root = output_dir.parent.as_posix()
    if case_dir and case_dir in normalized:
        normalized = normalized.replace(case_dir, "<case_output_dir>")
    if output_root and output_root in normalized:
        normalized = normalized.replace(output_root, "<output_root>")
    if normalized.startswith("/"):
        suffix = Path(normalized).name
        return f"<abs>/{suffix}" if suffix else "<abs>"
    if _ABS_WIN_PATH.match(normalized):
        suffix = Path(normalized).name
        return f"<abs>/{suffix}" if suffix else "<abs>"
    return normalized


def _sortable_dict_key(item: dict[str, Any]) -> str:
    for key in _ORDER_HINT_KEYS:
        value = item.get(key)
        if value:
            return f"{key}:{value}"
    return json.dumps(item, sort_keys=True, separators=(",", ":"))


def normalize_snapshot_payload(payload: Any, *, output_dir: str | Path) -> Any:
    """Normalize JSON payloads so goldens capture semantics, not runtime noise."""

    case_dir = Path(output_dir)

    if isinstance(payload, dict):
        normalized_items: dict[str, Any] = {}
        for key in sorted(payload):
            if _is_transient_json_key(key):
                continue
            normalized_items[key] = normalize_snapshot_payload(
                payload[key],
                output_dir=case_dir,
            )
        return normalized_items

    if isinstance(payload, list):
        normalized_items = [
            normalize_snapshot_payload(item, output_dir=case_dir)
            for item in payload
        ]
        if all(not isinstance(item, (dict, list)) for item in normalized_items):
            return sorted(
                normalized_items,
                key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
            )
        if all(isinstance(item, dict) for item in normalized_items):
            has_hints = any(
                any(hint in item for hint in _ORDER_HINT_KEYS)
                for item in normalized_items
            )
            if has_hints:
                return sorted(normalized_items, key=_sortable_dict_key)
        return normalized_items

    if isinstance(payload, str):
        return _normalize_path_like_text(payload, output_dir=case_dir)

    return payload


def _model_payload(value: Any) -> Any:
    if value is None:
        return None
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return value


def _coerce_json_artifact_payload(artifact_name: str, payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    try:
        if artifact_name == "canonical_ir":
            return IngestIRPlan.model_validate(payload).model_dump(mode="json")
        if artifact_name == "planning_graph":
            return IngestPlanGraph.model_validate(payload).model_dump(mode="json")
        if artifact_name == "cdg":
            return CDGExport.model_validate(payload).model_dump(mode="json")
    except Exception:
        return payload
    return payload


def _serialize_normalized_artifact(
    artifact_name: str,
    payload: Any,
    *,
    output_dir: Path,
) -> str:
    if artifact_name in _JSON_ARTIFACTS:
        if isinstance(payload, str):
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError:
                decoded = {"raw": _normalize_source_text(payload)}
        else:
            decoded = payload
        decoded = _coerce_json_artifact_payload(artifact_name, decoded)
        normalized = normalize_snapshot_payload(decoded, output_dir=output_dir)
        return json.dumps(normalized, indent=2, sort_keys=True)
    if not isinstance(payload, str):
        payload = str(payload)
    return _normalize_source_text(payload)


def capture_normalized_artifact_bundle(
    case: IngestRegressionCase,
    *,
    output_dir: str | Path,
    bundle: IngestionBundle,
    final_state: dict[str, Any] | None,
) -> NormalizedArtifactBundle:
    """Collect normalized semantic artifact snapshots for one case."""

    case_output_dir = Path(output_dir)
    artifacts: dict[str, str] = {}

    validated_plan = (final_state or {}).get("validated_plan")
    plan = getattr(validated_plan, "plan", None) if validated_plan is not None else None
    canonical_ir = getattr(plan, "canonical_ir", None) if plan is not None else None
    planning_graph = getattr(plan, "planning_graph", None) if plan is not None else None

    if canonical_ir is not None:
        artifacts["canonical_ir"] = _serialize_normalized_artifact(
            "canonical_ir",
            _model_payload(canonical_ir),
            output_dir=case_output_dir,
        )
    if planning_graph is not None:
        artifacts["planning_graph"] = _serialize_normalized_artifact(
            "planning_graph",
            _model_payload(planning_graph),
            output_dir=case_output_dir,
        )
    if bundle.generated_atoms:
        artifacts["atoms"] = _serialize_normalized_artifact(
            "atoms",
            bundle.generated_atoms,
            output_dir=case_output_dir,
        )
    if bundle.generated_state_models:
        artifacts["state_models"] = _serialize_normalized_artifact(
            "state_models",
            bundle.generated_state_models,
            output_dir=case_output_dir,
        )
    if bundle.generated_witnesses:
        artifacts["witnesses"] = _serialize_normalized_artifact(
            "witnesses",
            bundle.generated_witnesses,
            output_dir=case_output_dir,
        )
    artifacts["cdg"] = _serialize_normalized_artifact(
        "cdg",
        _model_payload(bundle.cdg),
        output_dir=case_output_dir,
    )

    verification_path = case_output_dir / _ARTIFACT_FILE_NAMES["verification_failure"]
    if verification_path.exists():
        verification_payload = json.loads(verification_path.read_text())
        artifacts["verification_failure"] = _serialize_normalized_artifact(
            "verification_failure",
            verification_payload,
            output_dir=case_output_dir,
        )
    elif final_state is not None and final_state.get("error"):
        fallback_failure = {
            "stage": str((final_state.get("failed_phase") or final_state.get("phase") or "")),
            "reason_code": str(
                (final_state.get("type_failure_classification") or {}).get("reason_code")
                or (final_state.get("ghost_failure_classification") or {}).get("reason_code")
                or "ingest_failure"
            ),
            "error": str(final_state.get("error") or ""),
        }
        artifacts["verification_failure"] = _serialize_normalized_artifact(
            "verification_failure",
            fallback_failure,
            output_dir=case_output_dir,
        )

    return NormalizedArtifactBundle(
        case_id=case.case_id,
        artifacts=artifacts,
    )


def compare_case_artifacts_to_goldens(
    case: IngestRegressionCase,
    *,
    observed: NormalizedArtifactBundle,
    golden_root: str | Path,
    output_dir: str | Path,
) -> GoldenArtifactComparison:
    """Compare normalized artifacts against checked-in golden files."""

    case_output_dir = Path(output_dir)
    golden_dir = Path(golden_root) / "ingest_regression" / (case.golden_case_id or case.case_id)
    required = list(case.expected_artifacts or _DEFAULT_REQUIRED_ARTIFACTS)
    optional = set(case.optional_artifacts)

    mismatched: list[str] = []
    mismatch_details: dict[str, str] = {}
    compared: list[str] = []

    for artifact_name in required:
        compared.append(artifact_name)
        file_name = _ARTIFACT_FILE_NAMES.get(artifact_name)
        if not file_name:
            mismatched.append(artifact_name)
            mismatch_details[artifact_name] = "unsupported_artifact"
            continue

        observed_text = observed.artifacts.get(artifact_name)
        golden_path = golden_dir / file_name
        has_golden = golden_path.exists()

        if observed_text is None and not has_golden and artifact_name in optional:
            continue
        if observed_text is None and has_golden:
            mismatched.append(artifact_name)
            mismatch_details[artifact_name] = "missing_observed_artifact"
            continue
        if observed_text is None and not has_golden:
            mismatched.append(artifact_name)
            mismatch_details[artifact_name] = "missing_observed_and_golden_artifact"
            continue
        if not has_golden:
            mismatched.append(artifact_name)
            mismatch_details[artifact_name] = "missing_golden_artifact"
            continue

        golden_text = _serialize_normalized_artifact(
            artifact_name,
            golden_path.read_text(),
            output_dir=case_output_dir,
        )
        if observed_text != golden_text:
            mismatched.append(artifact_name)
            mismatch_details[artifact_name] = "content_mismatch"

    return GoldenArtifactComparison(
        matched=not mismatched,
        compared_artifacts=compared,
        mismatched_artifacts=mismatched,
        mismatch_details=mismatch_details,
    )


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


def _normalize_cache_state(value: Any) -> str:
    if isinstance(value, bool):
        return "hit" if value else "miss"
    text = str(value or "").strip().lower()
    if text in {"hit", "cache_hit", "cached", "true", "1", "yes"}:
        return "hit"
    if text in {"miss", "cache_miss", "uncached", "false", "0", "no"}:
        return "miss"
    return "unknown"


def _extract_cache_state_from_mapping(
    payload: Any, *, source: str
) -> tuple[str, str]:
    if not isinstance(payload, dict):
        return "unknown", ""

    for key in ("cache_state", "cache_status"):
        state = _normalize_cache_state(payload.get(key))
        if state != "unknown":
            return state, f"{source}.{key}"

    if "cache_hit" in payload:
        state = _normalize_cache_state(payload.get("cache_hit"))
        if state != "unknown":
            return state, f"{source}.cache_hit"

    cache_payload = payload.get("cache")
    if isinstance(cache_payload, dict):
        for key in ("state", "status", "hit"):
            state = _normalize_cache_state(cache_payload.get(key))
            if state != "unknown":
                return state, f"{source}.cache.{key}"

    return "unknown", ""


def _extract_cache_state_from_surface(surface: dict[str, Any]) -> tuple[str, str]:
    candidates: list[tuple[str, Any]] = [
        ("surface.status", surface.get("status")),
        ("surface.status.summary", (surface.get("status") or {}).get("summary")),
        ("surface.marker", surface.get("marker")),
        ("surface.marker.summary", (surface.get("marker") or {}).get("summary")),
    ]
    for source, payload in candidates:
        state, resolved_source = _extract_cache_state_from_mapping(payload, source=source)
        if state != "unknown":
            return state, resolved_source
    return "unknown", ""


def _p50(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


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

    surface = IngestMonitor.read_surface(output_dir, stale_seconds=stale_seconds)
    status = surface.get("status", {}) if isinstance(surface, dict) else {}
    marker = surface.get("marker", {}) if isinstance(surface, dict) else {}
    classified = str(surface.get("derived_state") or "missing")
    cache_state, cache_source = _extract_cache_state_from_surface(
        surface if isinstance(surface, dict) else {}
    )
    return MonitorTraceSummary(
        llm_call_count=sum(prompt_counts.values()),
        llm_prompt_counts=dict(sorted(prompt_counts.items())),
        timed_out_or_stalled=classified == "stalled",
        classified_state=classified,
        status_phase=str(status.get("phase") or ""),
        marker_state=str(marker.get("state") or ""),
        cache_state=cache_state,
        cache_state_source=cache_source,
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
    runtime_values = [max(0.0, float(item.runtime_ms)) for item in results]
    runtime_ms_total = sum(runtime_values)
    runtime_ms_avg = (runtime_ms_total / total_cases) if total_cases else 0.0
    runtime_ms_p50 = _p50(runtime_values)
    runtime_ms_max = max(runtime_values, default=0.0)
    golden_compared_cases = sum(1 for item in results if item.golden_match is not None)
    golden_matched_cases = sum(1 for item in results if item.golden_match is True)
    cache_hit_cases = sum(1 for item in results if _normalize_cache_state(item.cache_state) == "hit")
    cache_miss_cases = sum(
        1 for item in results if _normalize_cache_state(item.cache_state) == "miss"
    )
    cache_unknown_cases = total_cases - cache_hit_cases - cache_miss_cases
    cache_observed_cases = cache_hit_cases + cache_miss_cases

    semantic_total = sum(len(item.semantic_checks) for item in results)
    semantic_passed = sum(
        1
        for item in results
        for check in item.semantic_checks
        if check.passed
    )

    family_breakdown: dict[str, FamilyBreakdown] = {}
    family_runtime_values: dict[str, list[float]] = {}
    for item in results:
        family_row = family_breakdown.setdefault(item.family, FamilyBreakdown())
        family_row.total_cases += 1
        family_row.completed_cases += int(item.completed)
        family_row.mypy_passed_cases += int(item.mypy_passed)
        family_row.ghost_passed_cases += int(item.ghost_passed)
        family_row.llm_call_total += item.llm_call_count
        family_row.stalled_cases += int(item.timed_out_or_stalled)
        cache_state = _normalize_cache_state(item.cache_state)
        if cache_state == "hit":
            family_row.cache_hit_cases += 1
        elif cache_state == "miss":
            family_row.cache_miss_cases += 1
        else:
            family_row.cache_unknown_cases += 1
        runtime_ms = max(0.0, float(item.runtime_ms))
        family_row.runtime_ms_total += runtime_ms
        family_runtime_values.setdefault(item.family, []).append(runtime_ms)
        family_row.golden_compared_cases += int(item.golden_match is not None)
        family_row.golden_mismatched_cases += int(item.golden_match is False)

    for family, family_row in family_breakdown.items():
        runtimes = family_runtime_values.get(family, [])
        family_row.runtime_ms_avg = (
            family_row.runtime_ms_total / family_row.total_cases
            if family_row.total_cases
            else 0.0
        )
        family_row.runtime_ms_p50 = _p50(runtimes)
        family_row.runtime_ms_max = max(runtimes, default=0.0)

    failures: list[str] = []
    for item in results:
        if not item.completed or item.error or item.golden_match is False:
            failures.append(item.case_id)

    return IngestRegressionSummary(
        total_cases=total_cases,
        completed_cases=completed_cases,
        completion_rate=(completed_cases / total_cases) if total_cases else 0.0,
        mypy_pass_rate=(mypy_passed_cases / total_cases) if total_cases else 0.0,
        ghost_pass_rate=(ghost_passed_cases / total_cases) if total_cases else 0.0,
        timeout_or_stall_count=timeout_or_stall_count,
        llm_call_total=llm_call_total,
        cache_hit_cases=cache_hit_cases,
        cache_miss_cases=cache_miss_cases,
        cache_unknown_cases=cache_unknown_cases,
        cache_observed_cases=cache_observed_cases,
        cache_hit_rate=(cache_hit_cases / cache_observed_cases) if cache_observed_cases else 0.0,
        runtime_ms_total=runtime_ms_total,
        runtime_ms_avg=runtime_ms_avg,
        runtime_ms_p50=runtime_ms_p50,
        runtime_ms_max=runtime_ms_max,
        semantic_check_pass_rate=(
            semantic_passed / semantic_total if semantic_total else 0.0
        ),
        golden_compared_cases=golden_compared_cases,
        golden_matched_cases=golden_matched_cases,
        golden_match_rate=(
            golden_matched_cases / golden_compared_cases if golden_compared_cases else 0.0
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
    validated_plan: Any | None = None,
) -> list[str]:
    plan = getattr(validated_plan, "plan", None) if validated_plan is not None else None
    canonical_ir = getattr(plan, "canonical_ir", None) if plan is not None else None
    planning_graph = getattr(plan, "planning_graph", None) if plan is not None else None
    if canonical_ir is not None:
        monitor.stage_json("canonical_ir.json", _model_payload(canonical_ir))
    if planning_graph is not None:
        monitor.stage_json("planning_graph.json", _model_payload(planning_graph))
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
        plan = getattr(validated_plan, "plan", None)
        if plan is not None and getattr(plan, "canonical_ir", None) is not None:
            monitor.stage_json(
                "canonical_ir.json",
                _model_payload(plan.canonical_ir),
            )
        if plan is not None and getattr(plan, "planning_graph", None) is not None:
            monitor.stage_json(
                "planning_graph.json",
                _model_payload(plan.planning_graph),
            )
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
    golden_root: str | Path | None = None,
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
        output_scope=case.output_scope,
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
            published_artifacts = _stage_success_artifacts(
                monitor,
                bundle=bundle,
                validated_plan=None,
            )
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
                published_artifacts = _stage_success_artifacts(
                    monitor,
                    bundle=bundle,
                    validated_plan=final_state.get("validated_plan"),
                )
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
    failed_phase = failed_phase or trace_summary.status_phase

    semantic_checks = _evaluate_semantic_expectations(
        case,
        bundle=bundle,
        final_state=final_state,
        source_language=source_language,
    )
    normalized_bundle = capture_normalized_artifact_bundle(
        case,
        output_dir=case_dir,
        bundle=bundle,
        final_state=final_state,
    )
    golden_comparison = GoldenArtifactComparison(
        matched=True,
        compared_artifacts=[],
        mismatched_artifacts=[],
        mismatch_details={},
    )
    if golden_root is not None:
        golden_comparison = compare_case_artifacts_to_goldens(
            case,
            observed=normalized_bundle,
            golden_root=golden_root,
            output_dir=case_dir,
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
        cache_state=trace_summary.cache_state,
        cache_state_source=trace_summary.cache_state_source,
        runtime_ms=runtime_ms,
        published_artifacts=published_artifacts,
        semantic_checks=semantic_checks,
        error=error,
        output_dir=str(case_dir),
        source_language=source_language,
        has_canonical_ir=has_canonical_ir,
        has_planning_graph=has_planning_graph,
        compared_artifacts=golden_comparison.compared_artifacts,
        mismatched_artifacts=golden_comparison.mismatched_artifacts,
        golden_mismatch_details=golden_comparison.mismatch_details,
        golden_match=golden_comparison.matched if golden_root is not None else None,
        has_verification_failure_artifact=(
            "verification_failure" in normalized_bundle.artifacts
        ),
    )


async def run_ingest_regression_suite(
    cases: list[IngestRegressionCase],
    *,
    output_root: str | Path,
    agent_factory: AgentFactory,
    stale_seconds: int = 120,
    golden_root: str | Path | None = None,
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
                golden_root=golden_root,
            )
        )
    return results, summarize_ingest_regression_results(results)
