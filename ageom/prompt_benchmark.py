"""Prompt-key benchmark harness for provider A/B comparisons."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ageom.hunter.nodes import _INT_ARRAY_GBNF, _STRING_ARRAY_GBNF
from ageom.hunter.prompts import (
    ANALYZE_FAILURE_SYSTEM,
    ANALYZE_FAILURE_USER,
    REFORMULATE_QUERY_SYSTEM,
    REFORMULATE_QUERY_USER,
    SCORE_CANDIDATES_SYSTEM,
    SCORE_CANDIDATES_USER,
)
from ageom.json_utils import extract_json
from ageom.llm_router import (
    HUNTER_ANALYZE_FAILURE,
    HUNTER_REFORMULATE,
    HUNTER_SCORE,
)


@dataclass(frozen=True)
class PromptBenchmarkCase:
    case_id: str
    domain: str
    prompt_key: str
    system: str
    user: str
    expected: dict[str, Any]
    grammar: str = ""
    use_grammar: bool = True
    baseline_system: str = ""
    baseline_user: str = ""


@dataclass(frozen=True)
class PromptBenchmarkProvider:
    name: str
    client: Any


@dataclass
class PromptBenchmarkResult:
    case_id: str
    domain: str
    prompt_key: str
    provider: str
    model: str
    variant: str
    latency_ms: float
    ok: bool
    validation_error: str = ""
    output_preview: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromptBenchmarkAggregate:
    provider: str
    model: str
    variant: str
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    by_prompt_key: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def record(self, result: PromptBenchmarkResult) -> None:
        self.total_cases += 1
        if result.ok:
            self.passed_cases += 1
        else:
            self.failed_cases += 1
        total_latency = self.avg_latency_ms * (self.total_cases - 1)
        total_latency += result.latency_ms
        self.avg_latency_ms = total_latency / max(1, self.total_cases)
        self.max_latency_ms = max(self.max_latency_ms, result.latency_ms)

        bucket = self.by_prompt_key.setdefault(
            result.prompt_key,
            {"cases": 0, "passed": 0, "failed": 0, "avg_latency_ms": 0.0},
        )
        cases = int(bucket["cases"]) + 1
        bucket["cases"] = cases
        bucket["passed"] = int(bucket["passed"]) + (1 if result.ok else 0)
        bucket["failed"] = int(bucket["failed"]) + (0 if result.ok else 1)
        bucket["avg_latency_ms"] = (
            (float(bucket["avg_latency_ms"]) * (cases - 1)) + result.latency_ms
        ) / max(1, cases)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _score_case(
    *,
    case_id: str,
    domain: str,
    statement: str,
    informal_desc: str,
    candidates: Sequence[tuple[str, str]],
    expected_first_index: int,
) -> PromptBenchmarkCase:
    candidates_list = "\n".join(
        f"[{idx}] {name} : {type_signature}"
        for idx, (name, type_signature) in enumerate(candidates)
    )
    return PromptBenchmarkCase(
        case_id=case_id,
        domain=domain,
        prompt_key=HUNTER_SCORE,
        system=SCORE_CANDIDATES_SYSTEM,
        user=SCORE_CANDIDATES_USER.format(
            statement=statement,
            informal_desc=informal_desc,
            candidates_list=candidates_list,
        ),
        grammar=_INT_ARRAY_GBNF,
        expected={"kind": "int_array", "first_index": expected_first_index},
        baseline_system=(
            "Choose the best candidate functions for the task. "
            "Return ONLY a JSON array of integer indices ordered from best to worst."
        ),
        baseline_user=(
            f"Task: {statement}\n"
            f"Description: {informal_desc}\n"
            f"Candidates:\n{candidates_list}\n\n"
            "Return the JSON array of candidate indices:"
        ),
    )


def _reformulate_case(
    *,
    case_id: str,
    domain: str,
    predicate_id: str,
    statement: str,
    informal_desc: str,
    prover: str,
    queries_tried: Sequence[str],
    compiler_errors: str,
    required_terms: Sequence[str],
) -> PromptBenchmarkCase:
    return PromptBenchmarkCase(
        case_id=case_id,
        domain=domain,
        prompt_key=HUNTER_REFORMULATE,
        system=REFORMULATE_QUERY_SYSTEM,
        user=REFORMULATE_QUERY_USER.format(
            predicate_id=predicate_id,
            statement=statement,
            informal_desc=informal_desc,
            prover=prover,
            queries_tried="\n".join(f"- {q}" for q in queries_tried),
            compiler_errors=compiler_errors,
        ),
        grammar=_STRING_ARRAY_GBNF,
        expected={"kind": "string_array", "required_terms": list(required_terms)},
        baseline_system=(
            "Suggest better search queries for this failed retrieval attempt. "
            "Return ONLY a JSON array of strings."
        ),
        baseline_user=(
            f"Predicate ID: {predicate_id}\n"
            f"Task: {statement}\n"
            f"Description: {informal_desc}\n"
            f"Prover: {prover}\n"
            f"Queries tried:\n" + "\n".join(f"- {q}" for q in queries_tried) + "\n"
            f"Errors:\n{compiler_errors}\n\n"
            "Return the JSON array of better queries:"
        ),
    )


def _analyze_failure_case(
    *,
    case_id: str,
    domain: str,
    statement: str,
    candidate_name: str,
    candidate_type: str,
    compiler_output: str,
    target_terms: Sequence[str],
) -> PromptBenchmarkCase:
    return PromptBenchmarkCase(
        case_id=case_id,
        domain=domain,
        prompt_key=HUNTER_ANALYZE_FAILURE,
        system=ANALYZE_FAILURE_SYSTEM,
        user=ANALYZE_FAILURE_USER.format(
            statement=statement,
            candidate_name=candidate_name,
            candidate_type=candidate_type,
            compiler_output=compiler_output,
        ),
        use_grammar=False,
        expected={"kind": "failure_triplet", "target_terms": list(target_terms)},
        baseline_system=(
            "Briefly analyze the failed candidate. "
            "Return exactly three lines starting with CAUSE:, TARGET:, and NEXT:."
        ),
        baseline_user=(
            f"Task: {statement}\n"
            f"Candidate: {candidate_name}\n"
            f"Type: {candidate_type}\n"
            f"Compiler output:\n{compiler_output}"
        ),
    )


def default_prompt_benchmark_cases() -> list[PromptBenchmarkCase]:
    """Cross-domain prompt suite for hunter prompt-key comparisons."""
    return [
        _score_case(
            case_id="score_dsp_filter",
            domain="dsp",
            statement="Apply a stable bandpass filter to ECG samples.",
            informal_desc="select the filter primitive that directly filters the signal",
            candidates=[
                ("apply_iir_filter", "signal -> coeffs -> filtered_signal"),
                ("compute_frequency_response", "coeffs -> response"),
                ("plot_spectrum", "signal -> image"),
            ],
            expected_first_index=0,
        ),
        _score_case(
            case_id="score_graph_shortest_path",
            domain="graph",
            statement="Find shortest path distances from a source node.",
            informal_desc="single-source shortest path on weighted graph",
            candidates=[
                ("dijkstra", "graph -> source -> distances"),
                ("topological_sort", "graph -> ordering"),
                ("union_find_merge", "state -> edge -> state"),
            ],
            expected_first_index=0,
        ),
        _score_case(
            case_id="score_linear_algebra_solve",
            domain="linear_algebra",
            statement="Solve a symmetric positive definite linear system.",
            informal_desc="factor and solve Ax=b efficiently",
            candidates=[
                ("cholesky_solve", "matrix -> vector -> solution"),
                ("qr_decomposition", "matrix -> q_r"),
                ("normalize_vector", "vector -> vector"),
            ],
            expected_first_index=0,
        ),
        _score_case(
            case_id="score_string_lcs",
            domain="strings",
            statement="Compute the longest common subsequence of two strings.",
            informal_desc="dynamic programming over prefix pairs",
            candidates=[
                ("longest_common_subsequence", "string -> string -> string"),
                ("edit_distance", "string -> string -> nat"),
                ("kmp_search", "string -> pattern -> positions"),
            ],
            expected_first_index=0,
        ),
        _reformulate_case(
            case_id="reformulate_dsp_filter",
            domain="dsp",
            predicate_id="p_dsp",
            statement="Bandpass raw ECG into cardiac frequency region",
            informal_desc="stable digital filter design and application",
            prover="python",
            queries_tried=["ecg bandpass filter"],
            compiler_errors="Expected filtered_signal but got response tuple from compute_frequency_response",
            required_terms=["filter", "ecg", "bandpass"],
        ),
        _reformulate_case(
            case_id="reformulate_graph_shortest_path",
            domain="graph",
            predicate_id="p_graph",
            statement="Compute shortest path distances from source",
            informal_desc="weighted directed graph traversal",
            prover="python",
            queries_tried=["graph shortest path"],
            compiler_errors="Candidate topological_sort returns ordering, not distances",
            required_terms=["shortest", "distance", "dijkstra"],
        ),
        _reformulate_case(
            case_id="reformulate_linear_algebra_solve",
            domain="linear_algebra",
            predicate_id="p_la",
            statement="Solve SPD linear system",
            informal_desc="matrix factorization with triangular solves",
            prover="python",
            queries_tried=["solve symmetric positive definite"],
            compiler_errors="qr_decomposition does not return a solved vector",
            required_terms=["cholesky", "solve", "spd"],
        ),
        _reformulate_case(
            case_id="reformulate_string_lcs",
            domain="strings",
            predicate_id="p_str",
            statement="Find longest common subsequence",
            informal_desc="dynamic programming recurrence over strings",
            prover="python",
            queries_tried=["string subsequence"],
            compiler_errors="kmp_search finds pattern matches, not longest subsequence",
            required_terms=["longest common subsequence", "dynamic programming", "lcs"],
        ),
        _analyze_failure_case(
            case_id="analyze_dsp_filter",
            domain="dsp",
            statement="Apply a stable ECG filter",
            candidate_name="compute_frequency_response",
            candidate_type="coeffs -> response",
            compiler_output="Type mismatch: expected filtered_signal but got frequency_response",
            target_terms=["filter", "signal"],
        ),
        _analyze_failure_case(
            case_id="analyze_graph_shortest_path",
            domain="graph",
            statement="Find shortest path distances",
            candidate_name="topological_sort",
            candidate_type="graph -> ordering",
            compiler_output="Type mismatch: expected distance map but got ordering",
            target_terms=["distance", "path"],
        ),
        _analyze_failure_case(
            case_id="analyze_linear_algebra_solve",
            domain="linear_algebra",
            statement="Solve SPD linear system",
            candidate_name="qr_decomposition",
            candidate_type="matrix -> q_r",
            compiler_output="Expected solution vector but decomposition output lacks solve step",
            target_terms=["solve", "vector"],
        ),
        _analyze_failure_case(
            case_id="analyze_string_lcs",
            domain="strings",
            statement="Compute longest common subsequence",
            candidate_name="kmp_search",
            candidate_type="string -> pattern -> positions",
            compiler_output="Pattern match positions do not encode the longest common subsequence",
            target_terms=["subsequence", "dynamic"],
        ),
    ]


def select_prompt_benchmark_cases(
    *,
    prompt_keys: Sequence[str] | None = None,
) -> list[PromptBenchmarkCase]:
    cases = default_prompt_benchmark_cases()
    if not prompt_keys:
        return cases
    wanted = set(prompt_keys)
    return [case for case in cases if case.prompt_key in wanted]


def _validate_case_output(case: PromptBenchmarkCase, output: str) -> str:
    kind = str(case.expected.get("kind", "")).strip()
    if kind == "int_array":
        parsed = extract_json(output)
        if not isinstance(parsed, list) or not parsed:
            return "expected non-empty JSON integer array"
        if not all(isinstance(item, int) for item in parsed):
            return "expected all ranked indices to be integers"
        expected_first = int(case.expected["first_index"])
        if parsed[0] != expected_first:
            return f"expected first ranked index {expected_first}, got {parsed[0]}"
        return ""

    if kind == "string_array":
        parsed = extract_json(output)
        if not isinstance(parsed, list) or len(parsed) < 1:
            return "expected non-empty JSON string array"
        normalized = [str(item).strip().lower() for item in parsed if str(item).strip()]
        if not normalized:
            return "expected non-empty reformulation queries"
        required_terms = [term.lower() for term in case.expected.get("required_terms", [])]
        if required_terms and not any(
            any(term in query for term in required_terms) for query in normalized
        ):
            return f"expected at least one query containing one of {required_terms}"
        return ""

    if kind == "failure_triplet":
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if len(lines) != 3:
            return f"expected exactly 3 lines, got {len(lines)}"
        prefixes = ["CAUSE:", "TARGET:", "NEXT:"]
        for prefix, line in zip(prefixes, lines):
            if not line.startswith(prefix):
                return f"expected line starting with {prefix}"
        target_terms = [term.lower() for term in case.expected.get("target_terms", [])]
        target_line = lines[1].lower()
        if target_terms and not any(term in target_line for term in target_terms):
            return f"expected TARGET line to mention one of {target_terms}"
        return ""

    return f"unsupported benchmark expectation kind: {kind}"


async def run_prompt_benchmark(
    *,
    providers: Sequence[PromptBenchmarkProvider],
    cases: Sequence[PromptBenchmarkCase],
    repeats: int = 1,
    compare_direct_baseline: bool = False,
) -> list[PromptBenchmarkResult]:
    results: list[PromptBenchmarkResult] = []
    repeat_count = max(1, int(repeats))
    for provider in providers:
        model = str(getattr(provider.client, "_telemetry_model", "")) or str(
            getattr(provider.client, "_model", "")
        )
        for case in cases:
            variants = [("tuned", case.system, case.user)]
            if compare_direct_baseline:
                variants.append(
                    (
                        "direct_baseline",
                        case.baseline_system or case.system,
                        case.baseline_user or case.user,
                    )
                )
            for _ in range(repeat_count):
                for variant, system, user in variants:
                    started = time.perf_counter()
                    try:
                        if case.grammar and case.use_grammar and hasattr(
                            provider.client, "complete_with_grammar"
                        ):
                            output = await provider.client.complete_with_grammar(
                                system, user, case.grammar
                            )
                        else:
                            output = await provider.client.complete(system, user)
                        latency_ms = (time.perf_counter() - started) * 1000.0
                        validation_error = _validate_case_output(case, output)
                        results.append(
                            PromptBenchmarkResult(
                                case_id=case.case_id,
                                domain=case.domain,
                                prompt_key=case.prompt_key,
                                provider=provider.name,
                                model=model,
                                variant=variant,
                                latency_ms=latency_ms,
                                ok=validation_error == "",
                                validation_error=validation_error,
                                output_preview=output[:200],
                            )
                        )
                    except Exception as exc:
                        latency_ms = (time.perf_counter() - started) * 1000.0
                        results.append(
                            PromptBenchmarkResult(
                                case_id=case.case_id,
                                domain=case.domain,
                                prompt_key=case.prompt_key,
                                provider=provider.name,
                                model=model,
                                variant=variant,
                                latency_ms=latency_ms,
                                ok=False,
                                validation_error=str(exc),
                            )
                        )
    return results


def summarize_prompt_benchmark(
    results: Sequence[PromptBenchmarkResult],
) -> list[PromptBenchmarkAggregate]:
    aggregates: dict[tuple[str, str], PromptBenchmarkAggregate] = {}
    for result in results:
        key = (result.provider, result.model, result.variant)
        aggregate = aggregates.setdefault(
            key,
            PromptBenchmarkAggregate(
                provider=result.provider,
                model=result.model,
                variant=result.variant,
            ),
        )
        aggregate.record(result)
    return sorted(
        aggregates.values(),
        key=lambda item: (-item.passed_cases, item.avg_latency_ms, item.provider, item.variant),
    )


def format_prompt_benchmark_summary(
    aggregates: Sequence[PromptBenchmarkAggregate],
) -> str:
    lines = [
        "provider | variant | model | pass/total | avg ms | max ms",
        "--- | --- | --- | --- | ---: | ---:",
    ]
    for aggregate in aggregates:
        lines.append(
            f"{aggregate.provider} | {aggregate.variant} | {aggregate.model or '-'} | "
            f"{aggregate.passed_cases}/{aggregate.total_cases} | "
            f"{aggregate.avg_latency_ms:.1f} | {aggregate.max_latency_ms:.1f}"
        )
    return "\n".join(lines)


def save_prompt_benchmark_report(
    path: str | Path,
    *,
    results: Sequence[PromptBenchmarkResult],
    aggregates: Sequence[PromptBenchmarkAggregate],
) -> None:
    payload = {
        "results": [result.to_dict() for result in results],
        "aggregates": [aggregate.to_dict() for aggregate in aggregates],
    }
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
