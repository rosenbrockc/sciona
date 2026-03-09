"""Small full-flow benchmark harness for direct vs structured algorithm-generation paths."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ageom.architect.graph import DecompositionAgent
from ageom.architect.catalog import PrimitiveCatalog
from ageom.architect.models import AlgorithmicPrimitive, ConceptType, IOSpec
from ageom.hunter.graph import HunterAgent
from ageom.orchestrator import run_orchestration
from ageom.types import (
    CandidateMatch,
    Declaration,
    PDGNode,
    Prover,
    VerificationLevel,
    VerificationResult,
)


@dataclass(frozen=True)
class FlowLeafSpec:
    name: str
    description: str
    type_signature: str
    query_hint: str
    declaration_name: str
    inputs: tuple[tuple[str, str], ...]
    outputs: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class FlowBenchmarkCase:
    case_id: str
    domain: str
    prompt: str
    concept_type: ConceptType
    leaves: tuple[FlowLeafSpec, ...]


@dataclass
class FlowBenchmarkResult:
    case_id: str
    domain: str
    variant: str
    execution_path: str
    ok: bool
    latency_ms: float
    prompt_calls: int
    matched_leaves: int
    total_leaves: int
    node_count: int
    leaf_coverage: float = 0.0
    best_similarity: float = 0.0
    decomposition_depth: int = 0
    decomposition_leaf_count: int = 0
    decomposition_edge_count: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FlowBenchmarkAggregate:
    variant: str
    execution_paths: list[str] = field(default_factory=list)
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    avg_latency_ms: float = 0.0
    total_prompt_calls: int = 0
    avg_prompt_calls: float = 0.0
    avg_leaf_coverage: float = 0.0
    avg_best_similarity: float = 0.0
    repeat_groups: int = 0
    stable_groups: int = 0
    stability_rate: float = 1.0
    _case_outcomes: dict[str, list[bool]] = field(default_factory=dict)

    def record(self, result: FlowBenchmarkResult) -> None:
        self.total_cases += 1
        if result.execution_path not in self.execution_paths:
            self.execution_paths.append(result.execution_path)
        if result.ok:
            self.passed_cases += 1
        else:
            self.failed_cases += 1
        total_latency = self.avg_latency_ms * (self.total_cases - 1)
        total_latency += result.latency_ms
        self.avg_latency_ms = total_latency / max(1, self.total_cases)
        self.total_prompt_calls += int(result.prompt_calls)
        self.avg_prompt_calls = self.total_prompt_calls / max(1, self.total_cases)
        prev = self.total_cases - 1
        self.avg_leaf_coverage = (
            self.avg_leaf_coverage * prev + result.leaf_coverage
        ) / self.total_cases
        self.avg_best_similarity = (
            self.avg_best_similarity * prev + result.best_similarity
        ) / self.total_cases
        self._case_outcomes.setdefault(result.case_id, []).append(result.ok)

    def finalize(self) -> None:
        groups = list(self._case_outcomes.values())
        self.repeat_groups = len(groups)
        self.stable_groups = sum(
            1 for outcomes in groups if len(set(outcomes)) <= 1
        )
        self.stability_rate = (
            self.stable_groups / self.repeat_groups if self.repeat_groups else 1.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "execution_paths": sorted(self.execution_paths),
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "avg_latency_ms": self.avg_latency_ms,
            "total_prompt_calls": self.total_prompt_calls,
            "avg_prompt_calls": self.avg_prompt_calls,
            "avg_leaf_coverage": self.avg_leaf_coverage,
            "avg_best_similarity": self.avg_best_similarity,
            "repeat_groups": self.repeat_groups,
            "stable_groups": self.stable_groups,
            "stability_rate": self.stability_rate,
        }


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _hint_matches(text: str, hint: str) -> bool:
    text_tokens = {token for token in _slug(text).split("_") if token}
    hint_tokens = {token for token in _slug(hint).split("_") if token}
    if not text_tokens or not hint_tokens:
        return False
    return hint_tokens <= text_tokens or len(text_tokens & hint_tokens) >= min(2, len(hint_tokens))


class _FlowArchitectLLM:
    def __init__(self, case: FlowBenchmarkCase) -> None:
        self._case = case
        self.calls = 0

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        system_lower = system.lower()
        if "best" in system_lower and "paradigm" in system_lower:
            return json.dumps(
                {
                    "paradigm": self._case.concept_type.value,
                    "rationale": f"{self._case.case_id} fits this paradigm",
                    "variant_hint": self._case.case_id,
                }
            )
        if "sub-nodes" in system_lower or "sub_nodes" in system_lower:
            return json.dumps(
                {
                    "sub_nodes": [
                        {
                            "name": leaf.name,
                            "description": leaf.description,
                            "concept_type": self._case.concept_type.value,
                            "inputs": [
                                {"name": name, "type_desc": type_desc}
                                for name, type_desc in leaf.inputs
                            ],
                            "outputs": [
                                {"name": name, "type_desc": type_desc}
                                for name, type_desc in leaf.outputs
                            ],
                            "type_signature": leaf.type_signature,
                            "is_atomic": True,
                            "matched_primitive": _slug(leaf.name),
                        }
                        for leaf in self._case.leaves
                    ],
                    "edges": [],
                }
            )
        if "critic" in system_lower or "evaluate" in system_lower:
            return json.dumps(
                {
                    "approved": True,
                    "reason": "Valid decomposition",
                    "io_issues": [],
                    "flagged_nodes": [],
                }
            )
        return "{}"


class _BenchmarkHunterLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        lower = system.lower()
        if "rank" in lower or "score" in lower:
            return "[0, 1, 2, 3]"
        if "reformulate" in lower:
            return '["exact function name", "typed primitive query"]'
        if "analy" in lower:
            return "CAUSE: broad query\nTARGET: exact primitive\nNEXT: search by primitive name"
        return "[]"

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)


class _LexicalSemanticIndex:
    def __init__(self, declarations: list[Declaration]) -> None:
        self._declarations = declarations

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token for token in _slug(text).split("_") if token}

    def _score(self, query_text: str, decl: Declaration) -> float:
        query_tokens = self._tokens(query_text)
        decl_tokens = self._tokens(
            f"{decl.name} {decl.type_signature} {decl.docstring} {decl.conceptual_summary}"
        )
        return float(len(query_tokens & decl_tokens))

    def search_by_embedding(self, query_text: str, k: int = 10):
        ranked = sorted(
            ((decl, self._score(query_text, decl)) for decl in self._declarations),
            key=lambda item: (item[1], item[0].name),
            reverse=True,
        )
        return ranked[:k]

    def search_by_type(self, type_signature: str, k: int = 10):
        return [decl for decl, _score in self.search_by_embedding(type_signature, k=k)]


class _LeafOracle:
    def __init__(self, expected_by_query_hint: dict[str, str]) -> None:
        self._expected_by_query_hint = expected_by_query_hint

    async def verify_candidate(
        self, pdg_node: PDGNode, candidate: CandidateMatch
    ) -> VerificationResult:
        expected = ""
        node_text = f"{pdg_node.statement} {pdg_node.informal_desc}".lower()
        for hint, name in self._expected_by_query_hint.items():
            if _hint_matches(node_text, hint):
                expected = name
                break
        verified = candidate.declaration.name == expected and expected != ""
        return VerificationResult(
            candidate=candidate,
            verified=verified,
            compiler_output="ok" if verified else "type mismatch",
            proof_term=candidate.declaration.name if verified else "",
            error_message="" if verified else "type mismatch",
            verification_level=(
                VerificationLevel.TYPE_CHECKED
                if verified
                else VerificationLevel.UNVERIFIED
            ),
        )

    async def verify_candidates(
        self, pdg_node: PDGNode, candidates: list[CandidateMatch]
    ) -> list[VerificationResult]:
        results: list[VerificationResult] = []
        for candidate in candidates:
            result = await self.verify_candidate(pdg_node, candidate)
            results.append(result)
            if result.verified:
                break
        return results


class _EmptySkillIndex:
    def search(self, query: str, k: int = 10):
        return []


def _make_catalog(case: FlowBenchmarkCase) -> PrimitiveCatalog:
    catalog = PrimitiveCatalog()
    for leaf in case.leaves:
        catalog.add(
            AlgorithmicPrimitive(
                name=_slug(leaf.name),
                source="flow-benchmark",
                category=case.concept_type,
                description=leaf.description,
                inputs=[IOSpec(name=name, type_desc=type_desc) for name, type_desc in leaf.inputs],
                outputs=[IOSpec(name=name, type_desc=type_desc) for name, type_desc in leaf.outputs],
                type_signature=leaf.type_signature,
            )
        )
    return catalog


def _make_declarations(case: FlowBenchmarkCase) -> list[Declaration]:
    declarations: list[Declaration] = []
    for leaf in case.leaves:
        declarations.append(
            Declaration(
                name=leaf.declaration_name,
                type_signature=leaf.type_signature,
                docstring=leaf.description,
                conceptual_summary=leaf.query_hint,
                prover=Prover.PYTHON,
            )
        )
    declarations.extend(
        [
            Declaration(
                name="distractor.unrelated_one",
                type_signature="noise -> output",
                docstring="irrelevant",
                prover=Prover.PYTHON,
            ),
            Declaration(
                name="distractor.unrelated_two",
                type_signature="noise -> output",
                docstring="irrelevant",
                prover=Prover.PYTHON,
            ),
        ]
    )
    return declarations


async def _run_direct_baseline_case(case: FlowBenchmarkCase) -> FlowBenchmarkResult:
    started = time.perf_counter()
    declarations = _make_declarations(case)
    index = _LexicalSemanticIndex(declarations)
    hunter_llm = _BenchmarkHunterLLM()
    hunter = HunterAgent(
        index=index,  # type: ignore[arg-type]
        oracle=_LeafOracle({leaf.query_hint: leaf.declaration_name for leaf in case.leaves}),  # type: ignore[arg-type]
        llm=hunter_llm,
        max_iterations=1,
        top_k_verify=1,
        search_k=5,
    )
    result = await hunter.find_match(
        PDGNode(
            predicate_id=f"{case.case_id}-direct",
            statement=case.prompt,
            informal_desc="direct baseline without decomposition",
            prover=Prover.PYTHON,
        )
    )
    matched = 1 if result.success else 0
    total = len(case.leaves)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return FlowBenchmarkResult(
        case_id=case.case_id,
        domain=case.domain,
        variant="direct_baseline",
        execution_path="direct_baseline",
        ok=matched == total,
        latency_ms=latency_ms,
        prompt_calls=hunter_llm.calls,
        matched_leaves=matched,
        total_leaves=total,
        node_count=1,
        leaf_coverage=matched / max(1, total),
        best_similarity=_best_similarity_score(case, index),
        decomposition_depth=1,
        decomposition_leaf_count=1,
        decomposition_edge_count=0,
        error="" if matched == total else "single-shot retrieval did not cover all leaves",
    )


def _make_hunter(
    case: FlowBenchmarkCase,
) -> tuple[HunterAgent, _BenchmarkHunterLLM, _LexicalSemanticIndex]:
    declarations = _make_declarations(case)
    index = _LexicalSemanticIndex(declarations)
    hunter_llm = _BenchmarkHunterLLM()
    hunter = HunterAgent(
        index=index,  # type: ignore[arg-type]
        oracle=_LeafOracle({leaf.query_hint: leaf.declaration_name for leaf in case.leaves}),  # type: ignore[arg-type]
        llm=hunter_llm,
        max_iterations=2,
        top_k_verify=2,
        search_k=5,
    )
    return hunter, hunter_llm, index


def _matched_leaf_count(
    case: FlowBenchmarkCase,
    match_results: Sequence[Any],
) -> int:
    """Count unique expected leaf declarations covered by successful matches."""
    expected = {leaf.declaration_name for leaf in case.leaves}
    matched: set[str] = set()
    for result in match_results:
        verified = getattr(result, "verified_match", None)
        declaration = getattr(getattr(verified, "candidate", None), "declaration", None)
        name = str(getattr(declaration, "name", "") or "").strip()
        if name in expected:
            matched.add(name)
    return len(matched)


def _best_similarity_score(
    case: FlowBenchmarkCase,
    index: _LexicalSemanticIndex,
) -> float:
    """Max token-overlap score between any leaf query_hint and any declaration."""
    best = 0.0
    for leaf in case.leaves:
        ranked = index.search_by_embedding(leaf.query_hint, k=1)
        for _decl, score in ranked:
            best = max(best, score)
    return best


def _cdg_metrics(cdg: Any) -> tuple[int, int, int]:
    """Return (depth, leaf_count, edge_count) for a CDG."""
    leaf_count = len(cdg.leaf_nodes()) if hasattr(cdg, "leaf_nodes") else 0
    edge_count = len(cdg.edges) if hasattr(cdg, "edges") else 0
    children: dict[str, list[str]] = {}
    for edge in getattr(cdg, "edges", []):
        children.setdefault(edge.source_id, []).append(edge.target_id)
    node_ids = {n.node_id for n in cdg.nodes} if hasattr(cdg, "nodes") else set()
    child_ids = {edge.target_id for edge in getattr(cdg, "edges", [])}
    roots = list((node_ids - child_ids) or node_ids)
    depth = 0
    frontier = roots
    while frontier:
        depth += 1
        next_frontier: list[str] = []
        for nid in frontier:
            next_frontier.extend(children.get(nid, []))
        frontier = next_frontier
    return depth, leaf_count, edge_count


async def _decompose_case(
    case: FlowBenchmarkCase,
) -> tuple[Any, _FlowArchitectLLM]:
    catalog = _make_catalog(case)
    architect_llm = _FlowArchitectLLM(case)
    agent = DecompositionAgent(
        catalog=catalog,
        skill_index=_EmptySkillIndex(),  # type: ignore[arg-type]
        llm=architect_llm,  # type: ignore[arg-type]
        max_depth=6,
    )
    cdg = await agent.decompose(case.prompt)
    return cdg, architect_llm


async def _run_rapid_case(case: FlowBenchmarkCase) -> FlowBenchmarkResult:
    from ageom.cli import _run_rapid_direct_match

    started = time.perf_counter()
    hunter, hunter_llm, index = _make_hunter(case)
    result = await _run_rapid_direct_match(
        case.prompt,
        prover=Prover.PYTHON,
        hunter=hunter,
    )
    matched = _matched_leaf_count(case, result.match_results)
    total = len(case.leaves)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return FlowBenchmarkResult(
        case_id=case.case_id,
        domain=case.domain,
        variant="rapid",
        execution_path="rapid_direct",
        ok=matched == total,
        latency_ms=latency_ms,
        prompt_calls=hunter_llm.calls,
        matched_leaves=matched,
        total_leaves=total,
        node_count=len(result.cdg.nodes),
        leaf_coverage=matched / max(1, total),
        best_similarity=_best_similarity_score(case, index),
        decomposition_depth=1,
        decomposition_leaf_count=1,
        decomposition_edge_count=0,
        error="" if matched == total else "rapid direct match did not cover all leaves",
    )


async def _run_structured_case(case: FlowBenchmarkCase) -> FlowBenchmarkResult:
    from ageom.cli import _run_structured_single_pass

    started = time.perf_counter()
    cdg, architect_llm = await _decompose_case(case)
    hunter, hunter_llm, index = _make_hunter(case)
    result = await _run_structured_single_pass(
        cdg,
        prover=Prover.PYTHON,
        hunter=hunter,
    )
    matched = _matched_leaf_count(case, result.match_results)
    total = len(case.leaves)
    depth, leaf_count, edge_count = _cdg_metrics(cdg)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return FlowBenchmarkResult(
        case_id=case.case_id,
        domain=case.domain,
        variant="structured",
        execution_path="structured_single_pass",
        ok=matched == total,
        latency_ms=latency_ms,
        prompt_calls=architect_llm.calls + hunter_llm.calls,
        matched_leaves=matched,
        total_leaves=total,
        node_count=len(cdg.nodes),
        leaf_coverage=matched / max(1, total),
        best_similarity=_best_similarity_score(case, index),
        decomposition_depth=depth,
        decomposition_leaf_count=leaf_count,
        decomposition_edge_count=edge_count,
        error="" if matched == total else "not all decomposed leaves matched",
    )


async def _run_verified_case(case: FlowBenchmarkCase) -> FlowBenchmarkResult:
    started = time.perf_counter()
    cdg, architect_llm = await _decompose_case(case)
    hunter, hunter_llm, index = _make_hunter(case)
    result = await run_orchestration(
        cdg,
        hunter_agent=hunter,
        llm=architect_llm,  # type: ignore[arg-type]
        prover=Prover.PYTHON,
        max_rounds=2,
        hunter_concurrency=1,
    )
    matched = _matched_leaf_count(case, result.match_results)
    total = len(case.leaves)
    depth, leaf_count, edge_count = _cdg_metrics(result.cdg)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return FlowBenchmarkResult(
        case_id=case.case_id,
        domain=case.domain,
        variant="verified",
        execution_path="verified_orchestration",
        ok=matched == total,
        latency_ms=latency_ms,
        prompt_calls=architect_llm.calls + hunter_llm.calls,
        matched_leaves=matched,
        total_leaves=total,
        node_count=len(result.cdg.nodes),
        leaf_coverage=matched / max(1, total),
        best_similarity=_best_similarity_score(case, index),
        decomposition_depth=depth,
        decomposition_leaf_count=leaf_count,
        decomposition_edge_count=edge_count,
        error="" if matched == total else "verified refinement did not ground all leaves",
    )


def default_flow_benchmark_cases() -> list[FlowBenchmarkCase]:
    return [
        FlowBenchmarkCase(
            case_id="sorting_merge",
            domain="sorting",
            prompt="Sort a list by splitting it into halves, sorting each half, and merging the sorted halves.",
            concept_type=ConceptType.SORTING,
            leaves=(
                FlowLeafSpec(
                    name="Split List",
                    description="Split a list into left and right halves.",
                    type_signature="list[int] -> tuple[list[int], list[int]]",
                    query_hint="Split List",
                    declaration_name="algorithms.split_list_halves",
                    inputs=(("items", "list[int]"),),
                    outputs=(("left", "list[int]"), ("right", "list[int]")),
                ),
                FlowLeafSpec(
                    name="Merge Sorted Halves",
                    description="Merge two sorted halves into one sorted list.",
                    type_signature="list[int] -> list[int] -> list[int]",
                    query_hint="Merge Sorted Halves",
                    declaration_name="algorithms.merge_sorted_halves",
                    inputs=(("left", "list[int]"), ("right", "list[int]")),
                    outputs=(("merged", "list[int]"),),
                ),
            ),
        ),
        FlowBenchmarkCase(
            case_id="graph_shortest_path",
            domain="graph",
            prompt="Compute shortest path distances from a source node in a weighted graph.",
            concept_type=ConceptType.GRAPH_OPTIMIZATION,
            leaves=(
                FlowLeafSpec(
                    name="Initialize Distances",
                    description="Initialize distance map for a weighted graph shortest path routine.",
                    type_signature="graph -> source -> dict[node, float]",
                    query_hint="Initialize Distance",
                    declaration_name="algorithms.initialize_distances",
                    inputs=(("graph", "Graph"), ("source", "Node")),
                    outputs=(("distances", "dict[node, float]"),),
                ),
                FlowLeafSpec(
                    name="Relax Edges",
                    description="Relax weighted edges to improve tentative shortest path distances.",
                    type_signature="graph -> dict[node, float] -> dict[node, float]",
                    query_hint="Relax Edges",
                    declaration_name="algorithms.relax_edges",
                    inputs=(("graph", "Graph"), ("distances", "dict[node, float]")),
                    outputs=(("updated", "dict[node, float]"),),
                ),
            ),
        ),
        FlowBenchmarkCase(
            case_id="dsp_bandpass_filter",
            domain="dsp",
            prompt="Design and apply a stable bandpass filter to ECG samples.",
            concept_type=ConceptType.SIGNAL_FILTER,
            leaves=(
                FlowLeafSpec(
                    name="Design Filter",
                    description="Design stable bandpass filter coefficients for ECG samples.",
                    type_signature="spec -> coefficients",
                    query_hint="Design Filter",
                    declaration_name="algorithms.design_bandpass_filter",
                    inputs=(("spec", "FilterSpec"),),
                    outputs=(("coefficients", "Coefficients"),),
                ),
                FlowLeafSpec(
                    name="Apply Filter",
                    description="Apply stable bandpass filter coefficients to ECG samples.",
                    type_signature="signal -> coefficients -> signal",
                    query_hint="Apply Filter",
                    declaration_name="algorithms.apply_bandpass_filter",
                    inputs=(("signal", "np.ndarray"), ("coefficients", "Coefficients")),
                    outputs=(("filtered_signal", "np.ndarray"),),
                ),
            ),
        ),
    ]


async def run_flow_benchmark(
    *,
    cases: Sequence[FlowBenchmarkCase],
    variants: Sequence[str] = ("direct_baseline", "rapid", "structured", "verified"),
    repeats: int = 1,
) -> list[FlowBenchmarkResult]:
    results: list[FlowBenchmarkResult] = []
    repeat_count = max(1, int(repeats))
    for case in cases:
        for _ in range(repeat_count):
            for variant in variants:
                if variant == "direct_baseline":
                    results.append(await _run_direct_baseline_case(case))
                    continue
                if variant == "rapid":
                    results.append(await _run_rapid_case(case))
                    continue
                if variant == "structured":
                    results.append(await _run_structured_case(case))
                    continue
                if variant == "verified":
                    results.append(await _run_verified_case(case))
                    continue
                raise ValueError(f"Unsupported flow benchmark variant: {variant}")
    return results


def summarize_flow_benchmark(
    results: Sequence[FlowBenchmarkResult],
) -> list[FlowBenchmarkAggregate]:
    aggregates: dict[str, FlowBenchmarkAggregate] = {}
    for result in results:
        aggregate = aggregates.setdefault(result.variant, FlowBenchmarkAggregate(variant=result.variant))
        aggregate.record(result)
    for aggregate in aggregates.values():
        aggregate.finalize()
    return sorted(
        aggregates.values(),
        key=lambda item: (-item.passed_cases, -item.stability_rate, item.avg_latency_ms, item.variant),
    )


def format_flow_benchmark_summary(
    aggregates: Sequence[FlowBenchmarkAggregate],
) -> str:
    lines = [
        "variant | paths | pass/total | stable | avg ms | avg prompts",
        "--- | --- | --- | ---: | ---: | ---:",
    ]
    for aggregate in aggregates:
        paths = ",".join(aggregate.execution_paths) or "--"
        lines.append(
            f"{aggregate.variant} | {paths} | {aggregate.passed_cases}/{aggregate.total_cases} | "
            f"{aggregate.stable_groups}/{aggregate.repeat_groups} | {aggregate.avg_latency_ms:.1f} | "
            f"{aggregate.avg_prompt_calls:.1f}"
        )
    return "\n".join(lines)


def save_flow_benchmark_report(
    path: str | Path,
    *,
    results: Sequence[FlowBenchmarkResult],
    aggregates: Sequence[FlowBenchmarkAggregate],
) -> None:
    payload = {
        "results": [result.to_dict() for result in results],
        "aggregates": [aggregate.to_dict() for aggregate in aggregates],
    }
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
