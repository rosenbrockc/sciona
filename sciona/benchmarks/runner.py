"""Benchmark execution paths and runner."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any, Sequence

from sciona.architect.catalog import PrimitiveCatalog
from sciona.architect.graph import DecompositionAgent
from sciona.architect.models import AlgorithmicPrimitive, IOSpec
from sciona.benchmarks.cases import default_flow_benchmark_cases
from sciona.benchmarks.core import (
    FlowBenchmarkCase,
    FlowBenchmarkResult,
    _slug,
)
from sciona.benchmarks.fakes import (
    _BenchmarkHunterLLM,
    _EmptySkillIndex,
    _FailFirstHunterLLM,
    _FlowArchitectLLM,
    _LeafOracle,
    _LexicalSemanticIndex,
    _LLMFromScratchMock,
    _NoisyFlowArchitectLLM,
    _NoisyLLMFromScratchMock,
)
from sciona.hunter.graph import HunterAgent
from sciona.orchestrator import run_orchestration
from sciona.runtime_paths import _run_rapid_direct_match, _run_structured_single_pass
from sciona.services.hunter_service import HunterService
from sciona.services.orchestrator_service import OrchestratorService
from sciona.services.planner_service import SingleAgentPlanner
from sciona.services.skeleton_artifacts import build_local_skeleton_macro_retriever
from sciona.types import CandidateMatch, Declaration, PDGNode, Prover, VerificationLevel, VerificationResult


def _make_catalog(case: FlowBenchmarkCase) -> PrimitiveCatalog:
    catalog = PrimitiveCatalog()
    for leaf in case.leaves:
        catalog.add(
            AlgorithmicPrimitive(
                name=_slug(leaf.name),
                source="flow-benchmark",
                category=case.concept_type,
                description=leaf.description,
                inputs=[
                    IOSpec(name=name, type_desc=type_desc)
                    for name, type_desc in leaf.inputs
                ],
                outputs=[
                    IOSpec(name=name, type_desc=type_desc)
                    for name, type_desc in leaf.outputs
                ],
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
        oracle=_LeafOracle(
            {leaf.query_hint: leaf.declaration_name for leaf in case.leaves}
        ),  # type: ignore[arg-type]
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
        oracle=_LeafOracle(
            {leaf.query_hint: leaf.declaration_name for leaf in case.leaves}
        ),  # type: ignore[arg-type]
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
    *,
    noisy: bool = False,
    noise_seed: int | None = None,
) -> tuple[Any, _FlowArchitectLLM]:
    catalog = _make_catalog(case)
    architect_llm: _FlowArchitectLLM
    if noisy:
        architect_llm = _NoisyFlowArchitectLLM(case, seed=noise_seed)
    else:
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


async def _run_structured_case(
    case: FlowBenchmarkCase,
    *,
    noisy: bool = False,
    noise_seed: int | None = None,
) -> FlowBenchmarkResult:
    started = time.perf_counter()
    cdg, architect_llm = await _decompose_case(case, noisy=noisy, noise_seed=noise_seed)
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


async def _run_single_agent_case(
    case: FlowBenchmarkCase,
    *,
    noisy: bool = False,
    noise_seed: int | None = None,
) -> FlowBenchmarkResult:
    started = time.perf_counter()
    hunter, hunter_llm, index = _make_hunter(case)
    hunter_service = HunterService(hunter)
    architect_llm: _FlowArchitectLLM | None = None

    async def _architect_factory():
        nonlocal architect_llm
        cdg, architect_llm = await _decompose_case(
            case,
            noisy=noisy,
            noise_seed=noise_seed,
        )

        class _StaticArchitectService:
            async def decompose(self, request):
                return SimpleNamespace(goal=request.goal, cdg=cdg)

        return _StaticArchitectService()

    planner = SingleAgentPlanner(
        hunter=hunter_service,
        architect_factory=_architect_factory,
        orchestrator=OrchestratorService(hunter, run_orchestration),
        llm=object(),
        prover=Prover.PYTHON,
        max_rounds=2,
        hunter_concurrency=1,
        artifact_retriever=build_local_skeleton_macro_retriever(),
    )
    planner_result = await planner.run(case.prompt)
    result = planner_result.result
    matched = _matched_leaf_count(case, result.match_results)
    total = len(case.leaves)
    depth, leaf_count, edge_count = _cdg_metrics(result.cdg)
    latency_ms = (time.perf_counter() - started) * 1000.0
    architect_prompt_calls = 0 if architect_llm is None else architect_llm.calls
    return FlowBenchmarkResult(
        case_id=case.case_id,
        domain=case.domain,
        variant="single_agent",
        execution_path=planner_result.execution_path,
        ok=matched == total,
        latency_ms=latency_ms,
        prompt_calls=architect_prompt_calls + hunter_llm.calls,
        matched_leaves=matched,
        total_leaves=total,
        node_count=len(result.cdg.nodes),
        leaf_coverage=matched / max(1, total),
        best_similarity=_best_similarity_score(case, index),
        decomposition_depth=depth,
        decomposition_leaf_count=leaf_count,
        decomposition_edge_count=edge_count,
        planner_tool_dispatches=sum(
            int(metrics.get("dispatches", 0) or 0)
            for metrics in planner_result.state.tool_metrics.values()
        ),
        planner_tool_latency_ms=sum(
            float(metrics.get("latency_ms_total", 0.0) or 0.0)
            for metrics in planner_result.state.tool_metrics.values()
        ),
        planner_escalation_count=len(planner_result.state.escalation_events),
        planner_termination_reason=str(planner_result.state.termination_reason or ""),
        planner_action_signature=">".join(
            step.action for step in planner_result.steps if getattr(step, "action", "")
        ),
        planner_actions=tuple(
            step.action for step in planner_result.steps if getattr(step, "action", "")
        ),
        error="" if matched == total else "single-agent planner did not ground all leaves",
    )


async def _run_verified_case(
    case: FlowBenchmarkCase,
    *,
    noisy: bool = False,
    noise_seed: int | None = None,
) -> FlowBenchmarkResult:
    started = time.perf_counter()
    cdg, architect_llm = await _decompose_case(case, noisy=noisy, noise_seed=noise_seed)
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


def _make_fail_first_hunter(
    case: FlowBenchmarkCase,
) -> tuple[HunterAgent, _FailFirstHunterLLM, _LexicalSemanticIndex]:
    declarations = _make_declarations(case)
    index = _LexicalSemanticIndex(declarations)
    hunter_llm = _FailFirstHunterLLM()
    hunter = HunterAgent(
        index=index,  # type: ignore[arg-type]
        oracle=_LeafOracle(
            {leaf.query_hint: leaf.declaration_name for leaf in case.leaves}
        ),  # type: ignore[arg-type]
        llm=hunter_llm,
        max_iterations=3,
        top_k_verify=2,
        search_k=5,
    )
    return hunter, hunter_llm, index


async def _run_verified_refinement_case(
    case: FlowBenchmarkCase,
) -> FlowBenchmarkResult:
    """Verified variant using a fail-first Hunter to exercise the refinement loop."""
    started = time.perf_counter()
    cdg, architect_llm = await _decompose_case(case)
    hunter, hunter_llm, index = _make_fail_first_hunter(case)
    result = await run_orchestration(
        cdg,
        hunter_agent=hunter,
        llm=architect_llm,  # type: ignore[arg-type]
        prover=Prover.PYTHON,
        max_rounds=3,
        hunter_concurrency=1,
    )
    matched = _matched_leaf_count(case, result.match_results)
    total = len(case.leaves)
    depth, leaf_count, edge_count = _cdg_metrics(result.cdg)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return FlowBenchmarkResult(
        case_id=case.case_id,
        domain=case.domain,
        variant="verified_refinement",
        execution_path="verified_orchestration_refinement",
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
        error="" if matched == total else "refinement did not recover all leaves",
    )


async def _run_llm_from_scratch_case(
    case: FlowBenchmarkCase,
    *,
    noisy: bool = False,
    noise_seed: int | None = None,
) -> FlowBenchmarkResult:
    started = time.perf_counter()
    llm: _LLMFromScratchMock
    if noisy:
        llm = _NoisyLLMFromScratchMock(case, seed=noise_seed)
    else:
        llm = _LLMFromScratchMock(case)
    raw = await llm.identify(case.prompt)
    parsed = json.loads(raw)
    expected = {leaf.declaration_name for leaf in case.leaves}
    matched: set[str] = set()
    for entry in parsed:
        name = str(entry.get("name", ""))
        if name in expected:
            matched.add(name)
    latency_ms = (time.perf_counter() - started) * 1000.0
    total = len(case.leaves)
    n_matched = len(matched)
    return FlowBenchmarkResult(
        case_id=case.case_id,
        domain=case.domain,
        variant="llm_from_scratch",
        execution_path="llm_from_scratch",
        ok=n_matched == total,
        latency_ms=latency_ms,
        prompt_calls=llm.calls,
        matched_leaves=n_matched,
        total_leaves=total,
        node_count=0,
        leaf_coverage=n_matched / max(1, total),
        best_similarity=0.0,
        decomposition_depth=0,
        decomposition_leaf_count=0,
        decomposition_edge_count=0,
    )


async def run_flow_benchmark(
    *,
    cases: Sequence[FlowBenchmarkCase],
    variants: Sequence[str] = (
        "direct_baseline",
        "rapid",
        "single_agent",
        "structured",
        "verified",
    ),
    repeats: int = 1,
    noisy: bool = False,
) -> list[FlowBenchmarkResult]:
    results: list[FlowBenchmarkResult] = []
    repeat_count = max(1, int(repeats))
    for case in cases:
        for repeat_idx in range(repeat_count):
            noise_seed = repeat_idx if noisy else None
            for variant in variants:
                if variant == "direct_baseline":
                    results.append(await _run_direct_baseline_case(case))
                    continue
                if variant == "rapid":
                    results.append(await _run_rapid_case(case))
                    continue
                if variant == "single_agent":
                    results.append(
                        await _run_single_agent_case(
                            case, noisy=noisy, noise_seed=noise_seed
                        )
                    )
                    continue
                if variant == "structured":
                    results.append(
                        await _run_structured_case(
                            case, noisy=noisy, noise_seed=noise_seed
                        )
                    )
                    continue
                if variant == "verified":
                    results.append(
                        await _run_verified_case(
                            case, noisy=noisy, noise_seed=noise_seed
                        )
                    )
                    continue
                if variant == "verified_refinement":
                    results.append(await _run_verified_refinement_case(case))
                    continue
                if variant == "llm_from_scratch":
                    results.append(
                        await _run_llm_from_scratch_case(
                            case, noisy=noisy, noise_seed=noise_seed
                        )
                    )
                    continue
                raise ValueError(f"Unsupported flow benchmark variant: {variant}")
    return results
