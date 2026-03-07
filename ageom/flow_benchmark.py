"""Small full-flow benchmark harness for direct vs structured algorithm-generation paths."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from ageom.architect.graph import DecompositionAgent
from ageom.architect.handoff import to_pdg_nodes
from ageom.architect.catalog import PrimitiveCatalog
from ageom.architect.models import AlgorithmicPrimitive, ConceptType, IOSpec
from ageom.hunter.graph import HunterAgent
from ageom.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
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
    ok: bool
    latency_ms: float
    matched_leaves: int
    total_leaves: int
    node_count: int
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FlowBenchmarkAggregate:
    variant: str
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    avg_latency_ms: float = 0.0

    def record(self, result: FlowBenchmarkResult) -> None:
        self.total_cases += 1
        if result.ok:
            self.passed_cases += 1
        else:
            self.failed_cases += 1
        total_latency = self.avg_latency_ms * (self.total_cases - 1)
        total_latency += result.latency_ms
        self.avg_latency_ms = total_latency / max(1, self.total_cases)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    async def complete(self, system: str, user: str) -> str:
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
    async def complete(self, system: str, user: str) -> str:
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
    hunter = HunterAgent(
        index=_LexicalSemanticIndex(declarations),  # type: ignore[arg-type]
        oracle=_LeafOracle({leaf.query_hint: leaf.declaration_name for leaf in case.leaves}),  # type: ignore[arg-type]
        llm=_BenchmarkHunterLLM(),
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
    latency_ms = (time.perf_counter() - started) * 1000.0
    return FlowBenchmarkResult(
        case_id=case.case_id,
        domain=case.domain,
        variant="direct_baseline",
        ok=matched == len(case.leaves),
        latency_ms=latency_ms,
        matched_leaves=matched,
        total_leaves=len(case.leaves),
        node_count=1,
        error="" if matched == len(case.leaves) else "single-shot retrieval did not cover all leaves",
    )


async def _run_structured_case(
    case: FlowBenchmarkCase,
    *,
    variant: str,
) -> FlowBenchmarkResult:
    started = time.perf_counter()
    catalog = _make_catalog(case)
    agent = DecompositionAgent(
        catalog=catalog,
        skill_index=_EmptySkillIndex(),  # type: ignore[arg-type]
        llm=_FlowArchitectLLM(case),  # type: ignore[arg-type]
        max_depth=6,
    )
    cdg = await agent.decompose(case.prompt)
    pdg_nodes = to_pdg_nodes(cdg, prover=Prover.PYTHON, strict=False)
    declarations = _make_declarations(case)
    hunter = HunterAgent(
        index=_LexicalSemanticIndex(declarations),  # type: ignore[arg-type]
        oracle=_LeafOracle({leaf.query_hint: leaf.declaration_name for leaf in case.leaves}),  # type: ignore[arg-type]
        llm=_BenchmarkHunterLLM(),
        max_iterations=2,
        top_k_verify=2,
        search_k=5,
    )

    matched = 0
    for leaf in case.leaves:
        leaf_slug = _slug(leaf.name)
        matching_nodes = [
            node
            for node in pdg_nodes
            if _hint_matches(
                f"{node.statement} {node.informal_desc} {node.context.get('matched_primitive', '')}",
                leaf.query_hint,
            )
        ]
        matching_nodes.sort(
            key=lambda node: (
                0
                if node.context.get("matched_primitive", "") == leaf_slug
                else 1,
                int(node.context.get("depth", "0")),
            )
        )
        query_node = matching_nodes[0] if matching_nodes else None
        if query_node is None:
            continue
        result = await hunter.find_match(query_node)
        if result.success:
            matched += 1

    latency_ms = (time.perf_counter() - started) * 1000.0
    return FlowBenchmarkResult(
        case_id=case.case_id,
        domain=case.domain,
        variant=variant,
        ok=matched == len(case.leaves),
        latency_ms=latency_ms,
        matched_leaves=matched,
        total_leaves=len(case.leaves),
        node_count=len(cdg.nodes),
        error="" if matched == len(case.leaves) else "not all decomposed leaves matched",
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
) -> list[FlowBenchmarkResult]:
    results: list[FlowBenchmarkResult] = []
    for case in cases:
        for variant in variants:
            if variant == "direct_baseline":
                results.append(await _run_direct_baseline_case(case))
                continue
            results.append(await _run_structured_case(case, variant=variant))
    return results


def summarize_flow_benchmark(
    results: Sequence[FlowBenchmarkResult],
) -> list[FlowBenchmarkAggregate]:
    aggregates: dict[str, FlowBenchmarkAggregate] = {}
    for result in results:
        aggregate = aggregates.setdefault(result.variant, FlowBenchmarkAggregate(variant=result.variant))
        aggregate.record(result)
    return sorted(
        aggregates.values(),
        key=lambda item: (-item.passed_cases, item.avg_latency_ms, item.variant),
    )


def format_flow_benchmark_summary(
    aggregates: Sequence[FlowBenchmarkAggregate],
) -> str:
    lines = [
        "variant | pass/total | avg ms",
        "--- | --- | ---:",
    ]
    for aggregate in aggregates:
        lines.append(
            f"{aggregate.variant} | {aggregate.passed_cases}/{aggregate.total_cases} | {aggregate.avg_latency_ms:.1f}"
        )
    return "\n".join(lines)
