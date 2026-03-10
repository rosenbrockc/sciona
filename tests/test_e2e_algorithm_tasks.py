"""End-to-end algorithm task tests across Round 1 (decompose) and Round 2 (match)."""

from __future__ import annotations

import json
import importlib.util
import re
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

if (
    importlib.util.find_spec("langgraph") is None
    or importlib.util.find_spec("pydantic_graph") is None
):
    pytest.skip(
        "requires optional extras: langgraph and pydantic-graph",
        allow_module_level=True,
    )

from ageom.architect.catalog import PrimitiveCatalog
from ageom.architect.graph import DecompositionAgent
from ageom.architect.handoff import to_pdg_nodes
from ageom.architect.models import (
    AlgorithmicPrimitive,
    ConceptType,
    IOSpec,
)
from ageom.hunter.graph import HunterAgent
from ageom.types import (
    Declaration,
    Prover,
    VerificationResult,
)


@dataclass(frozen=True)
class SubNodeSpec:
    name: str
    description: str
    type_signature: str
    inputs: tuple[tuple[str, str], ...]
    outputs: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class EdgeSpec:
    source_name: str
    target_name: str
    output_name: str
    input_name: str
    data_type: str


@dataclass(frozen=True)
class Round2Check:
    query_hint: str
    expected_declaration: str


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    prompt: str
    sample_input: str
    sample_output: str
    paradigm: ConceptType
    sub_nodes: tuple[SubNodeSpec, ...]
    edges: tuple[EdgeSpec, ...]
    required_round1_nodes: tuple[str, ...]
    round2_checks: tuple[Round2Check, ...]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


class TaskAwareArchitectLLM:
    def __init__(self, task: TaskSpec) -> None:
        self._task = task

    async def complete(self, system: str, user: str) -> str:
        system_lower = system.lower()
        if "best" in system_lower and "paradigm" in system_lower:
            return json.dumps(
                {
                    "paradigm": ConceptType.CUSTOM.value,
                    "rationale": f"{self._task.task_id} is mocked as a direct root decomposition",
                    "variant_hint": "",
                }
            )
        if "sub-nodes" in system_lower or "sub_nodes" in system_lower:
            return json.dumps(_decompose_payload(self._task))
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


class MockSemanticIndex:
    def __init__(self, declarations: list[Declaration]) -> None:
        self._declarations = declarations

    def _score(self, text: str, decl: Declaration) -> float:
        q_words = set(_slug(text).split("_"))
        d_words = set(
            _slug(f"{decl.name} {decl.type_signature} {decl.docstring}").split("_")
        )
        return float(len(q_words & d_words))

    def search_by_embedding(self, query_text: str, k: int = 10):
        scored = sorted(
            ((d, self._score(query_text, d)) for d in self._declarations),
            key=lambda x: x[1],
            reverse=True,
        )
        return scored[:k]

    def search_by_type(self, type_signature: str, k: int = 10):
        ranked = sorted(
            self._declarations,
            key=lambda d: self._score(type_signature, d),
            reverse=True,
        )
        return ranked[:k]

    def get_declaration(self, name: str):
        for decl in self._declarations:
            if decl.name == name:
                return decl
        return None


class SingleTargetOracle:
    def __init__(self, expected_declaration: str) -> None:
        self._expected_declaration = expected_declaration

    async def verify_candidate(self, pdg_node, candidate):
        verified = candidate.declaration.name == self._expected_declaration
        return VerificationResult(
            candidate=candidate,
            verified=verified,
            compiler_output="ok" if verified else "type mismatch",
            proof_term=f"@{candidate.declaration.name}" if verified else "",
            error_message="" if verified else "type mismatch",
        )

    async def verify_candidates(self, pdg_node, candidates):
        results = []
        for candidate in candidates:
            result = await self.verify_candidate(pdg_node, candidate)
            results.append(result)
            if result.verified:
                break
        return results


def _make_hunter_llm():
    llm = AsyncMock()

    async def complete(system: str, user: str) -> str:
        lower = system.lower()
        if "rank" in lower or "score" in lower:
            return "[0, 1, 2]"
        if "reformulate" in lower:
            return '["fallback query"]'
        if "analy" in lower:
            return "Try a declaration that matches the key algorithmic term."
        return "[]"

    llm.complete = complete
    return llm


def _decompose_payload(task: TaskSpec) -> dict:
    sub_nodes = []
    for spec in task.sub_nodes:
        primitive_name = _slug(spec.name)
        sub_nodes.append(
            {
                "name": spec.name,
                "description": spec.description,
                "concept_type": task.paradigm.value,
                "inputs": [{"name": n, "type_desc": t} for n, t in spec.inputs],
                "outputs": [{"name": n, "type_desc": t} for n, t in spec.outputs],
                "type_signature": spec.type_signature,
                "is_atomic": True,
                "matched_primitive": primitive_name,
            }
        )

    edges = [
        {
            "source_name": e.source_name,
            "target_name": e.target_name,
            "output_name": e.output_name,
            "input_name": e.input_name,
            "data_type": e.data_type,
        }
        for e in task.edges
    ]
    return {"sub_nodes": sub_nodes, "edges": edges}


def _make_catalog(task: TaskSpec) -> PrimitiveCatalog:
    catalog = PrimitiveCatalog()
    for spec in task.sub_nodes:
        primitive_name = _slug(spec.name)
        catalog.add(
            AlgorithmicPrimitive(
                name=primitive_name,
                source="test-suite",
                category=task.paradigm,
                description=spec.description,
                inputs=[IOSpec(name=n, type_desc=t) for n, t in spec.inputs],
                outputs=[IOSpec(name=n, type_desc=t) for n, t in spec.outputs],
                type_signature=spec.type_signature,
            )
        )
    return catalog


def _empty_skill_index():
    skill_index = AsyncMock()
    skill_index.search = lambda query, k=10: []
    return skill_index


def _node_name_set(cdg) -> set[str]:
    return {node.name for node in cdg.nodes}


def _assert_edge_by_name(cdg, source_name: str, target_name: str) -> bool:
    by_id = {node.node_id: node.name for node in cdg.nodes}
    for edge in cdg.edges:
        if (
            by_id.get(edge.source_id) == source_name
            and by_id.get(edge.target_id) == target_name
        ):
            return True
    return False


def _task_specs() -> tuple[TaskSpec, ...]:
    return (
        TaskSpec(
            task_id="merge_sort",
            prompt=(
                "I need an efficient way to order a list of integers from smallest to largest. "
                "The method should split the list into two halves, order each half independently, "
                "and then combine them back together in the correct order."
            ),
            sample_input="[34, 7, 23, 32, 5, 62]",
            sample_output="[5, 7, 23, 32, 34, 62]",
            paradigm=ConceptType.DIVIDE_AND_CONQUER,
            sub_nodes=(
                SubNodeSpec(
                    name="List Merge Primitive",
                    description="Use list.merge to combine sorted halves.",
                    type_signature="list[int] -> list[int] -> list[int]",
                    inputs=(("left", "list[int]"), ("right", "list[int]")),
                    outputs=(("merged", "list[int]"),),
                ),
                SubNodeSpec(
                    name="Sortedness Theorem",
                    description="Apply sorting theorem to prove merged output remains sorted.",
                    type_signature="sorted(left) -> sorted(right) -> sorted(merged)",
                    inputs=(("merged", "list[int]"),),
                    outputs=(("proof", "Prop"),),
                ),
                SubNodeSpec(
                    name="Return Merged List",
                    description="Return the merged sorted result.",
                    type_signature="list[int] -> list[int]",
                    inputs=(("merged", "list[int]"),),
                    outputs=(("result", "list[int]"),),
                ),
            ),
            edges=(
                EdgeSpec(
                    "List Merge Primitive",
                    "Sortedness Theorem",
                    "merged",
                    "merged",
                    "list[int]",
                ),
                EdgeSpec(
                    "List Merge Primitive",
                    "Return Merged List",
                    "merged",
                    "merged",
                    "list[int]",
                ),
            ),
            required_round1_nodes=(
                "List Merge Primitive",
                "Sortedness Theorem",
                "Return Merged List",
            ),
            round2_checks=(Round2Check("list.merge", "List.merge"),),
        ),
        TaskSpec(
            task_id="maximum_subarray",
            prompt=(
                "Given a list of daily stock price changes (positive and negative integers), "
                "find the contiguous period that results in the highest profit. "
                "I need the start index, end index, and the total sum."
            ),
            sample_input="[-2, 1, -3, 4, -1, 2, 1, -5, 4]",
            sample_output="Subarray: [4, -1, 2, 1], Sum: 6",
            paradigm=ConceptType.DIVIDE_AND_CONQUER,
            sub_nodes=(
                SubNodeSpec(
                    name="Left_Sum",
                    description="Compute best subarray fully in left half.",
                    type_signature="array[int] -> int",
                    inputs=(("left_half", "array[int]"),),
                    outputs=(("left_best", "int"),),
                ),
                SubNodeSpec(
                    name="Right_Sum",
                    description="Compute best subarray fully in right half.",
                    type_signature="array[int] -> int",
                    inputs=(("right_half", "array[int]"),),
                    outputs=(("right_best", "int"),),
                ),
                SubNodeSpec(
                    name="Cross_Sum",
                    description="Compute best subarray crossing midpoint using left and right accumulations.",
                    type_signature="int -> int -> int",
                    inputs=(("left_best", "int"), ("right_best", "int")),
                    outputs=(("cross_best", "int"),),
                ),
            ),
            edges=(
                EdgeSpec("Left_Sum", "Cross_Sum", "left_best", "left_best", "int"),
                EdgeSpec("Right_Sum", "Cross_Sum", "right_best", "right_best", "int"),
            ),
            required_round1_nodes=("Left_Sum", "Right_Sum", "Cross_Sum"),
            round2_checks=(),
        ),
        TaskSpec(
            task_id="dijkstra",
            prompt=(
                "Find the cheapest path between two nodes in a network where every connection has "
                "a non-negative cost. I need to explore the closest connections first and update the "
                "costs to neighbors as I go."
            ),
            sample_input="Graph: {A:[(B,1), (C,4)], B:[(C,2), (D,5)], C:[(D,1)]}, Start: A, End: D",
            sample_output="Path: [A, B, C, D], Cost: 4",
            paradigm=ConceptType.GRAPH_OPTIMIZATION,
            sub_nodes=(
                SubNodeSpec(
                    name="Extract-Min Priority Queue",
                    description="Pop the next frontier node from a Priority Queue.",
                    type_signature="priority_queue[node] -> node",
                    inputs=(("frontier", "priority_queue[node]"),),
                    outputs=(("current", "node"),),
                ),
                SubNodeSpec(
                    name="Relax Neighbor Costs",
                    description="Relax outgoing edges and update tentative distances.",
                    type_signature="node -> graph -> map[node,int]",
                    inputs=(("current", "node"), ("graph", "weighted_graph")),
                    outputs=(("distances", "map[node,int]"),),
                ),
                SubNodeSpec(
                    name="Min-Heap Decrease-Key",
                    description="Apply Min-Heap decrease-key for improved neighbor costs.",
                    type_signature="min_heap[node] -> map[node,int] -> min_heap[node]",
                    inputs=(("distances", "map[node,int]"),),
                    outputs=(("heap", "min_heap[node]"),),
                ),
            ),
            edges=(
                EdgeSpec(
                    "Extract-Min Priority Queue",
                    "Relax Neighbor Costs",
                    "current",
                    "current",
                    "node",
                ),
                EdgeSpec(
                    "Relax Neighbor Costs",
                    "Min-Heap Decrease-Key",
                    "distances",
                    "distances",
                    "map[node,int]",
                ),
            ),
            required_round1_nodes=("Priority Queue", "Min-Heap"),
            round2_checks=(
                Round2Check("priority queue", "Data.PriorityQueue"),
                Round2Check("min-heap", "Data.MinHeap"),
            ),
        ),
        TaskSpec(
            task_id="bfs",
            prompt=(
                "I need to find the shortest distance (in number of hops) from a starting person "
                "to everyone else in a social network. Process friends level-by-level."
            ),
            sample_input="AdjList: {1:[2,3], 2:[4], 3:[4], 4:[]}, Start: 1",
            sample_output="Distances: {1:0, 2:1, 3:1, 4:2}",
            paradigm=ConceptType.GRAPH_TRAVERSAL,
            sub_nodes=(
                SubNodeSpec(
                    name="Frontier Queue (FIFO)",
                    description="Maintain traversal frontier with strict FIFO queue semantics.",
                    type_signature="queue[node] -> queue[node]",
                    inputs=(("frontier", "queue[node]"),),
                    outputs=(("frontier", "queue[node]"),),
                ),
                SubNodeSpec(
                    name="Process Layer",
                    description="Process nodes level-by-level from the queue.",
                    type_signature="queue[node] -> map[node,int]",
                    inputs=(("frontier", "queue[node]"),),
                    outputs=(("distances", "map[node,int]"),),
                ),
                SubNodeSpec(
                    name="Record Hop Distance",
                    description="Store shortest hop counts for newly discovered nodes.",
                    type_signature="map[node,int] -> map[node,int]",
                    inputs=(("distances", "map[node,int]"),),
                    outputs=(("result", "map[node,int]"),),
                ),
            ),
            edges=(
                EdgeSpec(
                    "Frontier Queue (FIFO)",
                    "Process Layer",
                    "frontier",
                    "frontier",
                    "queue[node]",
                ),
                EdgeSpec(
                    "Process Layer",
                    "Record Hop Distance",
                    "distances",
                    "distances",
                    "map[node,int]",
                ),
            ),
            required_round1_nodes=("FIFO", "Queue"),
            round2_checks=(),
        ),
        TaskSpec(
            task_id="topological_sort",
            prompt=(
                "I have a list of tasks where some tasks must be completed before others. "
                "Produce a valid linear order to perform all tasks. "
                "If there's a cycle where tasks wait on each other, tell me it's impossible."
            ),
            sample_input="Tasks: [A, B, C], Deps: [(A, B), (B, C)]",
            sample_output="[A, B, C]",
            paradigm=ConceptType.GRAPH_TRAVERSAL,
            sub_nodes=(
                SubNodeSpec(
                    name="Compute In-Degree",
                    description="Compute in-degree of each node in dependency graph.",
                    type_signature="dag -> map[node,int]",
                    inputs=(("graph", "dag"),),
                    outputs=(("in_degree", "map[node,int]"),),
                ),
                SubNodeSpec(
                    name="Kahn Queue",
                    description="Process zero in-degree nodes in a queue.",
                    type_signature="map[node,int] -> queue[node]",
                    inputs=(("in_degree", "map[node,int]"),),
                    outputs=(("frontier", "queue[node]"),),
                ),
                SubNodeSpec(
                    name="Detect Cycle (DAG Check)",
                    description="Fail when processed node count is below total nodes (cycle detected).",
                    type_signature="queue[node] -> bool",
                    inputs=(("frontier", "queue[node]"),),
                    outputs=(("is_dag", "bool"),),
                ),
            ),
            edges=(
                EdgeSpec(
                    "Compute In-Degree",
                    "Kahn Queue",
                    "in_degree",
                    "in_degree",
                    "map[node,int]",
                ),
                EdgeSpec(
                    "Kahn Queue",
                    "Detect Cycle (DAG Check)",
                    "frontier",
                    "frontier",
                    "queue[node]",
                ),
            ),
            required_round1_nodes=("DAG", "Cycle"),
            round2_checks=(),
        ),
        TaskSpec(
            task_id="lcs",
            prompt=(
                "Compare two strings of DNA. Find the length of the longest sequence of characters "
                "that appear in both strings in the same relative order, but not necessarily consecutively."
            ),
            sample_input='S1: "ABCDE", S2: "ACE"',
            sample_output='3 (matches "A", "C", "E")',
            paradigm=ConceptType.DYNAMIC_PROGRAMMING,
            sub_nodes=(
                SubNodeSpec(
                    name="Cell[i-1][j]",
                    description="DP value from the top cell in the 2D grid.",
                    type_signature="matrix[int] -> int",
                    inputs=(("dp", "matrix[int]"),),
                    outputs=(("top", "int"),),
                ),
                SubNodeSpec(
                    name="Cell[i][j-1]",
                    description="DP value from the left cell in the 2D grid.",
                    type_signature="matrix[int] -> int",
                    inputs=(("dp", "matrix[int]"),),
                    outputs=(("left", "int"),),
                ),
                SubNodeSpec(
                    name="Cell[i][j]",
                    description="Compute current grid cell from top and left dependencies.",
                    type_signature="int -> int -> int",
                    inputs=(("top", "int"), ("left", "int")),
                    outputs=(("value", "int"),),
                ),
            ),
            edges=(
                EdgeSpec("Cell[i-1][j]", "Cell[i][j]", "top", "top", "int"),
                EdgeSpec("Cell[i][j-1]", "Cell[i][j]", "left", "left", "int"),
            ),
            required_round1_nodes=("Cell[i-1][j]", "Cell[i][j-1]", "Cell[i][j]"),
            round2_checks=(),
        ),
        TaskSpec(
            task_id="activity_selection",
            prompt=(
                "I have a set of meetings with start and end times. I want to attend as many as "
                "possible, but I can't be in two meetings at once. Select the maximum number of "
                "non-overlapping meetings."
            ),
            sample_input="[(1,3), (2,4), (3,5), (0,6), (5,7), (8,9)]",
            sample_output="[(1,3), (3,5), (5,7), (8,9)]",
            paradigm=ConceptType.GREEDY,
            sub_nodes=(
                SubNodeSpec(
                    name="Sort by Finish Time",
                    description="Sort activities ascending by finish time (greedy heuristic).",
                    type_signature="list[activity] -> list[activity]",
                    inputs=(("activities", "list[activity]"),),
                    outputs=(("ordered", "list[activity]"),),
                ),
                SubNodeSpec(
                    name="Select Compatible Activity",
                    description="Pick next activity with earliest finish compatible with current schedule.",
                    type_signature="list[activity] -> activity",
                    inputs=(("ordered", "list[activity]"),),
                    outputs=(("chosen", "activity"),),
                ),
                SubNodeSpec(
                    name="Update Schedule",
                    description="Append compatible activity and continue selection.",
                    type_signature="activity -> list[activity]",
                    inputs=(("chosen", "activity"),),
                    outputs=(("schedule", "list[activity]"),),
                ),
            ),
            edges=(
                EdgeSpec(
                    "Sort by Finish Time",
                    "Select Compatible Activity",
                    "ordered",
                    "ordered",
                    "list[activity]",
                ),
                EdgeSpec(
                    "Select Compatible Activity",
                    "Update Schedule",
                    "chosen",
                    "chosen",
                    "activity",
                ),
            ),
            required_round1_nodes=("Sort by Finish Time",),
            round2_checks=(),
        ),
        TaskSpec(
            task_id="huffman",
            prompt=(
                "Compress a text string by assigning shorter binary codes to more frequent characters. "
                "Build a binary tree where the leaves are characters and the path determines the code."
            ),
            sample_input='"bananas"',
            sample_output="b:100, a:0, n:11, s:101",
            paradigm=ConceptType.GREEDY,
            sub_nodes=(
                SubNodeSpec(
                    name="Count Frequencies",
                    description="Count character frequencies for Huffman coding.",
                    type_signature="string -> map[char,int]",
                    inputs=(("text", "string"),),
                    outputs=(("freqs", "map[char,int]"),),
                ),
                SubNodeSpec(
                    name="Priority Queue Build (Min-Heap)",
                    description="Build a Priority Queue / Min-Heap of nodes by frequency.",
                    type_signature="map[char,int] -> min_heap[node]",
                    inputs=(("freqs", "map[char,int]"),),
                    outputs=(("heap", "min_heap[node]"),),
                ),
                SubNodeSpec(
                    name="BinaryTree Merge",
                    description="Merge lowest-frequency nodes into a BinaryTree until one root remains.",
                    type_signature="min_heap[node] -> binary_tree[node]",
                    inputs=(("heap", "min_heap[node]"),),
                    outputs=(("tree", "binary_tree[node]"),),
                ),
            ),
            edges=(
                EdgeSpec(
                    "Count Frequencies",
                    "Priority Queue Build (Min-Heap)",
                    "freqs",
                    "freqs",
                    "map[char,int]",
                ),
                EdgeSpec(
                    "Priority Queue Build (Min-Heap)",
                    "BinaryTree Merge",
                    "heap",
                    "heap",
                    "min_heap[node]",
                ),
            ),
            required_round1_nodes=("Priority Queue", "BinaryTree"),
            round2_checks=(
                Round2Check("priority queue", "Data.PriorityQueue"),
                Round2Check("binarytree", "Data.BinaryTree"),
            ),
        ),
        TaskSpec(
            task_id="graham_scan",
            prompt=(
                "Given a set of points on a 2D plane, find the smallest polygon that encloses all of them. "
                "I want the subset of points that form the perimeter."
            ),
            sample_input="[(0,0), (1,1), (2,2), (0,2), (2,0), (1,0.5)]",
            sample_output="[(0,0), (2,0), (2,2), (0,2)]",
            paradigm=ConceptType.GEOMETRY,
            sub_nodes=(
                SubNodeSpec(
                    name="Sort by polar_angle",
                    description="Sort points by polar_angle around pivot.",
                    type_signature="list[point] -> list[point]",
                    inputs=(("points", "list[point]"),),
                    outputs=(("sorted", "list[point]"),),
                ),
                SubNodeSpec(
                    name="ccw Turn Predicate",
                    description="Use ccw orientation test to keep only left turns.",
                    type_signature="point -> point -> point -> bool",
                    inputs=(("sorted", "list[point]"),),
                    outputs=(("turn_ok", "bool"),),
                ),
                SubNodeSpec(
                    name="Build Hull Stack",
                    description="Construct convex hull stack from sorted points and ccw checks.",
                    type_signature="bool -> list[point]",
                    inputs=(("turn_ok", "bool"),),
                    outputs=(("hull", "list[point]"),),
                ),
            ),
            edges=(
                EdgeSpec(
                    "Sort by polar_angle",
                    "ccw Turn Predicate",
                    "sorted",
                    "sorted",
                    "list[point]",
                ),
                EdgeSpec(
                    "ccw Turn Predicate",
                    "Build Hull Stack",
                    "turn_ok",
                    "turn_ok",
                    "bool",
                ),
            ),
            required_round1_nodes=("polar_angle", "ccw"),
            round2_checks=(
                Round2Check("polar_angle", "Geometry.polar_angle"),
                Round2Check("ccw", "Geometry.ccw"),
            ),
        ),
        TaskSpec(
            task_id="gcd",
            prompt=(
                "Find the largest positive integer that divides two numbers without leaving a remainder. "
                "Use the method of repeated subtraction or modulo."
            ),
            sample_input="48, 18",
            sample_output="6",
            paradigm=ConceptType.NUMBER_THEORY,
            sub_nodes=(
                SubNodeSpec(
                    name="Euclid Mod Step",
                    description="Compute a mod b in Euclid's recurrence.",
                    type_signature="nat -> nat -> nat",
                    inputs=(("a", "nat"), ("b", "nat")),
                    outputs=(("r", "nat"),),
                ),
                SubNodeSpec(
                    name="Base Case b = 0",
                    description="Check atomic stop condition for Euclid algorithm.",
                    type_signature="nat -> bool",
                    inputs=(("b", "nat"),),
                    outputs=(("done", "bool"),),
                ),
                SubNodeSpec(
                    name="gcd Primitive",
                    description="Use primitive gcd theorem/definition from library.",
                    type_signature="nat -> nat -> nat",
                    inputs=(("a", "nat"), ("b", "nat")),
                    outputs=(("g", "nat"),),
                ),
            ),
            edges=(
                EdgeSpec("Euclid Mod Step", "Base Case b = 0", "r", "b", "nat"),
                EdgeSpec("Base Case b = 0", "gcd Primitive", "done", "a", "nat"),
            ),
            required_round1_nodes=("gcd",),
            round2_checks=(Round2Check("gcd", "Nat.gcd"),),
        ),
    )


TASK_SPECS = _task_specs()


@pytest.mark.asyncio
@pytest.mark.parametrize("task", TASK_SPECS, ids=[t.task_id for t in TASK_SPECS])
async def test_algorithmic_tasks_round1_round2_end_to_end(task: TaskSpec):
    """E2E: NL prompt -> CDG decomposition -> PDG handoff -> retrieval+verification."""
    catalog = _make_catalog(task)
    llm = TaskAwareArchitectLLM(task)

    agent = DecompositionAgent(
        catalog=catalog,
        skill_index=_empty_skill_index(),
        llm=llm,  # type: ignore[arg-type]
        max_depth=8,
    )

    cdg = await agent.decompose(task.prompt)
    node_names = _node_name_set(cdg)

    for required in task.required_round1_nodes:
        assert any(
            required.lower() in name.lower() for name in node_names
        ), f"{task.task_id}: missing required Round 1 node containing '{required}'"

    if task.task_id == "lcs":
        assert _assert_edge_by_name(cdg, "Cell[i-1][j]", "Cell[i][j]")
        assert _assert_edge_by_name(cdg, "Cell[i][j-1]", "Cell[i][j]")

    if task.task_id == "gcd":
        assert len(cdg.nodes) <= 16, "GCD should remain a small decomposition graph."

    pdg_nodes = to_pdg_nodes(cdg, prover=Prover.LEAN4)
    assert pdg_nodes

    for check in task.round2_checks:
        query_node = next(
            (
                node
                for node in pdg_nodes
                if check.query_hint.lower()
                in f"{node.statement} {node.informal_desc}".lower()
            ),
            None,
        )
        assert (
            query_node is not None
        ), f"{task.task_id}: no PDG node contains query hint '{check.query_hint}'"

        declarations = [
            Declaration(
                name=check.expected_declaration,
                type_signature=query_node.statement,
                docstring=f"expected match for {check.query_hint}",
                prover=Prover.LEAN4,
            ),
            Declaration(
                name="Distractor.One",
                type_signature="unrelated_type_1",
                docstring="irrelevant",
                prover=Prover.LEAN4,
            ),
            Declaration(
                name="Distractor.Two",
                type_signature="unrelated_type_2",
                docstring="irrelevant",
                prover=Prover.LEAN4,
            ),
        ]

        hunter = HunterAgent(
            index=MockSemanticIndex(declarations),  # type: ignore[arg-type]
            oracle=SingleTargetOracle(check.expected_declaration),  # type: ignore[arg-type]
            llm=_make_hunter_llm(),
            max_iterations=2,
            top_k_verify=2,
            search_k=5,
        )
        result = await hunter.find_match(query_node)

        assert (
            result.success
        ), f"{task.task_id}: Hunter failed for {check.expected_declaration}"
        assert result.verified_match is not None
        assert (
            result.verified_match.candidate.declaration.name
            == check.expected_declaration
        )
