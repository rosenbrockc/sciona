"""End-to-end tests verifying the generalized pipeline produces handoff-ready CDGs
for non-ECG domains (graph algorithms, linear algebra, sorting).

Tests 1-3 exercise DecompositionAgent with mock LLMs.
Tests 4-7 build CDGs directly from AlgorithmicNode objects for speed and reliability.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from unittest.mock import AsyncMock

import pytest

if importlib.util.find_spec("langgraph") is None:
    pytest.skip("requires langgraph", allow_module_level=True)

from ageom.architect.catalog import PrimitiveCatalog
from ageom.architect.graph import DecompositionAgent
from ageom.architect.handoff import CDGExport, to_pdg_nodes, validate_handoff
from ageom.architect.models import (
    AlgorithmicNode,
    AlgorithmicPrimitive,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from ageom.synthesizer.assembler import Assembler
from ageom.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
    PDGNode,
    Prover,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _empty_skill_index():
    skill_index = AsyncMock()
    skill_index.search = lambda query, k=10: []
    return skill_index


# ---------------------------------------------------------------------------
# Mock LLM for DecompositionAgent (tests 1-3)
# ---------------------------------------------------------------------------

_DIJKSTRA_SUBNODES = [
    {
        "name": "Initialize Distances",
        "description": "Set all node distances to infinity except source to zero.",
        "concept_type": "graph_optimization",
        "inputs": [{"name": "graph", "type_desc": "weighted_graph"}, {"name": "source", "type_desc": "node"}],
        "outputs": [{"name": "distances", "type_desc": "map[node, float]"}],
        "type_signature": "weighted_graph -> node -> map[node, float]",
        "is_atomic": True,
        "matched_primitive": "initialize_distances",
    },
    {
        "name": "Relax Edges",
        "description": "Iterate over neighbors and update tentative distances via relaxation.",
        "concept_type": "graph_optimization",
        "inputs": [{"name": "distances", "type_desc": "map[node, float]"}, {"name": "graph", "type_desc": "weighted_graph"}],
        "outputs": [{"name": "distances", "type_desc": "map[node, float]"}],
        "type_signature": "map[node, float] -> weighted_graph -> map[node, float]",
        "is_atomic": True,
        "matched_primitive": "relax_edges",
    },
    {
        "name": "Extract Path",
        "description": "Trace back from target to source using predecessor map.",
        "concept_type": "graph_optimization",
        "inputs": [{"name": "distances", "type_desc": "map[node, float]"}, {"name": "target", "type_desc": "node"}],
        "outputs": [{"name": "path", "type_desc": "list[node]"}],
        "type_signature": "map[node, float] -> node -> list[node]",
        "is_atomic": True,
        "matched_primitive": "extract_path",
    },
]

_DIJKSTRA_EDGES = [
    {"source_name": "Initialize Distances", "target_name": "Relax Edges",
     "output_name": "distances", "input_name": "distances", "data_type": "map[node, float]"},
    {"source_name": "Relax Edges", "target_name": "Extract Path",
     "output_name": "distances", "input_name": "distances", "data_type": "map[node, float]"},
]

_CHOLESKY_SUBNODES = [
    {
        "name": "Cholesky Factorization",
        "description": "Decompose SPD matrix A into L * L^T.",
        "concept_type": "algebra",
        "inputs": [{"name": "A", "type_desc": "matrix[float]"}],
        "outputs": [{"name": "L", "type_desc": "matrix[float]"}],
        "type_signature": "matrix[float] -> matrix[float]",
        "is_atomic": True,
        "matched_primitive": "cholesky_factorization",
    },
    {
        "name": "Triangular Solve",
        "description": "Solve L * y = b by forward substitution, then L^T * x = y by back substitution.",
        "concept_type": "algebra",
        "inputs": [{"name": "L", "type_desc": "matrix[float]"}, {"name": "b", "type_desc": "vector[float]"}],
        "outputs": [{"name": "x", "type_desc": "vector[float]"}],
        "type_signature": "matrix[float] -> vector[float] -> vector[float]",
        "is_atomic": True,
        "matched_primitive": "triangular_solve",
    },
]

_CHOLESKY_EDGES = [
    {"source_name": "Cholesky Factorization", "target_name": "Triangular Solve",
     "output_name": "L", "input_name": "L", "data_type": "matrix[float]"},
]

_MERGE_SORT_SUBNODES = [
    {
        "name": "Split List",
        "description": "Divide list into two roughly equal halves.",
        "concept_type": "divide_and_conquer",
        "inputs": [{"name": "lst", "type_desc": "list[int]"}],
        "outputs": [{"name": "left", "type_desc": "list[int]"}, {"name": "right", "type_desc": "list[int]"}],
        "type_signature": "list[int] -> (list[int], list[int])",
        "is_atomic": True,
        "matched_primitive": "split_list",
    },
    {
        "name": "Merge Sorted",
        "description": "Merge two sorted lists into one sorted list.",
        "concept_type": "divide_and_conquer",
        "inputs": [{"name": "left", "type_desc": "list[int]"}, {"name": "right", "type_desc": "list[int]"}],
        "outputs": [{"name": "merged", "type_desc": "list[int]"}],
        "type_signature": "list[int] -> list[int] -> list[int]",
        "is_atomic": True,
        "matched_primitive": "merge_sorted",
    },
]

_MERGE_SORT_EDGES = [
    {"source_name": "Split List", "target_name": "Merge Sorted",
     "output_name": "left", "input_name": "left", "data_type": "list[int]"},
]

# Map of paradigm keyword to response payload
_DOMAIN_CONFIGS = {
    "dijkstra": {
        "paradigm": "graph_optimization",
        "sub_nodes": _DIJKSTRA_SUBNODES,
        "edges": _DIJKSTRA_EDGES,
    },
    "cholesky": {
        "paradigm": "algebra",
        "sub_nodes": _CHOLESKY_SUBNODES,
        "edges": _CHOLESKY_EDGES,
    },
    "merge_sort": {
        "paradigm": "divide_and_conquer",
        "sub_nodes": _MERGE_SORT_SUBNODES,
        "edges": _MERGE_SORT_EDGES,
    },
}


class CrossDomainArchitectLLM:
    """Mock LLM that returns domain-appropriate decomposition responses."""

    def __init__(self, domain_key: str) -> None:
        self._config = _DOMAIN_CONFIGS[domain_key]

    async def complete(self, system: str, user: str) -> str:
        system_lower = system.lower()

        # Strategy selection prompt — return CUSTOM to bypass skeleton templates
        if "best" in system_lower and "paradigm" in system_lower:
            return json.dumps({
                "paradigm": ConceptType.CUSTOM.value,
                "rationale": f"Mocked paradigm for cross-domain test",
                "variant_hint": "",
            })

        # Decompose prompt
        if "sub-nodes" in system_lower or "sub_nodes" in system_lower:
            return json.dumps({
                "sub_nodes": self._config["sub_nodes"],
                "edges": self._config["edges"],
            })

        # Critique prompt
        if "critic" in system_lower or "evaluate" in system_lower:
            return json.dumps({
                "approved": True,
                "reason": "Valid decomposition",
                "io_issues": [],
                "flagged_nodes": [],
            })

        return "{}"


def _make_catalog_for_domain(domain_key: str) -> PrimitiveCatalog:
    """Build a PrimitiveCatalog containing primitives for the given domain."""
    config = _DOMAIN_CONFIGS[domain_key]
    catalog = PrimitiveCatalog()
    paradigm = ConceptType(config["paradigm"])
    for sn in config["sub_nodes"]:
        catalog.add(AlgorithmicPrimitive(
            name=sn["matched_primitive"],
            source="test-suite",
            category=paradigm,
            description=sn["description"],
            inputs=[IOSpec(name=i["name"], type_desc=i["type_desc"]) for i in sn["inputs"]],
            outputs=[IOSpec(name=o["name"], type_desc=o["type_desc"]) for o in sn["outputs"]],
            type_signature=sn["type_signature"],
        ))
    return catalog


# ---------------------------------------------------------------------------
# Manually-built CDG fixtures (tests 4-7)
# ---------------------------------------------------------------------------

def _build_dijkstra_cdg() -> CDGExport:
    root = AlgorithmicNode(
        node_id="dijkstra_root",
        name="Find Shortest Path",
        description="Find shortest path in weighted graph using Dijkstra.",
        concept_type=ConceptType.GRAPH_OPTIMIZATION,
        status=NodeStatus.DECOMPOSED,
        children=["dj_init", "dj_relax", "dj_extract"],
        type_signature="weighted_graph -> node -> node -> list[node]",
        inputs=[
            IOSpec(name="graph", type_desc="weighted_graph"),
            IOSpec(name="source", type_desc="node"),
            IOSpec(name="target", type_desc="node"),
        ],
        outputs=[IOSpec(name="path", type_desc="list[node]")],
    )
    n1 = AlgorithmicNode(
        node_id="dj_init",
        parent_id="dijkstra_root",
        name="Initialize Distances",
        description="Set all node distances to infinity except source to zero.",
        concept_type=ConceptType.GRAPH_OPTIMIZATION,
        status=NodeStatus.ATOMIC,
        depth=1,
        type_signature="weighted_graph -> node -> map[node, float]",
        inputs=[
            IOSpec(name="graph", type_desc="weighted_graph"),
            IOSpec(name="source", type_desc="node"),
        ],
        outputs=[IOSpec(name="distances", type_desc="map[node, float]")],
    )
    n2 = AlgorithmicNode(
        node_id="dj_relax",
        parent_id="dijkstra_root",
        name="Relax Edges",
        description="Iterate over neighbors and update tentative distances.",
        concept_type=ConceptType.GRAPH_OPTIMIZATION,
        status=NodeStatus.ATOMIC,
        depth=1,
        type_signature="map[node, float] -> weighted_graph -> map[node, float]",
        inputs=[
            IOSpec(name="distances", type_desc="map[node, float]"),
            IOSpec(name="graph", type_desc="weighted_graph"),
        ],
        outputs=[IOSpec(name="distances", type_desc="map[node, float]")],
    )
    n3 = AlgorithmicNode(
        node_id="dj_extract",
        parent_id="dijkstra_root",
        name="Extract Path",
        description="Trace back from target to source using predecessor map.",
        concept_type=ConceptType.GRAPH_OPTIMIZATION,
        status=NodeStatus.ATOMIC,
        depth=1,
        type_signature="map[node, float] -> node -> list[node]",
        inputs=[
            IOSpec(name="distances", type_desc="map[node, float]"),
            IOSpec(name="target", type_desc="node"),
        ],
        outputs=[IOSpec(name="path", type_desc="list[node]")],
    )
    edges = [
        DependencyEdge(
            source_id="dj_init", target_id="dj_relax",
            output_name="distances", input_name="distances",
            source_type="map[node, float]", target_type="map[node, float]",
        ),
        DependencyEdge(
            source_id="dj_relax", target_id="dj_extract",
            output_name="distances", input_name="distances",
            source_type="map[node, float]", target_type="map[node, float]",
        ),
    ]
    return CDGExport(nodes=[root, n1, n2, n3], edges=edges, metadata={"goal": "Dijkstra shortest path"})


def _build_cholesky_cdg() -> CDGExport:
    root = AlgorithmicNode(
        node_id="chol_root",
        name="Solve SPD System",
        description="Solve SPD linear system via Cholesky factorization.",
        concept_type=ConceptType.ALGEBRA,
        status=NodeStatus.DECOMPOSED,
        children=["chol_factor", "chol_solve"],
        type_signature="matrix[float] -> vector[float] -> vector[float]",
        inputs=[
            IOSpec(name="A", type_desc="matrix[float]"),
            IOSpec(name="b", type_desc="vector[float]"),
        ],
        outputs=[IOSpec(name="x", type_desc="vector[float]")],
    )
    n1 = AlgorithmicNode(
        node_id="chol_factor",
        parent_id="chol_root",
        name="Cholesky Factorization",
        description="Decompose SPD matrix A into L * L^T.",
        concept_type=ConceptType.ALGEBRA,
        status=NodeStatus.ATOMIC,
        depth=1,
        type_signature="matrix[float] -> matrix[float]",
        inputs=[IOSpec(name="A", type_desc="matrix[float]")],
        outputs=[IOSpec(name="L", type_desc="matrix[float]")],
    )
    n2 = AlgorithmicNode(
        node_id="chol_solve",
        parent_id="chol_root",
        name="Triangular Solve",
        description="Solve L * y = b then L^T * x = y.",
        concept_type=ConceptType.ALGEBRA,
        status=NodeStatus.ATOMIC,
        depth=1,
        type_signature="matrix[float] -> vector[float] -> vector[float]",
        inputs=[
            IOSpec(name="L", type_desc="matrix[float]"),
            IOSpec(name="b", type_desc="vector[float]"),
        ],
        outputs=[IOSpec(name="x", type_desc="vector[float]")],
    )
    edges = [
        DependencyEdge(
            source_id="chol_factor", target_id="chol_solve",
            output_name="L", input_name="L",
            source_type="matrix[float]", target_type="matrix[float]",
        ),
    ]
    return CDGExport(nodes=[root, n1, n2], edges=edges, metadata={"goal": "Cholesky solve"})


def _build_merge_sort_cdg() -> CDGExport:
    root = AlgorithmicNode(
        node_id="ms_root",
        name="Merge Sort",
        description="Sort a list by splitting and merging.",
        concept_type=ConceptType.DIVIDE_AND_CONQUER,
        status=NodeStatus.DECOMPOSED,
        children=["ms_split", "ms_merge"],
        type_signature="list[int] -> list[int]",
        inputs=[IOSpec(name="lst", type_desc="list[int]")],
        outputs=[IOSpec(name="sorted_list", type_desc="list[int]")],
    )
    n1 = AlgorithmicNode(
        node_id="ms_split",
        parent_id="ms_root",
        name="Split List",
        description="Divide list into two roughly equal halves.",
        concept_type=ConceptType.DIVIDE_AND_CONQUER,
        status=NodeStatus.ATOMIC,
        depth=1,
        type_signature="list[int] -> tuple[list[int], list[int]]",
        inputs=[IOSpec(name="lst", type_desc="list[int]")],
        outputs=[
            IOSpec(name="left", type_desc="list[int]"),
            IOSpec(name="right", type_desc="list[int]"),
        ],
    )
    n2 = AlgorithmicNode(
        node_id="ms_merge",
        parent_id="ms_root",
        name="Merge Sorted",
        description="Merge two sorted lists into one sorted list.",
        concept_type=ConceptType.DIVIDE_AND_CONQUER,
        status=NodeStatus.ATOMIC,
        depth=1,
        type_signature="list[int] -> list[int] -> list[int]",
        inputs=[
            IOSpec(name="left", type_desc="list[int]"),
            IOSpec(name="right", type_desc="list[int]"),
        ],
        outputs=[IOSpec(name="merged", type_desc="list[int]")],
    )
    edges = [
        DependencyEdge(
            source_id="ms_split", target_id="ms_merge",
            output_name="left", input_name="left",
            source_type="list[int]", target_type="list[int]",
        ),
    ]
    return CDGExport(nodes=[root, n1, n2], edges=edges, metadata={"goal": "Merge sort"})


_CROSS_DOMAIN_CDGS = [
    _build_dijkstra_cdg,
    _build_cholesky_cdg,
    _build_merge_sort_cdg,
]


def _make_match_result(node: AlgorithmicNode) -> MatchResult:
    """Build a mock MatchResult for an atomic node."""
    decl_name = f"mock_impl.{_slug(node.name)}"
    type_sig = node.type_signature or "object -> object"
    decl = Declaration(
        name=decl_name,
        type_signature=type_sig,
        prover=Prover.PYTHON,
    )
    candidate = CandidateMatch(
        declaration=decl, score=1.0, retrieval_method="test_mock",
    )
    vr = VerificationResult(
        candidate=candidate, verified=True, proof_term=decl_name,
    )
    return MatchResult(
        pdg_node=PDGNode(
            predicate_id=node.node_id,
            statement=type_sig,
            prover=Prover.PYTHON,
        ),
        verified_match=vr,
        all_candidates=[candidate],
        all_verifications=[vr],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Tests 1-3: DecompositionAgent-based
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossDomainDecomposition:
    """Tests 1-3: Decompose non-ECG algorithms via DecompositionAgent."""

    @pytest.mark.asyncio
    async def test_dijkstra_decompose_produces_atomic_nodes(self):
        """Decompose shortest-path produces >= 2 ATOMIC leaves and is handoff-ready."""
        catalog = _make_catalog_for_domain("dijkstra")
        llm = CrossDomainArchitectLLM("dijkstra")
        agent = DecompositionAgent(
            catalog=catalog,
            skill_index=_empty_skill_index(),
            llm=llm,  # type: ignore[arg-type]
            max_depth=8,
        )
        cdg = await agent.decompose("Find shortest path in weighted graph")

        atomic = [n for n in cdg.nodes if n.status == NodeStatus.ATOMIC]
        assert len(atomic) >= 2, f"Expected >= 2 ATOMIC leaves, got {len(atomic)}: {[n.name for n in atomic]}"
        assert cdg.is_handoff_ready(), f"CDG not handoff-ready: {cdg.architect_issues()}"

    @pytest.mark.asyncio
    async def test_cholesky_decompose_produces_atomic_nodes(self):
        """Decompose Cholesky solve produces leaves with non-empty type_signature."""
        catalog = _make_catalog_for_domain("cholesky")
        llm = CrossDomainArchitectLLM("cholesky")
        agent = DecompositionAgent(
            catalog=catalog,
            skill_index=_empty_skill_index(),
            llm=llm,  # type: ignore[arg-type]
            max_depth=8,
        )
        cdg = await agent.decompose("Solve SPD linear system via Cholesky")

        leaves = cdg.leaf_nodes()
        assert len(leaves) >= 1, "Expected at least 1 leaf node"
        for leaf in leaves:
            assert leaf.type_signature, f"Leaf '{leaf.name}' has empty type_signature"

    @pytest.mark.asyncio
    async def test_merge_sort_decompose_produces_atomic_nodes(self):
        """Decompose merge sort produces >= 2 ATOMIC leaves."""
        catalog = _make_catalog_for_domain("merge_sort")
        llm = CrossDomainArchitectLLM("merge_sort")
        agent = DecompositionAgent(
            catalog=catalog,
            skill_index=_empty_skill_index(),
            llm=llm,  # type: ignore[arg-type]
            max_depth=8,
        )
        cdg = await agent.decompose("Sort a list using merge sort divide and conquer")

        atomic = [n for n in cdg.nodes if n.status == NodeStatus.ATOMIC]
        assert len(atomic) >= 2, f"Expected >= 2 ATOMIC leaves, got {len(atomic)}: {[n.name for n in atomic]}"


# ═══════════════════════════════════════════════════════════════════════════
# Tests 4-7: Directly-built CDGs
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossDomainHandoff:
    """Tests 4-7: Validate handoff, PDG conversion, assembly, and concept types."""

    @pytest.mark.parametrize(
        "cdg_factory",
        _CROSS_DOMAIN_CDGS,
        ids=["dijkstra", "cholesky", "merge_sort"],
    )
    def test_cross_domain_handoff_validation(self, cdg_factory):
        """validate_handoff() returns empty issues list for each domain CDG."""
        cdg = cdg_factory()
        issues = validate_handoff(cdg)
        assert issues == [], f"Handoff validation issues: {issues}"

    @pytest.mark.parametrize(
        "cdg_factory",
        _CROSS_DOMAIN_CDGS,
        ids=["dijkstra", "cholesky", "merge_sort"],
    )
    def test_cross_domain_pdg_conversion(self, cdg_factory):
        """to_pdg_nodes() returns one PDGNode per ATOMIC leaf with valid fields."""
        cdg = cdg_factory()
        pdg_nodes = to_pdg_nodes(cdg, prover=Prover.PYTHON)

        atomic_leaves = cdg.leaf_nodes()
        assert len(pdg_nodes) == len(atomic_leaves), (
            f"Expected {len(atomic_leaves)} PDGNodes, got {len(pdg_nodes)}"
        )
        for pdg_node in pdg_nodes:
            assert pdg_node.statement, f"PDGNode {pdg_node.predicate_id} has empty statement"
            assert pdg_node.informal_desc, f"PDGNode {pdg_node.predicate_id} has empty informal_desc"

    @pytest.mark.parametrize(
        "cdg_factory",
        _CROSS_DOMAIN_CDGS,
        ids=["dijkstra", "cholesky", "merge_sort"],
    )
    def test_cross_domain_assembler_produces_valid_python(self, cdg_factory):
        """Assembler produces code that passes ast.parse() with sorry_count == 0."""
        cdg = cdg_factory()
        match_results = [_make_match_result(n) for n in cdg.leaf_nodes()]

        assembler = Assembler(Prover.PYTHON)
        skeleton = assembler.assemble(cdg, match_results)

        # Must parse without SyntaxError
        try:
            ast.parse(skeleton.source_code)
        except SyntaxError as exc:
            pytest.fail(
                f"Generated code has syntax error: {exc}\n\n"
                f"--- source ---\n{skeleton.source_code}"
            )
        assert skeleton.sorry_count == 0, (
            f"Expected sorry_count == 0, got {skeleton.sorry_count}\n"
            f"--- source ---\n{skeleton.source_code}"
        )

    @pytest.mark.parametrize(
        "cdg_factory",
        _CROSS_DOMAIN_CDGS,
        ids=["dijkstra", "cholesky", "merge_sort"],
    )
    def test_non_signal_cdg_has_no_signal_concept_types(self, cdg_factory):
        """No node in a non-signal CDG has SIGNAL_FILTER or SIGNAL_TRANSFORM."""
        cdg = cdg_factory()
        signal_types = {ConceptType.SIGNAL_FILTER, ConceptType.SIGNAL_TRANSFORM}
        for node in cdg.nodes:
            assert node.concept_type not in signal_types, (
                f"Node '{node.name}' has signal concept type {node.concept_type.value} "
                f"in a non-signal domain CDG"
            )
