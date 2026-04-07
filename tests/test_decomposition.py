"""Tests for the Decomposition Engine (Phase 2) with mocked LLM."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from sciona.architect.catalog import PrimitiveCatalog, seed_builtin_primitives
from sciona.architect.deterministic_decompose import (
    DeterministicRewriteError,
    build_deterministic_decomposition,
)
from sciona.architect.models import (
    AlgorithmicNode,
    AlgorithmicPrimitive,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.architect.state import DecompositionState, _merge_nodes
from sciona.shared_context import InMemorySharedContextStore, SharedContextMetrics

# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------


def _make_catalog() -> PrimitiveCatalog:
    """Small catalog with merge, compare, binary_search primitives."""
    catalog = PrimitiveCatalog()
    catalog.add(
        AlgorithmicPrimitive(
            name="merge",
            source="clrs-30",
            category=ConceptType.SORTING,
            description="Merge two sorted lists into one sorted list",
            inputs=[
                IOSpec(name="left", type_desc="list[comparable]"),
                IOSpec(name="right", type_desc="list[comparable]"),
            ],
            outputs=[IOSpec(name="result", type_desc="list[comparable]")],
            type_signature="list[T] -> list[T] -> list[T]",
        )
    )
    catalog.add(
        AlgorithmicPrimitive(
            name="compare",
            source="clrs-30",
            category=ConceptType.SORTING,
            description="Compare two elements and return ordering",
            inputs=[
                IOSpec(name="a", type_desc="comparable"),
                IOSpec(name="b", type_desc="comparable"),
            ],
            outputs=[IOSpec(name="order", type_desc="bool")],
            type_signature="T -> T -> bool",
        )
    )
    catalog.add(
        AlgorithmicPrimitive(
            name="binary_search",
            source="clrs-30",
            category=ConceptType.SEARCHING,
            description="Search for a target in a sorted array using binary search",
            inputs=[
                IOSpec(name="data", type_desc="sorted list[comparable]"),
                IOSpec(name="target", type_desc="comparable"),
            ],
            outputs=[IOSpec(name="index", type_desc="int")],
            type_signature="list[T] -> T -> int",
        )
    )
    return catalog


def _make_skill_index():
    """No-op SkillIndex (search returns [])."""
    index = AsyncMock()
    index.search = lambda query, k=10: []
    return index


def _make_mock_llm(
    strategy_response: str | None = None,
    decompose_response: str | None = None,
    critique_response: str | None = None,
):
    """Create a mock LLMClient that routes responses by system prompt keywords."""
    if strategy_response is None:
        strategy_response = json.dumps(
            {
                "paradigm": "divide_and_conquer",
                "rationale": "Merge sort is a classic D&C algorithm",
                "variant_hint": "merge_sort",
            }
        )
    if decompose_response is None:
        decompose_response = json.dumps(
            {
                "sub_nodes": [
                    {
                        "name": "Split Input",
                        "description": "Split the input list into two halves",
                        "concept_type": "divide_and_conquer",
                        "inputs": [{"name": "data", "type_desc": "list[comparable]"}],
                        "outputs": [
                            {"name": "left", "type_desc": "list[comparable]"},
                            {"name": "right", "type_desc": "list[comparable]"},
                        ],
                        "type_signature": "",
                        "is_atomic": False,
                        "matched_primitive": None,
                    },
                    {
                        "name": "merge",
                        "description": "Merge two sorted lists into one sorted list",
                        "concept_type": "sorting",
                        "inputs": [
                            {"name": "left", "type_desc": "list[comparable]"},
                            {"name": "right", "type_desc": "list[comparable]"},
                        ],
                        "outputs": [
                            {"name": "result", "type_desc": "list[comparable]"}
                        ],
                        "type_signature": "list[T] -> list[T] -> list[T]",
                        "is_atomic": True,
                        "matched_primitive": "merge",
                    },
                ],
                "edges": [
                    {
                        "source_name": "Split Input",
                        "target_name": "merge",
                        "output_name": "left",
                        "input_name": "left",
                        "data_type": "list[comparable]",
                    },
                ],
            }
        )
    if critique_response is None:
        critique_response = json.dumps(
            {
                "approved": True,
                "reason": "Decomposition is correct and complete",
                "io_issues": [],
                "flagged_nodes": [],
            }
        )

    llm = AsyncMock()

    async def complete(system: str, user: str) -> str:
        system_lower = system.lower()
        if "critic" in system_lower or "evaluate" in system_lower:
            return critique_response
        elif "sub-nodes" in system_lower or "sub_nodes" in system_lower:
            return decompose_response
        elif "best" in system_lower and "paradigm" in system_lower:
            return strategy_response
        return "{}"

    llm.complete = complete
    return llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDecompositionState:
    """Test state TypedDict construction and custom _merge_nodes reducer."""

    def test_merge_nodes_deduplicates(self):
        """Latest entry per node_id wins."""
        node_v1 = AlgorithmicNode(
            node_id="n1",
            name="Test",
            description="v1",
            concept_type=ConceptType.SORTING,
            status=NodeStatus.PENDING,
        )
        node_v2 = AlgorithmicNode(
            node_id="n1",
            name="Test",
            description="v2",
            concept_type=ConceptType.SORTING,
            status=NodeStatus.DECOMPOSED,
        )
        other = AlgorithmicNode(
            node_id="n2",
            name="Other",
            description="other",
            concept_type=ConceptType.SORTING,
            status=NodeStatus.PENDING,
        )

        merged = _merge_nodes([node_v1, other], [node_v2])
        assert len(merged) == 2
        by_id = {n.node_id: n for n in merged}
        assert by_id["n1"].status == NodeStatus.DECOMPOSED
        assert by_id["n1"].description == "v2"
        assert by_id["n2"].status == NodeStatus.PENDING

    def test_merge_nodes_preserves_order(self):
        """Existing nodes not in updates are preserved."""
        n1 = AlgorithmicNode(
            node_id="a",
            name="A",
            description="a",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.PENDING,
        )
        n2 = AlgorithmicNode(
            node_id="b",
            name="B",
            description="b",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.PENDING,
        )

        merged = _merge_nodes([n1, n2], [])
        assert len(merged) == 2

    def test_merge_nodes_empty_existing(self):
        """Merging into empty list works."""
        n1 = AlgorithmicNode(
            node_id="a",
            name="A",
            description="a",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.PENDING,
        )
        merged = _merge_nodes([], [n1])
        assert len(merged) == 1
        assert merged[0].node_id == "a"


class TestDeterministicRewrite:
    def test_elides_validation_wrappers_when_substantive_steps_remain(self):
        catalog = PrimitiveCatalog()
        parent = AlgorithmicNode(
            node_id="parent_filter",
            name="Design Filter",
            description="Design a typed filter from requirements",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[IOSpec(name="spec", type_desc="filter specification")],
            outputs=[IOSpec(name="coefficients", type_desc="filter coefficients")],
            status=NodeStatus.PENDING,
            depth=1,
        )

        result = build_deterministic_decomposition(
            parsed={
                "sub_nodes": [
                    {
                        "name": "Validate Specification Feasibility",
                        "description": "Verify the requested design is valid.",
                    },
                    {
                        "name": "Choose Filter Strategy",
                        "description": "Choose a topology for the design.",
                    },
                    {
                        "name": "Synthesize Candidate Coefficients",
                        "description": "Generate coefficients from the chosen topology.",
                    },
                ]
            },
            parent=parent,
            catalog=catalog,
        )

        names = {node.name for node in result.nodes}
        assert "Validate Specification Feasibility" not in names
        assert "Choose Filter Strategy" in names
        assert "Synthesize Candidate Coefficients" in names

    def test_replenishes_after_validation_wrapper_removal(self):
        catalog = PrimitiveCatalog()
        parent = AlgorithmicNode(
            node_id="parent_generic",
            name="Prepare Result",
            description="Prepare a result from an input payload",
            concept_type=ConceptType.DATA_ASSEMBLY,
            inputs=[IOSpec(name="payload", type_desc="dict[str, Any]")],
            outputs=[IOSpec(name="result", type_desc="dict[str, Any]")],
            status=NodeStatus.PENDING,
            depth=1,
        )

        result = build_deterministic_decomposition(
            parsed={
                "sub_nodes": [
                    {
                        "name": "Validate Input Payload",
                        "description": "Check the payload before any work starts.",
                    },
                    {
                        "name": "Compute Core Transformation",
                        "description": "Perform the main transformation.",
                    },
                ]
            },
            parent=parent,
            catalog=catalog,
        )

        assert len(result.nodes) >= 2
        assert any(node.name == "Compute Core Transformation" for node in result.nodes)

    def test_preserves_detection_nodes_that_compute_boolean_results(self):
        catalog = PrimitiveCatalog()
        parent = AlgorithmicNode(
            node_id="parent_graph",
            name="Topological Sort",
            description="Produce a DAG ordering or detect a cycle",
            concept_type=ConceptType.GRAPH_TRAVERSAL,
            inputs=[IOSpec(name="graph", type_desc="dag")],
            outputs=[IOSpec(name="is_dag", type_desc="bool")],
            status=NodeStatus.PENDING,
            depth=1,
        )

        result = build_deterministic_decomposition(
            parsed={
                "sub_nodes": [
                    {
                        "name": "Compute In-Degree",
                        "description": "Compute in-degree of each node.",
                    },
                    {
                        "name": "Detect Cycle (DAG Check)",
                        "description": "Detect whether the graph violates DAG ordering.",
                        "outputs": [{"name": "is_dag", "type_desc": "bool"}],
                    },
                ]
            },
            parent=parent,
            catalog=catalog,
        )

        assert any(node.name == "Detect Cycle (DAG Check)" for node in result.nodes)

    def test_normalizes_near_match_concepts_to_builtin_primitives(self):
        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)
        parent = AlgorithmicNode(
            node_id="parent_response",
            name="Frequency Response",
            description="Compute the frequency response of a filter",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[IOSpec(name="valid_coefficients", type_desc="filter coefficients")],
            outputs=[IOSpec(name="response", type_desc="tuple[np.ndarray, np.ndarray]")],
            status=NodeStatus.PENDING,
            depth=1,
        )

        result = build_deterministic_decomposition(
            parsed={
                "sub_nodes": [
                    {
                        "name": "Normalize Coefficient Form",
                        "description": "Canonicalize coefficients for downstream analysis.",
                    },
                    {
                        "name": "Evaluate Complex Filter Response",
                        "description": "Compute the complex response across the frequency grid.",
                    },
                    {
                        "name": "Assemble Frequency Response Tuple",
                        "description": "Return the response as a typed tuple.",
                    },
                ]
            },
            parent=parent,
            catalog=catalog,
        )

        by_name = {node.name: node for node in result.nodes}
        assert by_name["Normalize Coefficient Form"].matched_primitive == "canonicalize_filter_coefficients"
        assert by_name["Evaluate Complex Filter Response"].matched_primitive == "compute_frequency_response"
        assert by_name["Assemble Frequency Response Tuple"].matched_primitive == "summarize_frequency_response"
        assert by_name["Normalize Coefficient Form"].primitive_binding_confidence >= 0.9
        assert by_name["Normalize Coefficient Form"].primitive_binding_source == "exact_name"
        assert any(action["stage"] == "primitive_normalization" for action in result.rewrite_actions)

    def test_normalizes_apply_filter_concept_without_exact_alias(self):
        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)
        parent = AlgorithmicNode(
            node_id="parent_apply",
            name="Apply Filter",
            description="Apply stable coefficients to a signal",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[
                IOSpec(name="valid_coefficients", type_desc="filter coefficients"),
                IOSpec(name="signal", type_desc="np.ndarray"),
            ],
            outputs=[IOSpec(name="filtered", type_desc="np.ndarray")],
            status=NodeStatus.PENDING,
            depth=1,
        )

        result = build_deterministic_decomposition(
            parsed={
                "sub_nodes": [
                    {
                        "name": "Apply Coefficients to Canonical Signal",
                        "description": "Run the filter coefficients across the sanitized signal array.",
                    }
                ]
            },
            parent=parent,
            catalog=catalog,
        )

        assert result.nodes
        assert result.nodes[0].matched_primitive == "apply_iir_filter"
        assert result.nodes[0].primitive_binding_confidence >= 0.7

    def test_weak_overlap_primitive_binding_stays_non_atomic(self):
        catalog = _make_catalog()
        parent = AlgorithmicNode(
            node_id="parent_sorting",
            name="Plan Search Step",
            description="Coordinate search behavior inside a larger sorting workflow",
            concept_type=ConceptType.SORTING,
            inputs=[IOSpec(name="items", type_desc="list[int]")],
            outputs=[IOSpec(name="index", type_desc="int")],
            status=NodeStatus.PENDING,
            depth=1,
        )

        result = build_deterministic_decomposition(
            parsed={
                "sub_nodes": [
                    {
                        "name": "Search Sorted Target",
                        "description": "Search the sorted target item.",
                    }
                ]
            },
            parent=parent,
            catalog=catalog,
        )

        assert result.nodes
        node = result.nodes[0]
        assert node.matched_primitive == "binary_search"
        assert node.primitive_binding_source == "token_overlap_cross_family"
        assert node.primitive_binding_confidence < 0.70
        assert node.status == NodeStatus.PENDING

    def test_cross_family_exact_name_binding_is_labeled_and_atomic(self):
        catalog = _make_catalog()
        parent = AlgorithmicNode(
            node_id="parent_sorting",
            name="Plan Search Step",
            description="Coordinate search behavior inside a larger sorting workflow",
            concept_type=ConceptType.SORTING,
            inputs=[
                IOSpec(name="data", type_desc="sorted list[comparable]"),
                IOSpec(name="target", type_desc="comparable"),
            ],
            outputs=[IOSpec(name="index", type_desc="int")],
            status=NodeStatus.PENDING,
            depth=1,
        )

        result = build_deterministic_decomposition(
            parsed={
                "sub_nodes": [
                    {
                        "name": "binary_search",
                        "description": "Search a sorted array for a target value.",
                    }
                ]
            },
            parent=parent,
            catalog=catalog,
        )

        assert result.nodes
        node = result.nodes[0]
        assert node.matched_primitive == "binary_search"
        assert node.primitive_binding_source == "exact_name_cross_family"
        assert node.primitive_binding_confidence >= 0.9
        assert node.status == NodeStatus.ATOMIC

    def test_collapses_routing_wrappers_when_work_nodes_exist(self):
        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)
        parent = AlgorithmicNode(
            node_id="parent_stability",
            name="Validate Stability",
            description="Validate coefficient stability",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[IOSpec(name="coefficients", type_desc="filter coefficients")],
            outputs=[IOSpec(name="valid_coefficients", type_desc="filter coefficients")],
            status=NodeStatus.PENDING,
            depth=1,
        )

        result = build_deterministic_decomposition(
            parsed={
                "sub_nodes": [
                    {
                        "name": "Compute Pole Locations",
                        "description": "Solve for the filter poles.",
                    },
                    {
                        "name": "Evaluate Discrete-Time Stability",
                        "description": "Assess the unit-circle stability margin.",
                    },
                    {
                        "name": "Pass Stable Coefficients",
                        "description": "Route the stable result to the next stage.",
                    },
                    {
                        "name": "Escalate Unstable Coefficients",
                        "description": "Escalate failures to a fallback branch.",
                    },
                ]
            },
            parent=parent,
            catalog=catalog,
        )

        names = {node.name for node in result.nodes}
        assert "Pass Stable Coefficients" not in names
        assert "Escalate Unstable Coefficients" not in names
        assert "Compute Pole Locations" in names
        assert "Evaluate Discrete-Time Stability" in names
        assert any(action["stage"] == "routing_wrapper_elision" for action in result.rewrite_actions)

    def test_rewrite_rejects_primitive_signature_violation_with_invariant_code(self):
        catalog = _make_catalog()
        parent = AlgorithmicNode(
            node_id="parent_search",
            name="Target Lookup",
            description="Search for a target in a sorted array",
            concept_type=ConceptType.SEARCHING,
            inputs=[
                IOSpec(name="data", type_desc="sorted list[int]"),
                IOSpec(name="target", type_desc="int"),
            ],
            outputs=[IOSpec(name="index", type_desc="int")],
            status=NodeStatus.PENDING,
            depth=1,
        )

        with pytest.raises(DeterministicRewriteError, match=r"\[primitive_signature_violation\]"):
            build_deterministic_decomposition(
                parsed={
                    "sub_nodes": [
                        {
                            "name": "binary_search",
                            "description": "Find the target index.",
                            "matched_primitive_hint": "binary_search",
                            "inputs": [
                                {"name": "data", "type_desc": "Any"},
                                {"name": "target", "type_desc": "Any"},
                            ],
                            "outputs": [
                                {"name": "index", "type_desc": "Any"},
                                {"name": "debug", "type_desc": "int"},
                            ],
                        }
                    ],
                },
                parent=parent,
                catalog=catalog,
            )

    def test_rewrite_rejects_disconnected_child_with_invariant_code(self):
        catalog = _make_catalog()
        parent = AlgorithmicNode(
            node_id="parent_graph",
            name="Graph Path",
            description="Connect one step into another",
            concept_type=ConceptType.SEARCHING,
            inputs=[IOSpec(name="data", type_desc="sorted list[comparable]")],
            outputs=[IOSpec(name="index", type_desc="int")],
            status=NodeStatus.PENDING,
            depth=1,
        )

        with pytest.raises(DeterministicRewriteError, match=r"\[disconnected_child\]"):
            build_deterministic_decomposition(
                parsed={
                    "sub_nodes": [
                        {
                            "name": "Prepare Search",
                            "description": "Prepare search inputs.",
                            "inputs": [{"name": "data", "type_desc": "sorted list[comparable]"}],
                            "outputs": [{"name": "prepared", "type_desc": "sorted list[comparable]"}],
                        },
                        {
                            "name": "binary_search",
                            "description": "Search sorted data for a target.",
                            "matched_primitive_hint": "binary_search",
                        },
                        {
                            "name": "Unused Step",
                            "description": "A detached child with no edges.",
                            "inputs": [{"name": "orphan", "type_desc": "int"}],
                            "outputs": [{"name": "dangling", "type_desc": "int"}],
                        },
                    ],
                    "flow_hints": [
                        {"from": "Prepare Search", "to": "binary_search", "why": "setup then search"}
                    ],
                },
                parent=parent,
                catalog=catalog,
            )

    def test_collapses_duplicate_concepts_that_bind_to_same_primitive(self):
        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)
        parent = AlgorithmicNode(
            node_id="parent_response",
            name="Frequency Response",
            description="Compute the frequency response of a filter",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[IOSpec(name="valid_coefficients", type_desc="filter coefficients")],
            outputs=[IOSpec(name="response", type_desc="tuple[np.ndarray, np.ndarray]")],
            status=NodeStatus.PENDING,
            depth=1,
        )

        result = build_deterministic_decomposition(
            parsed={
                "sub_nodes": [
                    {
                        "name": "Normalize Coefficient Form",
                        "description": "Canonicalize coefficients for downstream analysis.",
                    },
                    {
                        "name": "Canonicalize Coefficient Representation",
                        "description": "Normalize coefficient ordering and representation.",
                    },
                ]
            },
            parent=parent,
            catalog=catalog,
        )

        matches = [
            node for node in result.nodes
            if node.matched_primitive == "canonicalize_filter_coefficients"
        ]
        assert len(matches) == 1

    def test_synthesizes_helper_for_missing_required_primitive_input(self):
        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)
        parent = AlgorithmicNode(
            node_id="parent_response",
            name="Frequency Response",
            description="Compute the frequency response of a filter",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[IOSpec(name="valid_coefficients", type_desc="filter coefficients")],
            outputs=[IOSpec(name="response", type_desc="tuple[np.ndarray, np.ndarray]")],
            status=NodeStatus.PENDING,
            depth=1,
        )

        result = build_deterministic_decomposition(
            parsed={
                "sub_nodes": [
                    {
                        "name": "Evaluate Complex Filter Response",
                        "description": "Compute the complex response across the frequency grid.",
                    }
                ]
            },
            parent=parent,
            catalog=catalog,
        )

        names = {node.name for node in result.nodes}
        assert "Prepare Frequency Grid" in names
        compute_node = next(
            node for node in result.nodes
            if node.matched_primitive == "compute_frequency_response"
        )
        assert any(port.name == "frequency_grid" for port in compute_node.inputs)

    def test_repairs_any_ports_from_matched_primitive_signature(self):
        catalog = _make_catalog()
        parent = AlgorithmicNode(
            node_id="parent_search",
            name="Target Lookup",
            description="Search for a target in a sorted array",
            concept_type=ConceptType.SEARCHING,
            inputs=[
                IOSpec(name="data", type_desc="sorted list[int]"),
                IOSpec(name="target", type_desc="int"),
            ],
            outputs=[IOSpec(name="index", type_desc="int")],
            status=NodeStatus.PENDING,
            depth=1,
        )

        result = build_deterministic_decomposition(
            parsed={
                "sub_nodes": [
                    {
                        "name": "binary_search",
                        "description": "Find the target index.",
                        "matched_primitive_hint": "binary_search",
                        "inputs": [
                            {"name": "data", "type_desc": "Any"},
                            {"name": "target", "type_desc": "Any"},
                        ],
                        "outputs": [{"name": "index", "type_desc": "Any"}],
                    }
                ]
            },
            parent=parent,
            catalog=catalog,
        )

        node = result.nodes[0]
        assert [port.type_desc for port in node.inputs] == ["sorted list[comparable]", "comparable"]
        assert [port.type_desc for port in node.outputs] == ["int"]

    def test_does_not_append_parent_inputs_to_explicit_child_signature(self):
        catalog = PrimitiveCatalog()
        parent = AlgorithmicNode(
            node_id="parent_hodges",
            name="Hodges time-domain EMG onset detection",
            description="Detect onset in EMG",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[
                IOSpec(name="signal", type_desc="ndarray"),
                IOSpec(name="rest_signal", type_desc="ndarray"),
                IOSpec(name="sampling_rate", type_desc="float"),
                IOSpec(name="threshold", type_desc="float"),
            ],
            outputs=[IOSpec(name="onsets", type_desc="ndarray")],
            status=NodeStatus.PENDING,
            depth=1,
        )

        result = build_deterministic_decomposition(
            parsed={
                "sub_nodes": [
                    {
                        "name": "Estimate Rest Baseline Statistics",
                        "description": "Compute baseline statistics.",
                        "inputs": [
                            {"name": "rest_signal", "type_desc": "ndarray"},
                            {"name": "sampling_rate", "type_desc": "float"},
                        ],
                        "outputs": [
                            {"name": "rest_mean", "type_desc": "float"},
                            {"name": "rest_std", "type_desc": "float"},
                        ],
                    },
                    {
                        "name": "Threshold Crossing State Machine",
                        "description": "Emit onset indices.",
                        "inputs": [
                            {"name": "rest_std", "type_desc": "float"},
                            {"name": "threshold", "type_desc": "float"},
                        ],
                        "outputs": [{"name": "onsets", "type_desc": "ndarray"}],
                    },
                ]
            },
            parent=parent,
            catalog=catalog,
        )

        first = result.nodes[0]
        assert [port.name for port in first.inputs] == ["rest_signal", "sampling_rate"]

    def test_fails_fast_when_primitive_bound_node_violates_signature(self):
        catalog = _make_catalog()
        parent = AlgorithmicNode(
            node_id="parent_search",
            name="Target Lookup",
            description="Search for a target in a sorted array",
            concept_type=ConceptType.SEARCHING,
            inputs=[
                IOSpec(name="data", type_desc="sorted list[int]"),
                IOSpec(name="target", type_desc="int"),
            ],
            outputs=[IOSpec(name="index", type_desc="int")],
            status=NodeStatus.PENDING,
            depth=1,
        )

        with pytest.raises(
            DeterministicRewriteError,
            match=r"\[primitive_signature_violation\].*violates primitive signature",
        ):
            build_deterministic_decomposition(
                parsed={
                    "sub_nodes": [
                        {
                            "name": "binary_search",
                            "description": "Find the target index.",
                            "matched_primitive_hint": "binary_search",
                            "inputs": [
                                {"name": "data", "type_desc": "Any"},
                                {"name": "target", "type_desc": "Any"},
                            ],
                            "outputs": [
                                {"name": "index", "type_desc": "Any"},
                                {"name": "artifact", "type_desc": "Any"},
                            ],
                        }
                    ]
                },
                parent=parent,
                catalog=catalog,
            )


class TestSelectStrategy:
    """Test that select_strategy picks paradigm and populates pending queue."""

    @pytest.mark.asyncio
    async def test_picks_paradigm_and_instantiates_skeleton(self):
        from sciona.architect.nodes import select_strategy

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        llm = _make_mock_llm()

        state: DecompositionState = {
            "goal": "Implement merge sort",
            "max_depth": 8,
            "nodes": [],
            "edges": [],
            "history": [],
            "planning_artifact": None,
            "pending_node_ids": [],
            "current_node_id": "",
            "paradigm": "",
            "skeleton_instantiated": False,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }

        from sciona.architect.state import DecompositionDeps

        deps = DecompositionDeps(
            catalog=catalog,
            skill_index=skill_index,
            llm=llm,
            architect_critique_llm_enabled=True,
        )
        config = {"configurable": {"deps": deps}}

        result = await select_strategy(state, config)

        assert result["paradigm"] == "divide_and_conquer"
        assert result["skeleton_instantiated"] is True
        assert len(result["nodes"]) > 1  # root + skeleton nodes
        assert result["pending_node_ids"]  # at least some pending
        assert result["current_node_id"]  # first pending node selected
        assert result["planning_artifact"]["artifact_version"] == "phase1.v1"
        assert result["planning_artifact"]["skeleton_intent"]["variant_hint"] == "merge_sort"
        assert (
            result["planning_artifact"]["skeleton_intent"]["asset"]["asset_id"]
            == "family.divide_and_conquer.v1"
        )
        assert result["planning_artifact"]["planning_constraints"]
        assert result["skeleton_asset"]["asset_id"] == "family.divide_and_conquer.v1"

        # Root should be DECOMPOSED
        root = result["nodes"][0]
        assert root.status == NodeStatus.DECOMPOSED

    @pytest.mark.asyncio
    async def test_fallback_on_parse_error(self):
        """JSON parse failure falls back to CUSTOM paradigm."""
        from sciona.architect.nodes import select_strategy

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        llm = _make_mock_llm(strategy_response="not valid json at all")

        state: DecompositionState = {
            "goal": "Something unusual",
            "max_depth": 8,
            "nodes": [],
            "edges": [],
            "history": [],
            "planning_artifact": None,
            "pending_node_ids": [],
            "current_node_id": "",
            "paradigm": "",
            "skeleton_instantiated": False,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }

        from sciona.architect.state import DecompositionDeps

        deps = DecompositionDeps(
            catalog=catalog,
            skill_index=skill_index,
            llm=llm,
            architect_critique_llm_enabled=True,
        )
        config = {"configurable": {"deps": deps}}

        result = await select_strategy(state, config)

        assert result["paradigm"] == "custom"
        # No skeleton for CUSTOM, so only root node
        assert len(result["nodes"]) == 1
        assert result["pending_node_ids"] == [result["nodes"][0].node_id]
        assert result["current_node_id"] == result["nodes"][0].node_id
        assert result["done"] is False
        assert result["planning_artifact"]["artifact_version"] == "phase1.v1"
        assert result["planning_artifact"]["skeleton_intent"]["skeleton_instantiated"] is False

    @pytest.mark.asyncio
    async def test_signal_transform_skeleton_nodes_bind_to_builtin_primitives(self):
        from sciona.architect.nodes import select_strategy
        from sciona.architect.state import DecompositionDeps

        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)
        skill_index = _make_skill_index()
        llm = _make_mock_llm(
            strategy_response=json.dumps(
                {
                    "paradigm": "signal_transform",
                    "rationale": "Window, transform, process, and reconstruct the signal.",
                    "variant_hint": "fft_filter",
                }
            )
        )

        state: DecompositionState = {
            "goal": "Denoise a signal in the spectral domain",
            "max_depth": 8,
            "nodes": [],
            "edges": [],
            "history": [],
            "planning_artifact": None,
            "pending_node_ids": [],
            "current_node_id": "",
            "paradigm": "",
            "skeleton_instantiated": False,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }

        deps = DecompositionDeps(
            catalog=catalog,
            skill_index=skill_index,
            llm=llm,
            architect_critique_llm_enabled=True,
        )
        result = await select_strategy(state, {"configurable": {"deps": deps}})

        bound = {
            node.name: node.matched_primitive
            for node in result["nodes"]
            if node.status == NodeStatus.ATOMIC
        }

        assert bound["Window"] == "apply_window_function"

    @pytest.mark.asyncio
    async def test_signal_detect_measure_records_skeleton_asset_identity(self):
        from sciona.architect.nodes import select_strategy
        from sciona.architect.state import DecompositionDeps

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        llm = _make_mock_llm(
            strategy_response=json.dumps(
                {
                    "paradigm": "signal_filter",
                    "rationale": "This is a signal conditioning and event-rate problem.",
                    "variant_hint": "signal_detect_measure",
                }
            )
        )

        state: DecompositionState = {
            "goal": "Estimate event rate from a waveform",
            "max_depth": 8,
            "nodes": [],
            "edges": [],
            "history": [],
            "planning_artifact": None,
            "skeleton_asset": None,
            "pending_node_ids": [],
            "current_node_id": "",
            "paradigm": "",
            "skeleton_instantiated": False,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }

        deps = DecompositionDeps(
            catalog=catalog,
            skill_index=skill_index,
            llm=llm,
            architect_critique_llm_enabled=True,
        )

        result = await select_strategy(state, {"configurable": {"deps": deps}})

        assert result["skeleton_asset"]["asset_id"] == "signal_detect_measure"
        assert result["planning_artifact"]["skeleton_intent"]["asset"]["asset_version"] == "phase2.v1"
        assert result["planning_artifact"]["skeleton_intent"]["asset"]["source_kind"] == "local_asset"
        assert result["skeleton_instantiated"] is True


class TestRouteAfterCritic:
    """Test the 4 routing cases for route_after_critic."""

    def test_retry_on_failure_under_limit(self):
        from sciona.architect.nodes import route_after_critic

        state = {"critique_passed": False, "critique_retries": 1}
        assert route_after_critic(state) == "retry_decompose"

    def test_block_node_on_max_retries(self):
        from sciona.architect.nodes import route_after_critic

        state = {"critique_passed": False, "critique_retries": 3}
        assert route_after_critic(state) == "block_node"

    def test_next_node_on_pass(self):
        from sciona.architect.nodes import route_after_critic

        state = {"critique_passed": True, "critique_retries": 0}
        assert route_after_critic(state) == "next_node"

    def test_next_node_on_pass_with_retries(self):
        from sciona.architect.nodes import route_after_critic

        state = {"critique_passed": True, "critique_retries": 2}
        assert route_after_critic(state) == "next_node"


class TestRouteAfterAdvance:
    """Test route_after_advance routing."""

    def test_end_when_done(self):
        from sciona.architect.nodes import route_after_advance

        state = {"done": True, "pending_node_ids": []}
        assert route_after_advance(state) == "end"

    def test_end_when_no_pending(self):
        from sciona.architect.nodes import route_after_advance

        state = {"done": False, "pending_node_ids": []}
        assert route_after_advance(state) == "end"

    def test_decompose_when_pending(self):
        from sciona.architect.nodes import route_after_advance

        state = {"done": False, "pending_node_ids": ["n1"]}
        assert route_after_advance(state) == "decompose"


class TestDecompositionHappyPath:
    """Full cycle: strategy -> decompose -> critique (approved) -> END."""

    @pytest.mark.asyncio
    async def test_full_decomposition(self):
        from sciona.architect.graph import DecompositionAgent

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        llm = _make_mock_llm()

        agent = DecompositionAgent(
            catalog=catalog,
            skill_index=skill_index,
            llm=llm,
            max_depth=8,
        )

        cdg = await agent.decompose("Implement merge sort")

        # Should have nodes from skeleton + decomposition
        assert len(cdg.nodes) > 0
        assert len(cdg.edges) > 0

        # Should have some atomic leaf nodes
        atomic = [n for n in cdg.nodes if n.status == NodeStatus.ATOMIC]
        assert len(atomic) > 0

        # Rejected nodes should be filtered out
        rejected = [n for n in cdg.nodes if n.status == NodeStatus.REJECTED]
        assert len(rejected) == 0

        # Metadata should be populated
        assert cdg.metadata["goal"] == "Implement merge sort"
        assert cdg.metadata["paradigm"] == "divide_and_conquer"
        assert cdg.metadata["skeleton_asset"]["asset_id"] == "family.divide_and_conquer.v1"


class TestCritiqueRejection:
    """Critique rejects -> retry -> approve on second attempt."""

    @pytest.mark.asyncio
    async def test_retry_then_approve(self):
        from sciona.architect.graph import DecompositionAgent

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        goal = "Build a bespoke list reconciliation workflow"

        call_count = 0

        critique_reject = json.dumps(
            {
                "approved": False,
                "reason": "Missing edge between split and merge",
                "io_issues": ["No data flow from split to merge"],
                "flagged_nodes": [],
            }
        )
        critique_approve = json.dumps(
            {
                "approved": True,
                "reason": "Decomposition is now correct",
                "io_issues": [],
                "flagged_nodes": [],
            }
        )

        async def complete(system: str, user: str) -> str:
            nonlocal call_count
            system_lower = system.lower()
            if "critic" in system_lower or "evaluate" in system_lower:
                call_count += 1
                if call_count == 1:
                    return critique_reject
                return critique_approve
            elif "sub-nodes" in system_lower or "sub_nodes" in system_lower:
                return json.dumps(
                    {
                        "sub_nodes": [
                            {
                                "name": "Split",
                                "description": "Split input",
                                "concept_type": "divide_and_conquer",
                                "inputs": [{"name": "data", "type_desc": "list"}],
                                "outputs": [
                                    {"name": "left", "type_desc": "list"},
                                    {"name": "right", "type_desc": "list"},
                                ],
                                "is_atomic": False,
                                "matched_primitive": None,
                            },
                            {
                                "name": "merge",
                                "description": "Merge sorted lists",
                                "concept_type": "sorting",
                                "inputs": [
                                    {"name": "left", "type_desc": "list"},
                                    {"name": "right", "type_desc": "list"},
                                ],
                                "outputs": [{"name": "result", "type_desc": "list"}],
                                "is_atomic": True,
                                "matched_primitive": "merge",
                            },
                        ],
                        "edges": [
                            {
                                "source_name": "Split",
                                "target_name": "merge",
                                "output_name": "left",
                                "input_name": "left",
                                "data_type": "list",
                            },
                            {
                                "source_name": "Split",
                                "target_name": "merge",
                                "output_name": "right",
                                "input_name": "right",
                                "data_type": "list",
                            }
                        ],
                    }
                )
            elif "best" in system_lower and "paradigm" in system_lower:
                return json.dumps(
                    {
                        "paradigm": "custom",
                        "rationale": "Needs a bespoke decomposition",
                        "variant_hint": "",
                    }
                )
            return "{}"

        llm = AsyncMock()
        llm.complete = complete

        agent = DecompositionAgent(
            catalog=catalog,
            skill_index=skill_index,
            llm=llm,
            max_depth=8,
            architect_critique_llm_enabled=True,
        )

        cdg = await agent.decompose(goal)

        # Should eventually succeed after retry
        assert len(cdg.nodes) > 0
        # The critique was called at least twice
        assert call_count >= 2


class TestBlockedDecomposition:
    @pytest.mark.asyncio
    async def test_agent_metadata_marks_blocked_decomposition(self):
        from sciona.architect.graph import DecompositionAgent

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        llm = AsyncMock()

        async def complete(system: str, user: str) -> str:
            system_lower = system.lower()
            if "best" in system_lower and "paradigm" in system_lower:
                return json.dumps(
                    {
                        "paradigm": "divide_and_conquer",
                        "rationale": "D&C",
                        "variant_hint": "merge_sort",
                    }
                )
            if "critic" in system_lower or "evaluate" in system_lower:
                return json.dumps(
                    {
                        "approved": False,
                        "reason": "Typed flow is incomplete",
                        "io_issues": ["missing typed edge"],
                        "flagged_nodes": [],
                    }
                )
            return json.dumps(
                {
                    "sub_nodes": [
                        {
                            "name": "Split",
                            "description": "Split input",
                        },
                        {
                            "name": "Combine",
                            "description": "Combine partial results",
                        },
                    ]
                }
            )

        llm.complete = complete
        agent = DecompositionAgent(
            catalog=catalog,
            skill_index=skill_index,
            llm=llm,
            max_depth=8,
        )

        cdg = await agent.decompose("Implement merge sort")

        assert cdg.metadata["architect_status"] == "blocked"
        assert cdg.metadata["architect_error"]
        assert cdg.metadata["blocked_nodes"]
        assert any(node.status == NodeStatus.BLOCKED for node in cdg.nodes)

    @pytest.mark.asyncio
    async def test_heart_rate_goal_uses_skeleton_when_variant_matched(self):
        """When the LLM selects bandpass_hr_detection variant, the skeleton
        covers the entire goal — no decompose/critique loop needed."""
        from sciona.architect.graph import DecompositionAgent

        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)
        skill_index = _make_skill_index()
        llm = AsyncMock()

        async def complete(system: str, user: str) -> str:
            system_lower = system.lower()
            if "best" in system_lower and "paradigm" in system_lower:
                return json.dumps(
                    {
                        "paradigm": "signal_filter",
                        "rationale": "Filter and inspect the ECG trace before deriving heart rate.",
                        "variant_hint": "bandpass_hr_detection",
                    }
                )
            # Critic / decompose should not be called when skeleton
            # fully covers the goal, but provide fallback responses.
            if "critic" in system_lower or "evaluate" in system_lower:
                return json.dumps(
                    {
                        "approved": False,
                        "reason": "Typed ECG/filter flow remains incomplete.",
                        "io_issues": ["missing typed path to final output"],
                        "flagged_nodes": [],
                    }
                )
            return json.dumps(
                {
                    "sub_nodes": [
                        {
                            "name": "Parse Filter Requirements",
                            "description": "Extract ECG band constraints.",
                            "matched_primitive_hint": "parse_filter_spec",
                        },
                        {
                            "name": "Select Filter Family",
                            "description": "Choose the topology.",
                            "matched_primitive_hint": "choose_filter_topology",
                        },
                        {
                            "name": "Synthesize Candidate Coefficients",
                            "description": "Generate candidate coefficients.",
                            "matched_primitive_hint": "design_filter_coefficients",
                        },
                    ]
                }
            )

        llm.complete = complete
        agent = DecompositionAgent(
            catalog=catalog,
            skill_index=skill_index,
            llm=llm,
            max_depth=8,
        )

        cdg = await agent.decompose("Detect heart rate from raw ECG signal")

        assert cdg.metadata["architect_status"] == "ready"
        assert "Detect heart rate from raw ECG signal" == cdg.metadata["goal"]
        assert cdg.metadata["paradigm"] == "signal_filter"
        # Skeleton instantiated all nodes — no pending decompositions needed
        assert len(cdg.nodes) >= 2  # root + at least one child


class TestMaxDepth:
    """Depth violation caught by deterministic critique check."""

    @pytest.mark.asyncio
    async def test_depth_violation_rejected(self):
        from sciona.architect.nodes import critique_decomposition
        from sciona.architect.state import DecompositionDeps

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        llm = _make_mock_llm()

        parent = AlgorithmicNode(
            node_id="parent1",
            name="Deep Node",
            description="A node at max depth",
            concept_type=ConceptType.SORTING,
            depth=3,
            status=NodeStatus.PENDING,
        )
        # Children that exceed max_depth=3
        child = AlgorithmicNode(
            node_id="child1",
            parent_id="parent1",
            name="Too Deep",
            description="This node is too deep",
            concept_type=ConceptType.SORTING,
            depth=4,
            status=NodeStatus.PENDING,
        )
        child2 = AlgorithmicNode(
            node_id="child2",
            parent_id="parent1",
            name="Also Deep",
            description="Also too deep",
            concept_type=ConceptType.SORTING,
            depth=4,
            status=NodeStatus.PENDING,
        )

        state: DecompositionState = {
            "goal": "test",
            "max_depth": 3,
            "nodes": [parent, child, child2],
            "edges": [],
            "history": [],
            "pending_node_ids": ["parent1"],
            "current_node_id": "parent1",
            "paradigm": "sorting",
            "skeleton_instantiated": True,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }

        deps = DecompositionDeps(catalog=catalog, skill_index=skill_index, llm=llm)
        config = {"configurable": {"deps": deps}}

        result = await critique_decomposition(state, config)

        assert result["critique_passed"] is False
        assert (
            "max depth" in result["critique_reason"].lower()
            or "depth" in result["critique_reason"].lower()
        )


class TestCritiqueHardening:
    """Hardening behaviors for malformed critique payloads and retries."""

    @pytest.mark.asyncio
    async def test_deterministic_only_critique_skips_llm_when_disabled(self):
        from sciona.architect.nodes import critique_decomposition
        from sciona.architect.state import DecompositionDeps

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        llm = AsyncMock()

        parent = AlgorithmicNode(
            node_id="parent",
            name="Filter Signal",
            description="Apply a filter to a signal.",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
            outputs=[IOSpec(name="filtered", type_desc="np.ndarray")],
            status=NodeStatus.PENDING,
            depth=1,
        )
        child1 = AlgorithmicNode(
            node_id="c1",
            parent_id="parent",
            name="Design Filter",
            description="Design coefficients.",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
            outputs=[IOSpec(name="coeffs", type_desc="np.ndarray")],
            status=NodeStatus.PENDING,
            depth=2,
        )
        child2 = AlgorithmicNode(
            node_id="c2",
            parent_id="parent",
            name="Apply Filter",
            description="Filter the signal.",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[
                IOSpec(name="signal", type_desc="np.ndarray"),
                IOSpec(name="coeffs", type_desc="np.ndarray"),
            ],
            outputs=[IOSpec(name="filtered", type_desc="np.ndarray")],
            status=NodeStatus.PENDING,
            depth=2,
        )

        state: DecompositionState = {
            "goal": "filter",
            "max_depth": 8,
            "nodes": [parent, child1, child2],
            "edges": [
                DependencyEdge(
                    source_id="c1",
                    target_id="c2",
                    output_name="coeffs",
                    input_name="coeffs",
                    source_type="np.ndarray",
                    target_type="np.ndarray",
                )
            ],
            "history": [],
            "pending_node_ids": [],
            "current_node_id": "parent",
            "paradigm": "signal_filter",
            "skeleton_instantiated": True,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }

        deps = DecompositionDeps(
            catalog=catalog,
            skill_index=skill_index,
            llm=llm,
            architect_critique_llm_enabled=False,
        )
        result = await critique_decomposition(state, {"configurable": {"deps": deps}})

        assert result["critique_passed"] is True
        assert "deterministic structural checks passed" in result["critique_reason"].lower()
        assert llm.complete.await_count == 0

    @pytest.mark.asyncio
    async def test_malformed_critique_schema_fails_open(self):
        from sciona.architect.nodes import critique_decomposition
        from sciona.architect.state import DecompositionDeps

        catalog = _make_catalog()
        skill_index = _make_skill_index()

        llm = AsyncMock()

        async def complete(system: str, user: str) -> str:
            # Valid JSON but wrong shape for a critique response.
            return json.dumps({"sub_nodes": [], "edges": []})

        llm.complete = complete

        parent = AlgorithmicNode(
            node_id="parent",
            name="Parent",
            description="Parent node",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.PENDING,
            depth=1,
        )
        child1 = AlgorithmicNode(
            node_id="c1",
            parent_id="parent",
            name="Child 1",
            description="child 1",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.PENDING,
            depth=2,
        )
        child2 = AlgorithmicNode(
            node_id="c2",
            parent_id="parent",
            name="Child 2",
            description="child 2",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.PENDING,
            depth=2,
        )

        state: DecompositionState = {
            "goal": "test",
            "max_depth": 8,
            "nodes": [parent, child1, child2],
            "edges": [],
            "history": [],
            "pending_node_ids": ["parent"],
            "current_node_id": "parent",
            "paradigm": "custom",
            "skeleton_instantiated": True,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }

        deps = DecompositionDeps(
            catalog=catalog,
            skill_index=skill_index,
            llm=llm,
            architect_critique_llm_enabled=True,
        )
        config = {"configurable": {"deps": deps}}
        result = await critique_decomposition(state, config)

        assert result["critique_passed"] is True
        assert "invalid schema" in result["critique_reason"].lower()

    @pytest.mark.asyncio
    async def test_deterministic_critique_rejects_uncovered_parent_outputs(self):
        from sciona.architect.nodes import critique_decomposition
        from sciona.architect.state import DecompositionDeps

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        llm = AsyncMock()

        parent = AlgorithmicNode(
            node_id="parent",
            name="Assemble Result",
            description="Produce a final result from input data.",
            concept_type=ConceptType.CUSTOM,
            inputs=[IOSpec(name="data", type_desc="DataFrame")],
            outputs=[IOSpec(name="result", type_desc="Report")],
            status=NodeStatus.PENDING,
            depth=1,
        )
        child1 = AlgorithmicNode(
            node_id="c1",
            parent_id="parent",
            name="Validate Input",
            description="Inspect the input data.",
            concept_type=ConceptType.CUSTOM,
            inputs=[IOSpec(name="data", type_desc="DataFrame")],
            outputs=[IOSpec(name="validated", type_desc="DataFrame")],
            status=NodeStatus.PENDING,
            depth=2,
        )
        child2 = AlgorithmicNode(
            node_id="c2",
            parent_id="parent",
            name="Summarize",
            description="Generate diagnostics.",
            concept_type=ConceptType.CUSTOM,
            inputs=[IOSpec(name="validated", type_desc="DataFrame")],
            outputs=[IOSpec(name="summary", type_desc="dict")],
            status=NodeStatus.PENDING,
            depth=2,
        )

        state: DecompositionState = {
            "goal": "summarize",
            "max_depth": 8,
            "nodes": [parent, child1, child2],
            "edges": [],
            "history": [],
            "pending_node_ids": [],
            "current_node_id": "parent",
            "paradigm": "custom",
            "skeleton_instantiated": True,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }

        deps = DecompositionDeps(catalog=catalog, skill_index=skill_index, llm=llm)
        result = await critique_decomposition(state, {"configurable": {"deps": deps}})

        assert result["critique_passed"] is False
        assert "parent outputs not produced" in result["critique_reason"].lower()

    @pytest.mark.asyncio
    async def test_deterministic_critique_rejects_duplicate_children(self):
        from sciona.architect.nodes import critique_decomposition
        from sciona.architect.state import DecompositionDeps

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        llm = AsyncMock()

        parent = AlgorithmicNode(
            node_id="parent",
            name="Prepare Data",
            description="Prepare the data for use.",
            concept_type=ConceptType.CUSTOM,
            inputs=[IOSpec(name="data", type_desc="table")],
            outputs=[IOSpec(name="prepared", type_desc="table")],
            status=NodeStatus.PENDING,
            depth=1,
        )
        child1 = AlgorithmicNode(
            node_id="c1",
            parent_id="parent",
            name="Normalize Data",
            description="Normalize the data.",
            concept_type=ConceptType.CUSTOM,
            inputs=[IOSpec(name="data", type_desc="table")],
            outputs=[IOSpec(name="prepared", type_desc="table")],
            status=NodeStatus.PENDING,
            depth=2,
        )
        child2 = AlgorithmicNode(
            node_id="c2",
            parent_id="parent",
            name="Normalize Data",
            description="Normalize the data again.",
            concept_type=ConceptType.CUSTOM,
            inputs=[IOSpec(name="data", type_desc="table")],
            outputs=[IOSpec(name="prepared", type_desc="table")],
            status=NodeStatus.PENDING,
            depth=2,
        )

        state: DecompositionState = {
            "goal": "prepare data",
            "max_depth": 8,
            "nodes": [parent, child1, child2],
            "edges": [],
            "history": [],
            "pending_node_ids": [],
            "current_node_id": "parent",
            "paradigm": "custom",
            "skeleton_instantiated": True,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }

        deps = DecompositionDeps(catalog=catalog, skill_index=skill_index, llm=llm)
        result = await critique_decomposition(state, {"configurable": {"deps": deps}})

        assert result["critique_passed"] is False
        assert "near-duplicate child nodes" in result["critique_reason"].lower()

    @pytest.mark.asyncio
    async def test_prepare_retry_rejects_prior_atomic_children(self):
        from sciona.architect.nodes import prepare_retry

        parent = AlgorithmicNode(
            node_id="p",
            name="Parent",
            description="parent",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.PENDING,
            depth=0,
        )
        child_atomic = AlgorithmicNode(
            node_id="a",
            parent_id="p",
            name="Atomic Child",
            description="atomic",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.ATOMIC,
            depth=1,
        )
        child_pending = AlgorithmicNode(
            node_id="b",
            parent_id="p",
            name="Pending Child",
            description="pending",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.PENDING,
            depth=1,
        )

        state: DecompositionState = {
            "goal": "test",
            "max_depth": 8,
            "nodes": [parent, child_atomic, child_pending],
            "edges": [],
            "history": [],
            "pending_node_ids": ["p"],
            "current_node_id": "p",
            "paradigm": "custom",
            "skeleton_instantiated": True,
            "critique_passed": False,
            "critique_reason": "bad decomposition",
            "critique_retries": 1,
            "done": False,
            "error": "",
        }

        result = await prepare_retry(state, {"configurable": {"deps": None}})
        updated = result["nodes"]

        assert len(updated) == 2
        assert all(n.status == NodeStatus.REJECTED for n in updated)
        assert result["critique_retries"] == 2

    @pytest.mark.asyncio
    async def test_block_node_discards_descendants_and_sets_error(self):
        from sciona.architect.nodes import block_node

        parent = AlgorithmicNode(
            node_id="p",
            name="Parent",
            description="parent",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.PENDING,
            depth=0,
        )
        child = AlgorithmicNode(
            node_id="c",
            parent_id="p",
            name="Child",
            description="child",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.PENDING,
            depth=1,
        )
        grandchild = AlgorithmicNode(
            node_id="g",
            parent_id="c",
            name="Grandchild",
            description="grandchild",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.PENDING,
            depth=2,
        )

        state: DecompositionState = {
            "goal": "test",
            "max_depth": 8,
            "nodes": [parent, child, grandchild],
            "edges": [],
            "history": [],
            "pending_node_ids": ["p"],
            "current_node_id": "p",
            "paradigm": "custom",
            "skeleton_instantiated": True,
            "critique_passed": False,
            "critique_reason": "too many retries",
            "critique_retries": 3,
            "done": False,
            "error": "",
        }

        result = await block_node(state, {"configurable": {"deps": None}})
        by_id = {node.node_id: node for node in result["nodes"]}

        assert result["done"] is True
        assert "blocked" in result["error"].lower()
        assert result["pending_node_ids"] == []
        assert by_id["p"].status == NodeStatus.BLOCKED
        assert by_id["c"].status == NodeStatus.REJECTED
        assert by_id["g"].status == NodeStatus.REJECTED


class TestSharedContext:
    @pytest.mark.asyncio
    async def test_select_strategy_injects_and_writes_shared_context(self):
        from sciona.architect.nodes import select_strategy
        from sciona.architect.state import DecompositionDeps

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        llm = AsyncMock()
        captured_users: list[str] = []

        async def complete(system: str, user: str) -> str:
            captured_users.append(user)
            return json.dumps(
                {
                    "paradigm": "divide_and_conquer",
                    "rationale": "standard choice",
                    "variant_hint": "merge_sort",
                }
            )

        llm.complete = complete
        store = InMemorySharedContextStore()
        await store.put(
            "architect/test/strategy",
            "Prior goal 'sort values' worked with divide_and_conquer",
        )

        state: DecompositionState = {
            "goal": "Implement merge sort",
            "max_depth": 8,
            "nodes": [],
            "edges": [],
            "history": [],
            "pending_node_ids": [],
            "current_node_id": "",
            "paradigm": "",
            "skeleton_instantiated": False,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }
        deps = DecompositionDeps(
            catalog=catalog,
            skill_index=skill_index,
            llm=llm,
            shared_context=store,
            context_namespace="architect/test",
        )
        config = {"configurable": {"deps": deps}}

        await select_strategy(state, config)

        assert captured_users
        assert "Shared Context" in captured_users[0]
        records = await store.recent("architect/test/strategy", limit=5)
        assert any("Paradigm: divide_and_conquer" in r.text for r in records)

    @pytest.mark.asyncio
    async def test_decompose_node_injects_and_writes_shared_context(self):
        from sciona.architect.nodes import decompose_node
        from sciona.architect.state import DecompositionDeps

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        llm = AsyncMock()
        captured_users: list[str] = []

        async def complete(system: str, user: str) -> str:
            captured_users.append(user)
            return json.dumps(
                {
                    "sub_nodes": [
                        {
                            "name": "Split",
                            "description": "Split list into two halves",
                            "concept_type": "divide_and_conquer",
                            "inputs": [{"name": "data", "type_desc": "list[int]"}],
                            "outputs": [
                                {"name": "left", "type_desc": "list[int]"},
                                {"name": "right", "type_desc": "list[int]"},
                            ],
                            "is_atomic": False,
                            "matched_primitive": None,
                        },
                        {
                            "name": "merge",
                            "description": "Merge sorted halves",
                            "concept_type": "sorting",
                            "inputs": [
                                {"name": "left", "type_desc": "list[int]"},
                                {"name": "right", "type_desc": "list[int]"},
                            ],
                            "outputs": [{"name": "result", "type_desc": "list[int]"}],
                            "is_atomic": True,
                            "matched_primitive": "merge",
                        },
                    ],
                    "edges": [
                        {
                            "source_name": "Split",
                            "target_name": "merge",
                            "output_name": "left",
                            "input_name": "left",
                            "data_type": "list[int]",
                        }
                    ],
                }
            )

        llm.complete = complete
        store = InMemorySharedContextStore()
        await store.put(
            "architect/test/decompose",
            "For merge sort, ensure split feeds merge via left/right outputs.",
        )

        parent = AlgorithmicNode(
            node_id="n_parent",
            name="Sort",
            description="Sort input list",
            concept_type=ConceptType.DIVIDE_AND_CONQUER,
            inputs=[IOSpec(name="data", type_desc="list[int]")],
            outputs=[IOSpec(name="result", type_desc="list[int]")],
            status=NodeStatus.PENDING,
            depth=1,
        )
        state: DecompositionState = {
            "goal": "Implement merge sort",
            "max_depth": 8,
            "nodes": [parent],
            "edges": [],
            "history": [],
            "pending_node_ids": ["n_parent"],
            "current_node_id": "n_parent",
            "paradigm": "divide_and_conquer",
            "skeleton_instantiated": True,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }
        deps = DecompositionDeps(
            catalog=catalog,
            skill_index=skill_index,
            llm=llm,
            shared_context=store,
            context_namespace="architect/test",
        )
        config = {"configurable": {"deps": deps}}

        result = await decompose_node(state, config)

        assert result["nodes"]
        assert result["history"][0]["primitive_proposal_count"] >= 1
        assert result["history"][0]["template_proposal_count"] == 0
        assert result["history"][0]["skeleton_proposal_count"] >= 0
        assert result["history"][0]["top_ranked_proposal_type"] in {
            "primitive",
            "template",
            "skeleton",
            "",
        }
        assert captured_users
        assert "Shared Context" in captured_users[0]
        records = await store.recent("architect/test/decompose", limit=5)
        assert any("Parent: Sort" in r.text for r in records)

    @pytest.mark.asyncio
    async def test_decompose_node_surfaces_passive_skeleton_proposal_count(self):
        from sciona.architect.nodes import decompose_node
        from sciona.architect.state import DecompositionDeps

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        llm = AsyncMock()

        async def complete(system: str, user: str) -> str:
            return json.dumps(
                {
                    "sub_nodes": [
                        {
                            "name": "Filter Signal",
                            "description": "Filter signal before downstream processing.",
                            "concept_type": "signal_filter",
                            "inputs": [{"name": "signal", "type_desc": "any"}],
                            "outputs": [{"name": "filtered_signal", "type_desc": "any"}],
                        }
                    ]
                }
            )

        llm.complete = complete
        parent = AlgorithmicNode(
            node_id="n_parent",
            name="Estimate Event Rate",
            description="Estimate a rate from a signal stream.",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[
                IOSpec(name="signal", type_desc="any"),
                IOSpec(name="sampling_rate", type_desc="any"),
            ],
            outputs=[IOSpec(name="result", type_desc="any")],
            status=NodeStatus.PENDING,
            depth=1,
        )
        state: DecompositionState = {
            "goal": "Estimate event rate from a signal",
            "max_depth": 8,
            "nodes": [parent],
            "edges": [],
            "history": [],
            "pending_node_ids": ["n_parent"],
            "current_node_id": "n_parent",
            "paradigm": "signal_filter",
            "skeleton_instantiated": True,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }
        deps = DecompositionDeps(catalog=catalog, skill_index=skill_index, llm=llm)

        result = await decompose_node(state, {"configurable": {"deps": deps}})

        assert result["nodes"]
        assert result["history"][0]["primitive_proposal_count"] >= 1
        assert result["history"][0]["skeleton_proposal_count"] >= 1
        assert result["history"][0]["top_ranked_proposal_type"] in {
            "primitive",
            "template",
            "skeleton",
        }
        assert result["history"][0]["ranked_proposal_types"]
        assert "skeleton_acceptance_reason" in result["history"][0]
        assert "skeleton_acceptance_margin" in result["history"][0]

    @pytest.mark.asyncio
    async def test_decompose_node_injects_template_context_from_shared_namespace(self):
        from sciona.architect.nodes import decompose_node
        from sciona.architect.state import DecompositionDeps

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        llm = AsyncMock()
        captured_users: list[str] = []

        async def complete(system: str, user: str) -> str:
            captured_users.append(user)
            return json.dumps(
                {
                    "sub_nodes": [
                        {
                            "name": "Split",
                            "description": "Split list into halves.",
                        },
                        {
                            "name": "merge",
                            "description": "Merge sorted halves.",
                            "matched_primitive_hint": "merge",
                        },
                    ]
                }
            )

        llm.complete = complete
        store = InMemorySharedContextStore()
        await store.put(
            "architect/templates",
            "Parent: Sort\nDescription: Sort input list\nChildren: Split, merge [merge]\nOutputs: result",
        )

        parent = AlgorithmicNode(
            node_id="n_parent",
            name="Sort",
            description="Sort input list",
            concept_type=ConceptType.DIVIDE_AND_CONQUER,
            inputs=[IOSpec(name="data", type_desc="list[int]")],
            outputs=[IOSpec(name="result", type_desc="list[int]")],
            status=NodeStatus.PENDING,
            depth=1,
        )
        state: DecompositionState = {
            "goal": "Implement merge sort",
            "max_depth": 8,
            "nodes": [parent],
            "edges": [],
            "history": [],
            "pending_node_ids": ["n_parent"],
            "current_node_id": "n_parent",
            "paradigm": "divide_and_conquer",
            "skeleton_instantiated": True,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }
        deps = DecompositionDeps(
            catalog=catalog,
            skill_index=skill_index,
            llm=llm,
            shared_context=store,
            shared_context_metrics=SharedContextMetrics(),
            context_namespace="architect/test",
        )
        config = {"configurable": {"deps": deps}}

        await decompose_node(state, config)

        assert captured_users
        assert "Prior Decomposition Templates" in captured_users[0]
        snap = deps.shared_context_metrics.snapshot()
        assert snap["template_searches_total"] == 1
        assert snap["template_search_hits"] == 1
        assert snap["template_injected_blocks"] == 1

    @pytest.mark.asyncio
    async def test_decompose_prompt_requests_conceptual_structure_only(self):
        from sciona.architect.nodes import decompose_node
        from sciona.architect.state import DecompositionDeps

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        llm = AsyncMock()
        captured_users: list[str] = []

        async def complete(system: str, user: str) -> str:
            captured_users.append(user)
            return json.dumps(
                {
                    "sub_nodes": [
                        {"name": "Split Input", "description": "Partition the input."},
                        {"name": "merge", "description": "Combine sorted halves."},
                    ]
                }
            )

        llm.complete = complete
        parent = AlgorithmicNode(
            node_id="n_parent",
            name="Sort",
            description="Sort input list",
            concept_type=ConceptType.DIVIDE_AND_CONQUER,
            inputs=[IOSpec(name="data", type_desc="list[int]")],
            outputs=[IOSpec(name="result", type_desc="list[int]")],
            status=NodeStatus.PENDING,
            depth=1,
        )
        state: DecompositionState = {
            "goal": "Implement merge sort",
            "max_depth": 8,
            "nodes": [parent],
            "edges": [],
            "history": [],
            "pending_node_ids": ["n_parent"],
            "current_node_id": "n_parent",
            "paradigm": "divide_and_conquer",
            "skeleton_instantiated": True,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }
        deps = DecompositionDeps(catalog=catalog, skill_index=skill_index, llm=llm)

        await decompose_node(state, {"configurable": {"deps": deps}})

        assert captured_users
        assert "Do not emit `inputs`, `outputs`, `type_signature`, `is_atomic`, or explicit `edges`." in captured_users[0]

    @pytest.mark.asyncio
    async def test_decompose_node_injects_failure_context_from_shared_namespace(self):
        from sciona.architect.nodes import decompose_node
        from sciona.architect.state import DecompositionDeps
        from sciona.shared_context import SharedContextMetrics

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        llm = AsyncMock()
        captured_users: list[str] = []
        metrics = SharedContextMetrics()

        async def complete(system: str, user: str) -> str:
            captured_users.append(user)
            return json.dumps(
                {
                    "sub_nodes": [
                        {"name": "Split", "description": "Split list into halves."},
                        {
                            "name": "merge",
                            "description": "Merge sorted halves.",
                            "matched_primitive_hint": "merge",
                        },
                    ]
                }
            )

        llm.complete = complete
        store = InMemorySharedContextStore()
        await store.put(
            "architect/failures",
            "Parent: Sort\nDescription: Sort input list\nCategory: semantic_completeness\nReason: Missing typed path to result",
        )

        parent = AlgorithmicNode(
            node_id="n_parent",
            name="Sort",
            description="Sort input list",
            concept_type=ConceptType.DIVIDE_AND_CONQUER,
            inputs=[IOSpec(name="data", type_desc="list[int]")],
            outputs=[IOSpec(name="result", type_desc="list[int]")],
            status=NodeStatus.PENDING,
            depth=1,
        )
        state: DecompositionState = {
            "goal": "Implement merge sort",
            "max_depth": 8,
            "nodes": [parent],
            "edges": [],
            "history": [],
            "pending_node_ids": ["n_parent"],
            "current_node_id": "n_parent",
            "paradigm": "divide_and_conquer",
            "skeleton_instantiated": True,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }
        deps = DecompositionDeps(
            catalog=catalog,
            skill_index=skill_index,
            llm=llm,
            shared_context=store,
            shared_context_metrics=metrics,
            context_namespace="architect/test",
        )

        await decompose_node(state, {"configurable": {"deps": deps}})

        assert captured_users
        assert "Prior Failure Patterns" in captured_users[0]
        snap = metrics.snapshot()
        assert snap["failure_searches_total"] == 1
        assert snap["failure_search_hits"] == 1
        assert snap["failure_injected_blocks"] == 1

    @pytest.mark.asyncio
    async def test_critique_rejection_writes_failure_context_to_shared_namespace(self):
        from sciona.architect.nodes import critique_decomposition
        from sciona.architect.state import DecompositionDeps
        from sciona.shared_context import SharedContextMetrics

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        llm = AsyncMock()
        store = InMemorySharedContextStore()
        metrics = SharedContextMetrics()

        parent = AlgorithmicNode(
            node_id="n_parent",
            name="Sort",
            description="Sort input list",
            concept_type=ConceptType.DIVIDE_AND_CONQUER,
            inputs=[IOSpec(name="data", type_desc="list[int]")],
            outputs=[IOSpec(name="result", type_desc="list[int]")],
            status=NodeStatus.PENDING,
            depth=1,
        )
        child = AlgorithmicNode(
            node_id="n_child",
            parent_id="n_parent",
            name="Candidate Step",
            description="A badly typed child step.",
            concept_type=ConceptType.DIVIDE_AND_CONQUER,
            inputs=[IOSpec(name="data", type_desc="Any")],
            outputs=[IOSpec(name="result", type_desc="Any")],
            status=NodeStatus.PENDING,
            depth=2,
        )
        state: DecompositionState = {
            "goal": "Implement merge sort",
            "max_depth": 8,
            "nodes": [parent, child],
            "edges": [],
            "history": [],
            "pending_node_ids": ["n_parent"],
            "current_node_id": "n_parent",
            "paradigm": "divide_and_conquer",
            "skeleton_instantiated": True,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }
        deps = DecompositionDeps(
            catalog=catalog,
            skill_index=skill_index,
            llm=llm,
            shared_context=store,
            shared_context_metrics=metrics,
            context_namespace="architect/test",
        )

        result = await critique_decomposition(state, {"configurable": {"deps": deps}})

        assert result["critique_passed"] is False
        records = await store.recent("architect/failures", limit=5)
        assert any("Parent: Sort" in r.text for r in records)
        snap = metrics.snapshot()
        assert snap["failure_puts_total"] == 1


class TestDeterministicDecompose:
    @pytest.mark.asyncio
    async def test_conceptual_payload_gets_deterministic_ports_and_edges(self):
        from sciona.architect.nodes import decompose_node
        from sciona.architect.state import DecompositionDeps

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        llm = AsyncMock()

        async def complete(system: str, user: str) -> str:
            return json.dumps(
                {
                    "progress_updates": [
                        "identify high-level phases",
                        "map phase order",
                    ],
                    "sub_nodes": [
                        {
                            "name": "Split Input",
                            "description": "Split list into halves for recursive processing.",
                        },
                        {
                            "name": "merge",
                            "description": "Merge sorted halves.",
                            "matched_primitive_hint": "merge",
                        },
                    ],
                    "flow_hints": [
                        {
                            "from": "Split Input",
                            "to": "merge",
                            "why": "split output feeds merge",
                        }
                    ],
                }
            )

        llm.complete = complete

        parent = AlgorithmicNode(
            node_id="n_parent",
            name="Sort",
            description="Sort input list",
            concept_type=ConceptType.DIVIDE_AND_CONQUER,
            inputs=[IOSpec(name="data", type_desc="list[int]")],
            outputs=[IOSpec(name="result", type_desc="list[int]")],
            status=NodeStatus.PENDING,
            depth=1,
        )
        state: DecompositionState = {
            "goal": "Implement merge sort",
            "max_depth": 8,
            "nodes": [parent],
            "edges": [],
            "history": [],
            "pending_node_ids": ["n_parent"],
            "current_node_id": "n_parent",
            "paradigm": "divide_and_conquer",
            "skeleton_instantiated": True,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }
        deps = DecompositionDeps(catalog=catalog, skill_index=skill_index, llm=llm)
        config = {"configurable": {"deps": deps}}

        result = await decompose_node(state, config)
        nodes = result["nodes"]
        edges = result["edges"]
        assert len(nodes) == 2
        assert edges

        by_name = {n.name: n for n in nodes}
        assert by_name["Split Input"].inputs
        assert by_name["Split Input"].outputs
        assert by_name["Split Input"].type_signature

        merge_node = by_name["merge"]
        assert merge_node.status == NodeStatus.ATOMIC
        assert merge_node.matched_primitive == "merge"
        assert merge_node.type_signature

        src_id = by_name["Split Input"].node_id
        tgt_id = by_name["merge"].node_id
        assert any(e.source_id == src_id and e.target_id == tgt_id for e in edges)

    @pytest.mark.asyncio
    async def test_signal_filter_uses_deterministic_fallback_steps(self):
        from sciona.architect.nodes import decompose_node
        from sciona.architect.state import DecompositionDeps

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        llm = AsyncMock()

        async def complete(system: str, user: str) -> str:
            # Intentionally under-specified: deterministic fallback should expand this.
            return json.dumps(
                {
                    "sub_nodes": [
                        {
                            "name": "Design Core",
                            "description": "Compute coefficients from the filter spec.",
                        }
                    ],
                    "flow_hints": [],
                }
            )

        llm.complete = complete
        parent = AlgorithmicNode(
            node_id="n_filter",
            name="Design Filter",
            description="Design filter coefficients from specification",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[IOSpec(name="spec", type_desc="filter_spec")],
            outputs=[IOSpec(name="coefficients", type_desc="vector[float]")],
            status=NodeStatus.PENDING,
            depth=1,
        )
        state: DecompositionState = {
            "goal": "Detect heart rate from raw ECG signal",
            "max_depth": 8,
            "nodes": [parent],
            "edges": [],
            "history": [],
            "pending_node_ids": ["n_filter"],
            "current_node_id": "n_filter",
            "paradigm": "signal_filter",
            "skeleton_instantiated": True,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }
        deps = DecompositionDeps(catalog=catalog, skill_index=skill_index, llm=llm)
        config = {"configurable": {"deps": deps}}

        result = await decompose_node(state, config)
        names = {n.name for n in result["nodes"]}
        assert len(result["nodes"]) >= 3
        assert "Parse Filter Requirements" in names
        assert "Select Filter Family" in names
        assert result["edges"]
        for node in result["nodes"]:
            assert node.inputs
            assert node.outputs
            assert all(io.type_desc != "Any" for io in node.inputs + node.outputs)

    @pytest.mark.asyncio
    async def test_signal_filter_hints_resolve_to_atomic_builtins(self):
        from sciona.architect.nodes import decompose_node
        from sciona.architect.state import DecompositionDeps

        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)
        skill_index = _make_skill_index()
        llm = AsyncMock()

        async def complete(system: str, user: str) -> str:
            return json.dumps(
                {
                    "sub_nodes": [
                        {
                            "name": "Parse Filter Requirements",
                            "description": "Extract the typed design constraints.",
                            "matched_primitive_hint": "parse_filter_spec",
                        },
                        {
                            "name": "Synthesize Candidate Coefficients",
                            "description": "Generate candidate coefficients.",
                            "matched_primitive_hint": "design_filter_coefficients",
                        },
                        {
                            "name": "Validate and Finalize Coefficients",
                            "description": "Finalize coefficients after checks.",
                            "matched_primitive_hint": "validate_filter_response",
                        },
                    ]
                }
            )

        llm.complete = complete
        parent = AlgorithmicNode(
            node_id="n_filter",
            name="Design Filter",
            description="Design filter coefficients from specification",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[IOSpec(name="spec", type_desc="filter specification")],
            outputs=[IOSpec(name="coefficients", type_desc="filter coefficients")],
            status=NodeStatus.PENDING,
            depth=1,
        )
        state: DecompositionState = {
            "goal": "Detect heart rate from raw ECG signal",
            "max_depth": 8,
            "nodes": [parent],
            "edges": [],
            "history": [],
            "pending_node_ids": ["n_filter"],
            "current_node_id": "n_filter",
            "paradigm": "signal_filter",
            "skeleton_instantiated": True,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }
        deps = DecompositionDeps(catalog=catalog, skill_index=skill_index, llm=llm)
        result = await decompose_node(state, {"configurable": {"deps": deps}})

        assert result["nodes"]
        assert all(node.status == NodeStatus.ATOMIC for node in result["nodes"])

    @pytest.mark.asyncio
    async def test_signal_transform_hints_resolve_to_atomic_builtins(self):
        from sciona.architect.nodes import decompose_node
        from sciona.architect.state import DecompositionDeps

        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)
        skill_index = _make_skill_index()
        llm = AsyncMock()

        async def complete(system: str, user: str) -> str:
            return json.dumps(
                {
                    "sub_nodes": [
                        {
                            "name": "Apply Window Function",
                            "description": "Window the signal deterministically.",
                            "matched_primitive_hint": "apply_window_function",
                        },
                        {
                            "name": "Compute Forward Transform",
                            "description": "Move into the spectral domain.",
                            "matched_primitive_hint": "compute_forward_transform",
                        },
                        {
                            "name": "Process Spectrum",
                            "description": "Apply spectral modifications.",
                            "matched_primitive_hint": "process_spectrum",
                        },
                        {
                            "name": "Compute Inverse Transform",
                            "description": "Return to the time domain.",
                            "matched_primitive_hint": "compute_inverse_transform",
                        },
                    ]
                }
            )

        llm.complete = complete
        parent = AlgorithmicNode(
            node_id="n_transform",
            name="Signal Transform",
            description="Transform a signal to the spectral domain and back",
            concept_type=ConceptType.SIGNAL_TRANSFORM,
            inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
            outputs=[IOSpec(name="result", type_desc="np.ndarray")],
            status=NodeStatus.PENDING,
            depth=1,
        )
        state: DecompositionState = {
            "goal": "Denoise a signal in the spectral domain",
            "max_depth": 8,
            "nodes": [parent],
            "edges": [],
            "history": [],
            "pending_node_ids": ["n_transform"],
            "current_node_id": "n_transform",
            "paradigm": "signal_transform",
            "skeleton_instantiated": True,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }
        deps = DecompositionDeps(catalog=catalog, skill_index=skill_index, llm=llm)
        result = await decompose_node(state, {"configurable": {"deps": deps}})

        assert [node.matched_primitive for node in result["nodes"]] == [
            "apply_window_function",
            "compute_forward_transform",
            "process_spectrum",
            "compute_inverse_transform",
        ]
        assert all(node.status == NodeStatus.ATOMIC for node in result["nodes"])
        assert all(io.type_desc != "Any" for node in result["nodes"] for io in node.inputs + node.outputs)

    def test_signal_filter_partial_chain_is_completed_deterministically(self):
        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)
        parent = AlgorithmicNode(
            node_id="n_filter",
            name="Design Filter",
            description="Design filter coefficients from specification",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[IOSpec(name="spec", type_desc="filter specification")],
            outputs=[IOSpec(name="coefficients", type_desc="filter coefficients")],
            status=NodeStatus.PENDING,
            depth=1,
        )

        result = build_deterministic_decomposition(
            parsed={
                "sub_nodes": [
                    {
                        "name": "Normalize Design Targets",
                        "description": "Translate ECG filter specification into concrete design targets.",
                    },
                    {
                        "name": "Choose Filter Strategy",
                        "description": "Select an appropriate filter topology and design strategy.",
                    },
                    {
                        "name": "Synthesize Coefficients",
                        "description": "Generate candidate coefficients and verify they satisfy design constraints.",
                    },
                ]
            },
            parent=parent,
            catalog=catalog,
        )

        assert [node.matched_primitive for node in result.nodes] == [
            "parse_filter_spec",
            "choose_filter_topology",
            "design_filter_coefficients",
            "validate_filter_response",
        ]
        assert result.nodes[0].outputs[0].name == "design_targets"
        assert result.nodes[-1].outputs[0].name == "coefficients"
        edge_pairs = {
            (edge.output_name, edge.input_name)
            for edge in result.edges
        }
        assert ("design_targets", "design_targets") in edge_pairs
        assert ("candidate_coefficients", "candidate_coefficients") in edge_pairs

    def test_signal_filter_scaffold_drops_disconnected_extra_branch_nodes(self):
        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)
        parent = AlgorithmicNode(
            node_id="n_filter",
            name="Design Filter",
            description="Design filter coefficients from specification",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[IOSpec(name="spec", type_desc="filter specification")],
            outputs=[IOSpec(name="coefficients", type_desc="filter coefficients")],
            status=NodeStatus.PENDING,
            depth=1,
        )

        result = build_deterministic_decomposition(
            parsed={
                "sub_nodes": [
                    {
                        "name": "Normalize Design Targets",
                        "description": "Normalize the filter specification into concrete design targets.",
                    },
                    {
                        "name": "Check Realizability",
                        "description": "Run a side-branch realizability check.",
                    },
                    {
                        "name": "Choose Filter Strategy",
                        "description": "Choose a valid topology.",
                    },
                    {
                        "name": "Synthesize Coefficients",
                        "description": "Generate candidate coefficients.",
                    },
                    {
                        "name": "Validate and Finalize Coefficients",
                        "description": "Finalize the design against targets.",
                    },
                ]
            },
            parent=parent,
            catalog=catalog,
        )

        assert [node.matched_primitive for node in result.nodes] == [
            "parse_filter_spec",
            "choose_filter_topology",
            "design_filter_coefficients",
            "validate_filter_response",
        ]
        assert all(node.name != "Check Realizability" for node in result.nodes)

    def test_validate_stability_partial_chain_is_completed_deterministically(self):
        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)
        parent = AlgorithmicNode(
            node_id="n_stability",
            name="Validate Stability",
            description="Check filter stability via pole analysis",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[IOSpec(name="coefficients", type_desc="filter coefficients")],
            outputs=[IOSpec(name="valid_coefficients", type_desc="filter coefficients")],
            status=NodeStatus.PENDING,
            depth=1,
        )

        result = build_deterministic_decomposition(
            parsed={
                "sub_nodes": [
                    {
                        "name": "Normalize Coefficient Form",
                        "description": "Normalize coefficient ordering and representation.",
                    },
                    {
                        "name": "Compute Pole Locations",
                        "description": "Compute discrete-time poles.",
                    },
                    {
                        "name": "Evaluate Discrete-Time Stability",
                        "description": "Assess whether the poles satisfy the stability criterion.",
                    },
                ]
            },
            parent=parent,
            catalog=catalog,
        )

        assert [node.matched_primitive for node in result.nodes] == [
            "canonicalize_filter_coefficients",
            "construct_characteristic_polynomial",
            "compute_pole_locations",
            "assess_discrete_time_stability",
            "finalize_stable_coefficients",
        ]
        edge_pairs = {(edge.output_name, edge.input_name) for edge in result.edges}
        assert ("normalized_coefficients", "normalized_coefficients") in edge_pairs
        assert ("characteristic_polynomial", "characteristic_polynomial") in edge_pairs
        assert ("poles", "poles") in edge_pairs
        assert ("stability_report", "stability_report") in edge_pairs

    def test_validate_stability_bad_flow_hint_is_overridden_by_typed_edges(self):
        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)
        parent = AlgorithmicNode(
            node_id="n_stability",
            name="Validate Stability",
            description="Check filter stability via pole analysis",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[IOSpec(name="coefficients", type_desc="filter coefficients")],
            outputs=[IOSpec(name="valid_coefficients", type_desc="filter coefficients")],
            status=NodeStatus.PENDING,
            depth=1,
        )

        result = build_deterministic_decomposition(
            parsed={
                "sub_nodes": [
                    {
                        "name": "Normalize Coefficient Form",
                        "description": "Normalize coefficient ordering and representation.",
                    },
                    {
                        "name": "Construct Characteristic Polynomial",
                        "description": "Build the characteristic polynomial.",
                    },
                    {
                        "name": "Compute Pole Locations",
                        "description": "Compute discrete-time poles.",
                    },
                    {
                        "name": "Evaluate Discrete-Time Stability",
                        "description": "Assess whether the poles satisfy the stability criterion.",
                    },
                    {
                        "name": "Emit Stable Coefficients",
                        "description": "Return validated coefficients after the stability check passes.",
                    },
                ],
                "flow_hints": [
                    {
                        "from": "Normalize Coefficient Form",
                        "to": "Compute Pole Locations",
                    }
                ],
            },
            parent=parent,
            catalog=catalog,
        )

        edge_pairs = {
            (edge.source_id, edge.target_id, edge.output_name, edge.input_name)
            for edge in result.edges
        }
        by_name = {node.name: node.node_id for node in result.nodes}
        assert (
            by_name["Construct Characteristic Polynomial"],
            by_name["Compute Pole Locations"],
            "characteristic_polynomial",
            "characteristic_polynomial",
        ) in edge_pairs
        assert (
            by_name["Normalize Coefficient Form"],
            by_name["Compute Pole Locations"],
            "normalized_coefficients",
            "characteristic_polynomial",
        ) not in edge_pairs

    @pytest.mark.asyncio
    async def test_critique_prompt_includes_matched_primitives(self):
        from sciona.architect.nodes import critique_decomposition
        from sciona.architect.state import DecompositionDeps

        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)
        skill_index = _make_skill_index()
        llm = AsyncMock()
        captured: dict[str, str] = {}

        async def complete(system: str, user: str) -> str:
            captured["user"] = user
            return json.dumps({"approved": True, "reason": "Looks good.", "flagged_nodes": []})

        llm.complete = complete
        parent = AlgorithmicNode(
            node_id="n_filter",
            name="Design Filter",
            description="Design filter coefficients from specification",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[IOSpec(name="spec", type_desc="filter specification")],
            outputs=[IOSpec(name="coefficients", type_desc="filter coefficients")],
            status=NodeStatus.DECOMPOSED,
            depth=1,
        )
        parse = AlgorithmicNode(
            node_id="n_parse",
            parent_id="n_filter",
            name="Normalize Design Targets",
            description="Parse the specification into design targets.",
            concept_type=ConceptType.DATA_ASSEMBLY,
            inputs=[IOSpec(name="spec", type_desc="filter specification")],
            outputs=[IOSpec(name="design_targets", type_desc="filter design targets")],
            matched_primitive="parse_filter_spec",
            status=NodeStatus.ATOMIC,
            depth=2,
        )
        choose = AlgorithmicNode(
            node_id="n_choose",
            parent_id="n_filter",
            name="Choose Filter Strategy",
            description="Choose a filter topology.",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[IOSpec(name="design_targets", type_desc="filter design targets")],
            outputs=[IOSpec(name="coefficients", type_desc="filter coefficients")],
            matched_primitive="choose_filter_topology",
            status=NodeStatus.ATOMIC,
            depth=2,
        )
        state: DecompositionState = {
            "goal": "Detect heart rate from raw ECG signal",
            "max_depth": 8,
            "nodes": [parent, parse, choose],
            "edges": [
                DependencyEdge(
                    source_id="n_parse",
                    target_id="n_choose",
                    output_name="design_targets",
                    input_name="design_targets",
                    source_type="filter design targets",
                    target_type="filter design targets",
                )
            ],
            "history": [],
            "pending_node_ids": [],
            "current_node_id": "n_filter",
            "paradigm": "signal_filter",
            "skeleton_instantiated": True,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }

        deps = DecompositionDeps(catalog=catalog, skill_index=skill_index, llm=llm)
        result = await critique_decomposition(state, {"configurable": {"deps": deps}})

        assert result["critique_passed"] is True
        assert "matched_primitive: parse_filter_spec" in captured["user"]
        assert "matched_primitive: choose_filter_topology" in captured["user"]

    @pytest.mark.asyncio
    async def test_approved_critique_does_not_downgrade_flagged_atomic_children(self):
        from sciona.architect.nodes import critique_decomposition
        from sciona.architect.state import DecompositionDeps

        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)
        skill_index = _make_skill_index()
        llm = AsyncMock()

        async def complete(system: str, user: str) -> str:
            return json.dumps(
                {
                    "approved": True,
                    "reason": "Decomposition is acceptable.",
                    "flagged_nodes": ["Interpret Design Specification"],
                }
            )

        llm.complete = complete
        parent = AlgorithmicNode(
            node_id="n_filter",
            name="Design Filter",
            description="Design filter coefficients from specification",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[IOSpec(name="spec", type_desc="filter specification")],
            outputs=[IOSpec(name="coefficients", type_desc="filter coefficients")],
            status=NodeStatus.DECOMPOSED,
            depth=1,
        )
        validate = AlgorithmicNode(
            node_id="n_validate",
            parent_id="n_filter",
            name="Interpret Design Specification",
            description="Validate candidate coefficients against design targets.",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[
                IOSpec(name="candidate_coefficients", type_desc="filter coefficients"),
                IOSpec(name="design_targets", type_desc="filter design targets"),
            ],
            outputs=[IOSpec(name="coefficients", type_desc="filter coefficients")],
            matched_primitive="validate_filter_response",
            status=NodeStatus.ATOMIC,
            depth=2,
        )
        parse = AlgorithmicNode(
            node_id="n_parse",
            parent_id="n_filter",
            name="Parse Filter Requirements",
            description="Parse the specification into design targets.",
            concept_type=ConceptType.DATA_ASSEMBLY,
            inputs=[IOSpec(name="spec", type_desc="filter specification")],
            outputs=[IOSpec(name="design_targets", type_desc="filter design targets")],
            matched_primitive="parse_filter_spec",
            status=NodeStatus.ATOMIC,
            depth=2,
        )
        state: DecompositionState = {
            "goal": "Detect heart rate from raw ECG signal",
            "max_depth": 8,
            "nodes": [parent, parse, validate],
            "edges": [],
            "history": [],
            "pending_node_ids": [],
            "current_node_id": "n_filter",
            "paradigm": "signal_filter",
            "skeleton_instantiated": True,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }

        deps = DecompositionDeps(catalog=catalog, skill_index=skill_index, llm=llm)
        result = await critique_decomposition(state, {"configurable": {"deps": deps}})

        assert result["critique_passed"] is True
        assert "nodes" not in result

    @pytest.mark.asyncio
    async def test_deterministic_critique_rejects_weak_atomic_token_overlap_binding(self):
        from sciona.architect.nodes import critique_decomposition
        from sciona.architect.state import DecompositionDeps

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        llm = AsyncMock()

        parent = AlgorithmicNode(
            node_id="parent",
            name="Plan Search Step",
            description="Coordinate search behavior inside a larger sorting workflow",
            concept_type=ConceptType.SORTING,
            inputs=[IOSpec(name="items", type_desc="list[int]")],
            outputs=[IOSpec(name="index", type_desc="int")],
            status=NodeStatus.DECOMPOSED,
            depth=1,
        )
        weak = AlgorithmicNode(
            node_id="weak",
            parent_id="parent",
            name="Search Sorted Array",
            description="Search a sorted array for a target value.",
            concept_type=ConceptType.SORTING,
            inputs=[
                IOSpec(name="data", type_desc="sorted list[comparable]"),
                IOSpec(name="target", type_desc="comparable"),
            ],
            outputs=[IOSpec(name="index", type_desc="int")],
            matched_primitive="binary_search",
            primitive_binding_confidence=0.69,
            primitive_binding_source="token_overlap",
            status=NodeStatus.ATOMIC,
            depth=2,
        )
        compare = AlgorithmicNode(
            node_id="compare",
            parent_id="parent",
            name="Compare",
            description="Compare two values.",
            concept_type=ConceptType.SORTING,
            inputs=[
                IOSpec(name="a", type_desc="comparable"),
                IOSpec(name="b", type_desc="comparable"),
            ],
            outputs=[IOSpec(name="order", type_desc="bool")],
            matched_primitive="compare",
            primitive_binding_confidence=1.0,
            primitive_binding_source="exact_name",
            status=NodeStatus.ATOMIC,
            depth=2,
        )
        state: DecompositionState = {
            "goal": "Search in a sorted array",
            "max_depth": 8,
            "nodes": [parent, weak, compare],
            "edges": [],
            "history": [],
            "pending_node_ids": [],
            "current_node_id": "parent",
            "paradigm": "sorting",
            "skeleton_instantiated": True,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }

        deps = DecompositionDeps(catalog=catalog, skill_index=skill_index, llm=llm)
        result = await critique_decomposition(state, {"configurable": {"deps": deps}})

        assert result["critique_passed"] is False
        assert "weak primitive binding" in result["critique_reason"].lower()

    @pytest.mark.asyncio
    async def test_decompose_uses_lexical_primitive_fallback_when_semantic_empty(self):
        from sciona.architect.nodes import decompose_node
        from sciona.architect.state import DecompositionDeps

        catalog = _make_catalog()
        # Force semantic/category retrieval path to return empty.
        catalog.find_matching_primitives = lambda node, k=5: []
        skill_index = _make_skill_index()
        llm = AsyncMock()
        captured_users: list[str] = []

        async def complete(system: str, user: str) -> str:
            captured_users.append(user)
            return json.dumps(
                {
                    "sub_nodes": [
                        {
                            "name": "Prepare Search",
                            "description": "Prepare sorted array and query target",
                        },
                        {
                            "name": "binary_search",
                            "description": "Find target index in sorted array",
                            "matched_primitive_hint": "binary_search",
                        },
                    ],
                    "flow_hints": [
                        {"from": "Prepare Search", "to": "binary_search", "why": "setup then search"}
                    ],
                }
            )

        llm.complete = complete
        parent = AlgorithmicNode(
            node_id="n_search",
            name="Target Lookup",
            description="Search for a target in a sorted array",
            concept_type=ConceptType.SEARCHING,
            inputs=[IOSpec(name="data", type_desc="list[int]"), IOSpec(name="target", type_desc="int")],
            outputs=[IOSpec(name="index", type_desc="int")],
            status=NodeStatus.PENDING,
            depth=1,
        )
        state: DecompositionState = {
            "goal": "Find target index",
            "max_depth": 8,
            "nodes": [parent],
            "edges": [],
            "history": [],
            "pending_node_ids": ["n_search"],
            "current_node_id": "n_search",
            "paradigm": "searching",
            "skeleton_instantiated": True,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }
        deps = DecompositionDeps(catalog=catalog, skill_index=skill_index, llm=llm)
        config = {"configurable": {"deps": deps}}

        await decompose_node(state, config)
        assert captured_users
        assert "No relevant primitives found." not in captured_users[0]
        assert "binary_search" in captured_users[0]

    @pytest.mark.asyncio
    async def test_critique_rejects_any_ports_when_parent_is_typed(self):
        from sciona.architect.nodes import critique_decomposition
        from sciona.architect.state import DecompositionDeps

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        llm = _make_mock_llm()

        parent = AlgorithmicNode(
            node_id="parent1",
            name="Typed Parent",
            description="Parent with typed inputs and outputs",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
            outputs=[IOSpec(name="filtered", type_desc="np.ndarray")],
            depth=1,
            status=NodeStatus.PENDING,
        )
        child1 = AlgorithmicNode(
            node_id="child1",
            parent_id="parent1",
            name="Weak Step",
            description="Uses Any ports",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[IOSpec(name="signal", type_desc="Any")],
            outputs=[IOSpec(name="temp", type_desc="np.ndarray")],
            depth=2,
            status=NodeStatus.PENDING,
        )
        child2 = AlgorithmicNode(
            node_id="child2",
            parent_id="parent1",
            name="Another Weak Step",
            description="Consumes Any ports",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[IOSpec(name="temp", type_desc="np.ndarray")],
            outputs=[IOSpec(name="filtered", type_desc="Any")],
            depth=2,
            status=NodeStatus.PENDING,
        )

        state: DecompositionState = {
            "goal": "filter signal",
            "max_depth": 8,
            "nodes": [parent, child1, child2],
            "edges": [
                DependencyEdge(
                    source_id="child1",
                    target_id="child2",
                    output_name="temp",
                    input_name="temp",
                    source_type="np.ndarray",
                    target_type="Any",
                )
            ],
            "history": [],
            "pending_node_ids": ["parent1"],
            "current_node_id": "parent1",
            "paradigm": "signal_filter",
            "skeleton_instantiated": True,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }

        deps = DecompositionDeps(catalog=catalog, skill_index=skill_index, llm=llm)
        result = await critique_decomposition(state, {"configurable": {"deps": deps}})

        assert result["critique_passed"] is False
        assert "unresolved any ports" in result["critique_reason"].lower()

    @pytest.mark.asyncio
    async def test_decompose_node_returns_rewrite_error_for_invalid_primitive_shape(self):
        from sciona.architect.nodes import decompose_node
        from sciona.architect.state import DecompositionDeps

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        llm = AsyncMock()

        async def complete(system: str, user: str) -> str:
            return json.dumps(
                {
                    "sub_nodes": [
                        {
                            "name": "binary_search",
                            "description": "Find the target index.",
                            "matched_primitive_hint": "binary_search",
                            "inputs": [
                                {"name": "data", "type_desc": "Any"},
                                {"name": "target", "type_desc": "Any"},
                            ],
                            "outputs": [
                                {"name": "index", "type_desc": "Any"},
                                {"name": "artifact", "type_desc": "Any"},
                            ],
                        }
                    ]
                }
            )

        llm.complete = complete
        parent = AlgorithmicNode(
            node_id="n_search",
            name="Target Lookup",
            description="Search for a target in a sorted array",
            concept_type=ConceptType.SEARCHING,
            inputs=[IOSpec(name="data", type_desc="sorted list[int]"), IOSpec(name="target", type_desc="int")],
            outputs=[IOSpec(name="index", type_desc="int")],
            status=NodeStatus.PENDING,
            depth=1,
        )
        state: DecompositionState = {
            "goal": "Find target index",
            "max_depth": 8,
            "nodes": [parent],
            "edges": [],
            "history": [],
            "pending_node_ids": ["n_search"],
            "current_node_id": "n_search",
            "paradigm": "searching",
            "skeleton_instantiated": True,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }

        deps = DecompositionDeps(catalog=catalog, skill_index=skill_index, llm=llm)
        result = await decompose_node(state, {"configurable": {"deps": deps}})

        assert result["nodes"] == []
        assert "violates primitive signature" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_decompose_node_history_records_rewrite_actions(self):
        from sciona.architect.nodes import decompose_node
        from sciona.architect.state import DecompositionDeps

        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)
        skill_index = _make_skill_index()
        llm = AsyncMock()

        async def complete(system: str, user: str) -> str:
            return json.dumps(
                {
                    "sub_nodes": [
                        {
                            "name": "Pass Stable Coefficients",
                            "description": "Route the stable result onward.",
                        },
                        {
                            "name": "Evaluate Discrete-Time Stability",
                            "description": "Assess the unit-circle stability margin.",
                        },
                        {
                            "name": "Compute Pole Locations",
                            "description": "Solve for the filter poles.",
                        },
                    ]
                }
            )

        llm.complete = complete
        parent = AlgorithmicNode(
            node_id="n_stability",
            name="Validate Stability",
            description="Validate coefficient stability",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[IOSpec(name="coefficients", type_desc="filter coefficients")],
            outputs=[IOSpec(name="valid_coefficients", type_desc="filter coefficients")],
            status=NodeStatus.PENDING,
            depth=1,
        )
        state: DecompositionState = {
            "goal": "Validate stability",
            "max_depth": 8,
            "nodes": [parent],
            "edges": [],
            "history": [],
            "pending_node_ids": ["n_stability"],
            "current_node_id": "n_stability",
            "paradigm": "signal_filter",
            "skeleton_instantiated": True,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }

        deps = DecompositionDeps(catalog=catalog, skill_index=skill_index, llm=llm)
        result = await decompose_node(state, {"configurable": {"deps": deps}})

        assert result["history"]
        actions = result["history"][0]["rewrite_actions"]
        assert any(action["stage"] == "routing_wrapper_elision" for action in actions)

    @pytest.mark.asyncio
    async def test_advance_node_writes_successful_template_to_shared_context(self):
        from sciona.architect.nodes import advance_node
        from sciona.architect.state import DecompositionDeps

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        llm = _make_mock_llm()
        store = InMemorySharedContextStore()

        parent = AlgorithmicNode(
            node_id="n_parent",
            name="Sort",
            description="Sort input list",
            concept_type=ConceptType.DIVIDE_AND_CONQUER,
            inputs=[IOSpec(name="data", type_desc="list[int]")],
            outputs=[IOSpec(name="result", type_desc="list[int]")],
            status=NodeStatus.PENDING,
            depth=1,
        )
        child = AlgorithmicNode(
            node_id="n_child",
            parent_id="n_parent",
            name="merge",
            description="Merge sorted halves",
            concept_type=ConceptType.SORTING,
            inputs=[
                IOSpec(name="left", type_desc="list[int]"),
                IOSpec(name="right", type_desc="list[int]"),
            ],
            outputs=[IOSpec(name="result", type_desc="list[int]")],
            matched_primitive="merge",
            status=NodeStatus.ATOMIC,
            depth=2,
        )
        state: DecompositionState = {
            "goal": "Implement merge sort",
            "max_depth": 8,
            "nodes": [parent, child],
            "edges": [],
            "history": [],
            "pending_node_ids": ["n_parent"],
            "current_node_id": "n_parent",
            "paradigm": "divide_and_conquer",
            "skeleton_instantiated": True,
            "critique_passed": True,
            "critique_reason": "approved",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }
        deps = DecompositionDeps(
            catalog=catalog,
            skill_index=skill_index,
            llm=llm,
            shared_context=store,
            shared_context_metrics=SharedContextMetrics(),
            context_namespace="architect/test",
        )

        await advance_node(state, {"configurable": {"deps": deps}})

        records = await store.recent("architect/templates", limit=5)
        assert any("Parent: Sort" in r.text for r in records)
        snap = deps.shared_context_metrics.snapshot()
        assert snap["template_puts_total"] == 1
