"""Tests for the Decomposition Engine (Phase 2) with mocked LLM."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from ageom.architect.catalog import PrimitiveCatalog
from ageom.architect.models import (
    AlgorithmicNode,
    AlgorithmicPrimitive,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from ageom.architect.state import DecompositionState, _merge_nodes
from ageom.shared_context import InMemorySharedContextStore

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


class TestSelectStrategy:
    """Test that select_strategy picks paradigm and populates pending queue."""

    @pytest.mark.asyncio
    async def test_picks_paradigm_and_instantiates_skeleton(self):
        from ageom.architect.nodes import select_strategy

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        llm = _make_mock_llm()

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

        from ageom.architect.state import DecompositionDeps

        deps = DecompositionDeps(catalog=catalog, skill_index=skill_index, llm=llm)
        config = {"configurable": {"deps": deps}}

        result = await select_strategy(state, config)

        assert result["paradigm"] == "divide_and_conquer"
        assert result["skeleton_instantiated"] is True
        assert len(result["nodes"]) > 1  # root + skeleton nodes
        assert result["pending_node_ids"]  # at least some pending
        assert result["current_node_id"]  # first pending node selected

        # Root should be DECOMPOSED
        root = result["nodes"][0]
        assert root.status == NodeStatus.DECOMPOSED

    @pytest.mark.asyncio
    async def test_fallback_on_parse_error(self):
        """JSON parse failure falls back to CUSTOM paradigm."""
        from ageom.architect.nodes import select_strategy

        catalog = _make_catalog()
        skill_index = _make_skill_index()
        llm = _make_mock_llm(strategy_response="not valid json at all")

        state: DecompositionState = {
            "goal": "Something unusual",
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

        from ageom.architect.state import DecompositionDeps

        deps = DecompositionDeps(catalog=catalog, skill_index=skill_index, llm=llm)
        config = {"configurable": {"deps": deps}}

        result = await select_strategy(state, config)

        assert result["paradigm"] == "custom"
        # No skeleton for CUSTOM, so only root node
        assert len(result["nodes"]) == 1


class TestRouteAfterCritic:
    """Test the 4 routing cases for route_after_critic."""

    def test_retry_on_failure_under_limit(self):
        from ageom.architect.nodes import route_after_critic

        state = {"critique_passed": False, "critique_retries": 1}
        assert route_after_critic(state) == "retry_decompose"

    def test_block_node_on_max_retries(self):
        from ageom.architect.nodes import route_after_critic

        state = {"critique_passed": False, "critique_retries": 3}
        assert route_after_critic(state) == "block_node"

    def test_next_node_on_pass(self):
        from ageom.architect.nodes import route_after_critic

        state = {"critique_passed": True, "critique_retries": 0}
        assert route_after_critic(state) == "next_node"

    def test_next_node_on_pass_with_retries(self):
        from ageom.architect.nodes import route_after_critic

        state = {"critique_passed": True, "critique_retries": 2}
        assert route_after_critic(state) == "next_node"


class TestRouteAfterAdvance:
    """Test route_after_advance routing."""

    def test_end_when_done(self):
        from ageom.architect.nodes import route_after_advance

        state = {"done": True, "pending_node_ids": []}
        assert route_after_advance(state) == "end"

    def test_end_when_no_pending(self):
        from ageom.architect.nodes import route_after_advance

        state = {"done": False, "pending_node_ids": []}
        assert route_after_advance(state) == "end"

    def test_decompose_when_pending(self):
        from ageom.architect.nodes import route_after_advance

        state = {"done": False, "pending_node_ids": ["n1"]}
        assert route_after_advance(state) == "decompose"


class TestDecompositionHappyPath:
    """Full cycle: strategy -> decompose -> critique (approved) -> END."""

    @pytest.mark.asyncio
    async def test_full_decomposition(self):
        from ageom.architect.graph import DecompositionAgent

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


class TestCritiqueRejection:
    """Critique rejects -> retry -> approve on second attempt."""

    @pytest.mark.asyncio
    async def test_retry_then_approve(self):
        from ageom.architect.graph import DecompositionAgent

        catalog = _make_catalog()
        skill_index = _make_skill_index()

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
                            }
                        ],
                    }
                )
            elif "best" in system_lower and "paradigm" in system_lower:
                return json.dumps(
                    {
                        "paradigm": "divide_and_conquer",
                        "rationale": "D&C",
                        "variant_hint": "merge_sort",
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
        )

        cdg = await agent.decompose("Implement merge sort")

        # Should eventually succeed after retry
        assert len(cdg.nodes) > 0
        # The critique was called at least twice
        assert call_count >= 2


class TestBlockedDecomposition:
    @pytest.mark.asyncio
    async def test_agent_metadata_marks_blocked_decomposition(self):
        from ageom.architect.graph import DecompositionAgent

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


class TestMaxDepth:
    """Depth violation caught by deterministic critique check."""

    @pytest.mark.asyncio
    async def test_depth_violation_rejected(self):
        from ageom.architect.nodes import critique_decomposition
        from ageom.architect.state import DecompositionDeps

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
    async def test_malformed_critique_schema_fails_open(self):
        from ageom.architect.nodes import critique_decomposition
        from ageom.architect.state import DecompositionDeps

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

        deps = DecompositionDeps(catalog=catalog, skill_index=skill_index, llm=llm)
        config = {"configurable": {"deps": deps}}
        result = await critique_decomposition(state, config)

        assert result["critique_passed"] is True
        assert "invalid schema" in result["critique_reason"].lower()

    @pytest.mark.asyncio
    async def test_prepare_retry_rejects_prior_atomic_children(self):
        from ageom.architect.nodes import prepare_retry

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
        from ageom.architect.nodes import block_node

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
        from ageom.architect.nodes import select_strategy
        from ageom.architect.state import DecompositionDeps

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
        from ageom.architect.nodes import decompose_node
        from ageom.architect.state import DecompositionDeps

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
        assert captured_users
        assert "Shared Context" in captured_users[0]
        records = await store.recent("architect/test/decompose", limit=5)
        assert any("Parent: Sort" in r.text for r in records)


class TestDeterministicDecompose:
    @pytest.mark.asyncio
    async def test_conceptual_payload_gets_deterministic_ports_and_edges(self):
        from ageom.architect.nodes import decompose_node
        from ageom.architect.state import DecompositionDeps

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
        from ageom.architect.nodes import decompose_node
        from ageom.architect.state import DecompositionDeps

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
    async def test_decompose_uses_lexical_primitive_fallback_when_semantic_empty(self):
        from ageom.architect.nodes import decompose_node
        from ageom.architect.state import DecompositionDeps

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
        from ageom.architect.nodes import critique_decomposition
        from ageom.architect.state import DecompositionDeps

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
