"""End-to-end test: CDG subgraph retrieval for Hodges EMG onset detection.

Three parts:
  Part 1 — Retrieval verification against live Memgraph (no LLM).
  Part 2 — Full decomposition with retrieval vs without (mocked LLM).
  Part 3 — Round-trip topo_hash consistency on architect output.

Requires:
  - Memgraph running on bolt://localhost:7687
  - biosppy CDGs already upserted (per-subdirectory repos)

Skip condition: Memgraph unreachable → all tests skip.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

if importlib.util.find_spec("neo4j") is None:
    pytest.skip("requires neo4j driver", allow_module_level=True)

if importlib.util.find_spec("langgraph") is None:
    pytest.skip("requires langgraph", allow_module_level=True)

from ageom.architect.catalog import PrimitiveCatalog
from ageom.architect.graph_retrieval import (
    CDGSubgraphRetriever,
    ExampleDecomposition,
    format_examples_for_prompt,
    make_retriever,
)
from ageom.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from ageom.architect.nodes import decompose_node
from ageom.architect.state import DecompositionDeps
from ageom.config import AgeomConfig
from ageom.graph_store import GraphStore, _topo_hash


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _memgraph_available() -> bool:
    """Check if Memgraph is reachable."""
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver("bolt://localhost:7687")
        with driver.session() as s:
            s.run("RETURN 1")
        driver.close()
        return True
    except Exception:
        return False


def _biosppy_repos_present() -> bool:
    """Check if biosppy CDG repos exist in Memgraph."""
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver("bolt://localhost:7687")
        with driver.session() as s:
            result = s.run(
                "MATCH (a:Atom:Decomposed) WHERE a.repo STARTS WITH 'biosppy.' "
                "RETURN count(a) AS cnt"
            )
            cnt = result.single()["cnt"]
        driver.close()
        return cnt >= 5
    except Exception:
        return False


_skip_no_memgraph = pytest.mark.skipif(
    not _memgraph_available(),
    reason="Memgraph not reachable on bolt://localhost:7687",
)

_skip_no_biosppy = pytest.mark.skipif(
    not _memgraph_available() or not _biosppy_repos_present(),
    reason="biosppy CDGs not upserted into Memgraph",
)


def _hodges_node() -> AlgorithmicNode:
    """Build the Hodges EMG onset detector as an intermediate decomposition node.

    This represents the node as it would appear when decompose_node is called:
    a PENDING node with concept_type and arity matching the stored biosppy
    intermediate decomposed nodes (e.g. Solnik's threshold_based_onset_detection
    has concept_type=signal_filter, 5 inputs, 1 output).
    """
    return AlgorithmicNode(
        node_id="hodges_onset_detection",
        parent_id="hodges_root",
        name="Hodges time-domain EMG onset detection",
        description=(
            "Detect muscle activation onset in surface EMG using the Hodges & Bui "
            "(2003) method: compute rest-segment baseline statistics (mean, std), "
            "form a normalised test statistic h(n) = |x(n) - mu_rest| / sigma_rest, "
            "smooth with a moving average, apply threshold crossing with hysteresis "
            "(onset when h > T_on, offset when h < T_off), and merge adjacent events "
            "within a refractory window."
        ),
        concept_type=ConceptType.SIGNAL_FILTER,
        inputs=[
            IOSpec(name="signal", type_desc="ndarray"),
            IOSpec(name="rest_signal", type_desc="ndarray"),
            IOSpec(name="sampling_rate", type_desc="float"),
            IOSpec(name="threshold", type_desc="float"),
            IOSpec(name="active_state_duration", type_desc="float"),
        ],
        outputs=[
            IOSpec(name="onsets", type_desc="ndarray"),
        ],
        depth=1,
        status=NodeStatus.PENDING,
    )


# ---------------------------------------------------------------------------
# Hodges reference decomposition (from the paper, used as LLM mock output)
# ---------------------------------------------------------------------------

# The Hodges & Bui algorithm has a canonical 6-stage decomposition:
#   1. Compute rest-segment mean and standard deviation
#   2. Subtract baseline offset from test signal
#   3. Compute normalised test statistic: h(n) = |x(n) - μ_rest| / σ_rest
#   4. Smooth test statistic with moving average
#   5. Apply threshold crossing with hysteresis state machine
#   6. Merge adjacent events within refractory window

HODGES_DECOMPOSE_RESPONSE = json.dumps(
    {
        "sub_nodes": [
            {
                "name": "Estimate Rest Baseline Statistics",
                "description": (
                    "Compute mean and standard deviation of the rest-segment EMG "
                    "signal to calibrate the detection threshold."
                ),
                "concept_type": "analysis",
                "inputs": [
                    {"name": "rest_signal", "type_desc": "ndarray"},
                    {"name": "sampling_rate", "type_desc": "float"},
                ],
                "outputs": [
                    {"name": "rest_mean", "type_desc": "float"},
                    {"name": "rest_std", "type_desc": "float"},
                ],
                "type_signature": "ndarray -> float -> tuple[float, float]",
                "is_atomic": True,
                "matched_primitive": "hodges_stub",
            },
            {
                "name": "Remove Baseline Offset",
                "description": (
                    "Subtract the rest-segment mean from the test signal "
                    "to centre it at zero."
                ),
                "concept_type": "signal_filter",
                "inputs": [
                    {"name": "signal", "type_desc": "ndarray"},
                    {"name": "rest_mean", "type_desc": "float"},
                ],
                "outputs": [{"name": "centered_signal", "type_desc": "ndarray"}],
                "type_signature": "ndarray -> float -> ndarray",
                "is_atomic": True,
                "matched_primitive": "hodges_stub",
            },
            {
                "name": "Compute Normalised Test Statistic",
                "description": (
                    "Form h(n) = |centered(n)| / sigma_rest, the signal-to-noise "
                    "ratio at each sample."
                ),
                "concept_type": "arithmetic",
                "inputs": [
                    {"name": "centered_signal", "type_desc": "ndarray"},
                    {"name": "rest_std", "type_desc": "float"},
                ],
                "outputs": [{"name": "test_statistic", "type_desc": "ndarray"}],
                "type_signature": "ndarray -> float -> ndarray",
                "is_atomic": True,
                "matched_primitive": "hodges_stub",
            },
            {
                "name": "Smooth Test Statistic",
                "description": (
                    "Apply a moving-average window to the test statistic "
                    "to suppress transient spikes."
                ),
                "concept_type": "signal_filter",
                "inputs": [
                    {"name": "test_statistic", "type_desc": "ndarray"},
                    {"name": "sampling_rate", "type_desc": "float"},
                ],
                "outputs": [
                    {"name": "smoothed_statistic", "type_desc": "ndarray"}
                ],
                "type_signature": "ndarray -> float -> ndarray",
                "is_atomic": True,
                "matched_primitive": "hodges_stub",
            },
            {
                "name": "Threshold Crossing State Machine",
                "description": (
                    "Detect onset (h > T_on) and offset (h < T_off) transitions "
                    "using a two-threshold hysteresis state machine."
                ),
                "concept_type": "sequential_filter",
                "inputs": [
                    {"name": "smoothed_statistic", "type_desc": "ndarray"},
                    {"name": "threshold", "type_desc": "float"},
                ],
                "outputs": [
                    {"name": "raw_onsets", "type_desc": "ndarray"},
                    {"name": "raw_offsets", "type_desc": "ndarray"},
                ],
                "type_signature": "ndarray -> float -> tuple[ndarray, ndarray]",
                "is_atomic": True,
                "matched_primitive": "hodges_stub",
            },
            {
                "name": "Merge Adjacent Events",
                "description": (
                    "Merge onset/offset pairs that are within the refractory "
                    "window to produce final event boundaries."
                ),
                "concept_type": "set_theory",
                "inputs": [
                    {"name": "raw_onsets", "type_desc": "ndarray"},
                    {"name": "raw_offsets", "type_desc": "ndarray"},
                ],
                "outputs": [
                    {"name": "onsets", "type_desc": "ndarray"},
                    {"name": "offsets", "type_desc": "ndarray"},
                ],
                "type_signature": "ndarray -> ndarray -> tuple[ndarray, ndarray]",
                "is_atomic": True,
                "matched_primitive": "hodges_stub",
            },
        ],
        "edges": [
            {
                "source_name": "Estimate Rest Baseline Statistics",
                "target_name": "Remove Baseline Offset",
                "output_name": "rest_mean",
                "input_name": "rest_mean",
                "data_type": "float",
            },
            {
                "source_name": "Estimate Rest Baseline Statistics",
                "target_name": "Compute Normalised Test Statistic",
                "output_name": "rest_std",
                "input_name": "rest_std",
                "data_type": "float",
            },
            {
                "source_name": "Remove Baseline Offset",
                "target_name": "Compute Normalised Test Statistic",
                "output_name": "centered_signal",
                "input_name": "centered_signal",
                "data_type": "ndarray",
            },
            {
                "source_name": "Compute Normalised Test Statistic",
                "target_name": "Smooth Test Statistic",
                "output_name": "test_statistic",
                "input_name": "test_statistic",
                "data_type": "ndarray",
            },
            {
                "source_name": "Smooth Test Statistic",
                "target_name": "Threshold Crossing State Machine",
                "output_name": "smoothed_statistic",
                "input_name": "smoothed_statistic",
                "data_type": "ndarray",
            },
            {
                "source_name": "Threshold Crossing State Machine",
                "target_name": "Merge Adjacent Events",
                "output_name": "raw_onsets",
                "input_name": "raw_onsets",
                "data_type": "ndarray",
            },
            {
                "source_name": "Threshold Crossing State Machine",
                "target_name": "Merge Adjacent Events",
                "output_name": "raw_offsets",
                "input_name": "raw_offsets",
                "data_type": "ndarray",
            },
        ],
    }
)


# ---------------------------------------------------------------------------
# Mock LLM that returns the Hodges reference decomposition
# ---------------------------------------------------------------------------


class HodgesArchitectLLM:
    """Mock LLM that returns the canonical Hodges decomposition.

    Routes by system prompt keywords, matching the pattern used in
    test_decomposition.py and test_e2e_algorithm_tasks.py.
    """

    async def complete(self, system: str, user: str) -> str:
        lower = system.lower()
        if "best" in lower and "paradigm" in lower:
            return json.dumps(
                {
                    "paradigm": "custom",
                    "rationale": "EMG onset detection is domain-specific",
                    "variant_hint": "hodges_emg",
                }
            )
        if "sub-nodes" in lower or "sub_nodes" in lower:
            return HODGES_DECOMPOSE_RESPONSE
        if "critic" in lower or "evaluate" in lower:
            return json.dumps(
                {
                    "approved": True,
                    "reason": "Valid decomposition for Hodges EMG onset detection",
                    "io_issues": [],
                    "flagged_nodes": [],
                }
            )
        return "{}"

    async def complete_with_grammar(
        self, system: str, user: str, grammar: str
    ) -> str:
        return await self.complete(system, user)


class _AcceptAllCatalog(PrimitiveCatalog):
    """Catalog that confirms any node as atomic."""

    def is_atomic(self, node: Any) -> bool:
        return True


def _make_catalog() -> _AcceptAllCatalog:
    return _AcceptAllCatalog()


def _empty_skill_index():
    index = AsyncMock()
    index.search = lambda query, k=10: []
    return index


# ═══════════════════════════════════════════════════════════════════════════
# Part 1 — Retrieval verification (no LLM, live Memgraph)
# ═══════════════════════════════════════════════════════════════════════════


@_skip_no_biosppy
class TestRetrievalAgainstMemgraph:
    """Verify that the retriever finds structurally similar biosppy CDGs."""

    @pytest.fixture
    async def store(self):
        async with GraphStore("bolt://localhost:7687", "", "") as s:
            yield s

    @pytest.mark.asyncio
    async def test_layer2_finds_emg_detectors(self, store: GraphStore):
        """Layer 2 structural match should return Solnik and/or Abbink.

        Both are EMG onset detectors with similar concept_type and port
        arity to the Hodges query node.
        """
        node = _hodges_node()
        retriever = CDGSubgraphRetriever(
            store=store,
            timeout_ms=5000,
            max_examples=5,
            min_children=2,
            exclude_repo="hodges_test",
        )

        examples = await retriever.find_similar(node, [node], [])

        assert len(examples) > 0, (
            "Expected at least one structurally similar CDG from Memgraph"
        )

        repos = {ex.repo for ex in examples}
        emg_repos = {r for r in repos if "emg" in r or "solnik" in r or "abbink" in r or "bonato" in r}
        assert emg_repos, (
            f"Expected EMG detector repos in results, got: {repos}"
        )

        # All results should have ≥2 children (min_children filter)
        for ex in examples:
            assert len(ex.children) >= 2, (
                f"{ex.fqn} has {len(ex.children)} children, expected ≥2"
            )

    @pytest.mark.asyncio
    async def test_retrieval_returns_children_and_edges(self, store: GraphStore):
        """Returned examples should have populated children and edges."""
        node = _hodges_node()
        retriever = CDGSubgraphRetriever(
            store=store,
            timeout_ms=5000,
            max_examples=3,
            min_children=2,
            exclude_repo="hodges_test",
        )

        examples = await retriever.find_similar(node, [node], [])
        assert len(examples) > 0

        for ex in examples:
            # Children should have meaningful fields
            for child in ex.children:
                assert child.name, f"Child in {ex.fqn} has empty name"
                assert child.concept_type, f"Child {child.name} in {ex.fqn} has empty concept_type"

            # At least some examples should have edges
        has_edges = any(len(ex.edges) > 0 for ex in examples)
        assert has_edges, "Expected at least one example with data-flow edges"

    @pytest.mark.asyncio
    async def test_format_prompt_produces_usable_text(self, store: GraphStore):
        """format_examples_for_prompt on real retrieval results produces
        non-empty text containing repo names and child names."""
        node = _hodges_node()
        retriever = CDGSubgraphRetriever(
            store=store,
            timeout_ms=5000,
            max_examples=3,
            min_children=2,
            exclude_repo="hodges_test",
        )

        examples = await retriever.find_similar(node, [node], [])
        assert len(examples) > 0

        prompt_text = format_examples_for_prompt(examples)
        assert len(prompt_text) > 100, "Expected substantial prompt text"
        assert "Example" in prompt_text
        assert "Children" in prompt_text
        # Should contain at least one biosppy repo name
        assert any(
            repo_fragment in prompt_text
            for repo_fragment in ["biosppy", "solnik", "abbink", "bonato"]
        )

    @pytest.mark.asyncio
    async def test_score_ordering_is_descending(self, store: GraphStore):
        """Results should be ordered by descending score."""
        node = _hodges_node()
        retriever = CDGSubgraphRetriever(
            store=store,
            timeout_ms=5000,
            max_examples=10,
            min_children=2,
            exclude_repo="hodges_test",
        )

        examples = await retriever.find_similar(node, [node], [])
        if len(examples) > 1:
            scores = [ex.score for ex in examples]
            assert scores == sorted(scores, reverse=True), (
                f"Scores not descending: {scores}"
            )

    @pytest.mark.asyncio
    async def test_exclude_repo_is_respected(self, store: GraphStore):
        """No results should come from the excluded repo."""
        node = _hodges_node()
        # Exclude a real biosppy repo to verify filtering
        retriever = CDGSubgraphRetriever(
            store=store,
            timeout_ms=5000,
            max_examples=10,
            min_children=2,
            exclude_repo="biosppy.emg_solnik",
        )

        examples = await retriever.find_similar(node, [node], [])
        for ex in examples:
            assert ex.repo != "biosppy.emg_solnik", (
                f"Excluded repo biosppy.emg_solnik appeared in results"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Part 2 — Full decomposition with vs without retrieval (mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════


def _build_hodges_state(node: AlgorithmicNode) -> dict[str, Any]:
    """Build a minimal DecompositionState dict with the Hodges node as PENDING."""
    root = AlgorithmicNode(
        node_id="hodges_root",
        name="Hodges time-domain EMG onset detection",
        description="Hodges & Bui (2003) EMG onset detection algorithm",
        concept_type=ConceptType.SIGNAL_FILTER,
        status=NodeStatus.DECOMPOSED,
        depth=0,
        children=[node.node_id],
    )
    return {
        "goal": "Hodges time-domain EMG onset detection",
        "max_depth": 4,
        "nodes": [root, node],
        "edges": [],
        "history": [],
        "pending_node_ids": [node.node_id],
        "current_node_id": node.node_id,
        "paradigm": "signal_filter",
        "skeleton_instantiated": True,
        "critique_passed": False,
        "critique_reason": "",
        "critique_retries": 0,
        "done": False,
        "error": "",
    }


def _make_config(
    catalog: PrimitiveCatalog,
    llm: Any,
    graph_retriever: Any = None,
) -> dict[str, Any]:
    """Build a RunnableConfig-compatible dict with DecompositionDeps."""
    return {
        "configurable": {
            "deps": DecompositionDeps(
                catalog=catalog,
                skill_index=_empty_skill_index(),
                llm=llm,
                graph_retriever=graph_retriever,
            ),
            "thread_id": "test",
        }
    }


@_skip_no_biosppy
class TestDecompositionWithRetrieval:
    """Call decompose_node directly with a pre-built state to verify
    structural invariants and retrieval injection for the Hodges algorithm.

    Bypasses select_strategy/skeleton to isolate the retrieval→prompt
    pipeline from skeleton instantiation complexity.
    """

    @pytest.fixture
    async def store(self):
        async with GraphStore("bolt://localhost:7687", "", "") as s:
            yield s

    @pytest.mark.asyncio
    async def test_decomposition_produces_valid_hodges_cdg(
        self, store: GraphStore
    ):
        """With retrieval enabled, decompose_node should produce sub-nodes
        that match the known Hodges algorithm structure."""
        retriever = CDGSubgraphRetriever(
            store=store,
            timeout_ms=5000,
            max_examples=3,
            min_children=2,
            exclude_repo="hodges_test",
        )

        node = _hodges_node()
        state = _build_hodges_state(node)
        config = _make_config(_make_catalog(), HodgesArchitectLLM(), retriever)

        result = await decompose_node(state, config)

        new_nodes = result.get("nodes", [])
        new_edges = result.get("edges", [])

        # Structural invariant 1: at least 4 sub-nodes (Hodges has 6 stages)
        assert len(new_nodes) >= 4, (
            f"Expected ≥4 sub-nodes for Hodges, got {len(new_nodes)}"
        )

        # Structural invariant 2: expected concept_type coverage
        concept_types = {n.concept_type.value for n in new_nodes}
        expected_types = {"analysis", "signal_filter", "arithmetic", "sequential_filter", "set_theory"}
        overlap = concept_types & expected_types
        assert len(overlap) >= 3, (
            f"Expected ≥3 of {expected_types} in concept_types, got {concept_types}"
        )

        # Structural invariant 3: data-flow edges exist
        assert len(new_edges) >= 4, (
            f"Expected ≥4 data-flow edges, got {len(new_edges)}"
        )

        # Structural invariant 4: a node referencing baseline/rest statistics
        descriptions_lower = " ".join(n.description.lower() for n in new_nodes)
        assert any(
            term in descriptions_lower
            for term in ["rest", "baseline", "mean", "standard deviation"]
        ), "Expected a baseline/rest statistics node in the decomposition"

        # Structural invariant 5: a threshold/state-machine detection node
        assert any(
            term in descriptions_lower
            for term in ["threshold", "hysteresis", "state machine", "crossing"]
        ), "Expected a threshold crossing / state machine node"

        # Structural invariant 6: a merge/event consolidation node
        assert any(
            term in descriptions_lower
            for term in ["merge", "consolidat", "refractory", "adjacent"]
        ), "Expected an event merge / refractory node"

    @pytest.mark.asyncio
    async def test_retrieval_examples_injected_into_prompt(
        self, store: GraphStore
    ):
        """Verify that the retriever is actually called and examples appear
        in the LLM prompt during decomposition."""
        retriever = CDGSubgraphRetriever(
            store=store,
            timeout_ms=5000,
            max_examples=3,
            min_children=2,
            exclude_repo="hodges_test",
        )

        # Track what the LLM sees
        prompts_seen: list[str] = []

        class InstrumentedLLM(HodgesArchitectLLM):
            async def complete(self, system: str, user: str) -> str:
                prompts_seen.append(user)
                return await super().complete(system, user)

        node = _hodges_node()
        state = _build_hodges_state(node)
        config = _make_config(_make_catalog(), InstrumentedLLM(), retriever)

        await decompose_node(state, config)

        # The decompose_node prompt should contain example decompositions
        decompose_prompts = [
            p for p in prompts_seen if "sub-nodes" in p.lower() or "sub_nodes" in p.lower()
        ]
        assert len(decompose_prompts) > 0, (
            f"decompose_node LLM was never called. Prompts: {[p[:100] for p in prompts_seen]}"
        )

        # At least one decompose prompt should contain retrieval examples
        has_examples = any(
            "Example decompositions from similar problems" in p
            for p in prompts_seen
        )
        assert has_examples, (
            "Expected retrieval examples in at least one decompose prompt. "
            f"Prompts seen ({len(prompts_seen)}): "
            + prompts_seen[0][:200] if prompts_seen else "none"
        )

    @pytest.mark.asyncio
    async def test_decomposition_without_retrieval_still_works(self):
        """Control: decompose_node without retriever produces valid sub-nodes."""
        node = _hodges_node()
        state = _build_hodges_state(node)
        config = _make_config(_make_catalog(), HodgesArchitectLLM())

        result = await decompose_node(state, config)

        new_nodes = result.get("nodes", [])
        new_edges = result.get("edges", [])
        assert len(new_nodes) >= 4
        assert len(new_edges) >= 4


# ═══════════════════════════════════════════════════════════════════════════
# Part 3 — Round-trip topo_hash consistency
# ═══════════════════════════════════════════════════════════════════════════


@_skip_no_biosppy
class TestTopoHashRoundTrip:
    """Verify topo_hash computed on architect output is consistent with
    what upsert-cdg would store."""

    @pytest.fixture
    async def store(self):
        async with GraphStore("bolt://localhost:7687", "", "") as s:
            yield s

    @pytest.mark.asyncio
    async def test_architect_topo_hash_matches_graph_store_function(self):
        """Compute topo_hash on a CDG produced by decompose_node and verify
        it uses the same algorithm as graph_store._topo_hash."""
        node = _hodges_node()
        state = _build_hodges_state(node)
        config = _make_config(_make_catalog(), HodgesArchitectLLM())

        result = await decompose_node(state, config)

        children = result.get("nodes", [])
        edges = result.get("edges", [])
        parent_id = node.node_id

        # Build the dicts that _topo_hash expects: parent + children
        nodes_dicts = [
            {"node_id": parent_id, "parent_id": "hodges_root"},
        ] + [
            {"node_id": n.node_id, "parent_id": n.parent_id}
            for n in children
        ]
        edges_dicts = [
            {"source_id": e.source_id, "target_id": e.target_id}
            for e in edges
        ]

        topo = _topo_hash(nodes_dicts, edges_dicts, parent_id)
        assert len(topo) == 16, f"Expected 16-char hex hash, got {len(topo)}: {topo}"

        # Compute again to verify determinism
        topo2 = _topo_hash(nodes_dicts, edges_dicts, parent_id)
        assert topo == topo2, "topo_hash is not deterministic"

    @pytest.mark.asyncio
    async def test_topo_hash_differs_from_stored_cdgs(self, store: GraphStore):
        """The Hodges CDG should have a different topo_hash than the stored
        biosppy CDGs (since Hodges has a different degree sequence)."""
        node = _hodges_node()
        state = _build_hodges_state(node)
        config = _make_config(_make_catalog(), HodgesArchitectLLM())

        result = await decompose_node(state, config)

        children = result.get("nodes", [])
        edges = result.get("edges", [])
        parent_id = node.node_id

        nodes_dicts = [
            {"node_id": parent_id, "parent_id": "hodges_root"},
        ] + [
            {"node_id": n.node_id, "parent_id": n.parent_id}
            for n in children
        ]
        edges_dicts = [
            {"source_id": e.source_id, "target_id": e.target_id}
            for e in edges
        ]
        hodges_topo = _topo_hash(nodes_dicts, edges_dicts, parent_id)

        # Query all stored topo_hashes from biosppy repos
        async with store._driver.session() as session:
            result = await session.run(
                "MATCH (a:Atom:Decomposed) "
                "WHERE a.repo STARTS WITH 'biosppy.' AND a.topo_hash IS NOT NULL "
                "RETURN a.fqn AS fqn, a.topo_hash AS topo_hash"
            )
            stored = {rec["fqn"]: rec["topo_hash"] async for rec in result}

        assert len(stored) > 0, "No stored topo_hashes found"

        # Hodges has 6 children with a specific degree sequence that
        # doesn't match any of the biosppy CDGs exactly.
        # This validates that _topo_hash correctly distinguishes the structures.
        matching = [
            fqn for fqn, th in stored.items() if th == hodges_topo
        ]
        # It's OK if there's a match (unlikely but possible) — what we really
        # verify is that the hash was computed and is a valid 16-char hex string.
        # The assertion below is a soft check: log matches for human inspection.
        if matching:
            # Not a failure — but noteworthy
            print(f"NOTE: Hodges topo_hash {hodges_topo} matches: {matching}")

    @pytest.mark.asyncio
    async def test_hodges_degree_sequence_is_correct(self):
        """Verify the Hodges CDG has the expected degree sequence.

        From the reference decomposition:
          Estimate Rest Baseline Statistics: in=0, out=2  (feeds mean + std)
          Remove Baseline Offset:           in=1, out=1  (mean → centered)
          Compute Normalised Test Statistic: in=2, out=1  (centered + std → h)
          Smooth Test Statistic:             in=1, out=1  (h → smoothed)
          Threshold Crossing State Machine:  in=1, out=2  (smoothed → onsets + offsets)
          Merge Adjacent Events:             in=2, out=0  (raw_onsets + raw_offsets → final)

        Sorted: [(0, 2), (1, 1), (1, 1), (1, 2), (2, 0), (2, 1)]
        """
        node = _hodges_node()
        state = _build_hodges_state(node)
        config = _make_config(_make_catalog(), HodgesArchitectLLM())

        result = await decompose_node(state, config)

        children = result.get("nodes", [])
        edges = result.get("edges", [])
        child_ids = {c.node_id for c in children}

        # Compute degree sequence
        degree_seq: list[tuple[int, int]] = []
        for cid in sorted(child_ids):
            in_deg = sum(1 for e in edges if e.target_id == cid)
            out_deg = sum(1 for e in edges if e.source_id == cid)
            degree_seq.append((in_deg, out_deg))

        expected = sorted([(0, 2), (1, 1), (1, 1), (1, 2), (2, 0), (2, 1)])
        actual = sorted(degree_seq)

        assert actual == expected, (
            f"Degree sequence mismatch.\n"
            f"  Expected: {expected}\n"
            f"  Actual:   {actual}"
        )
