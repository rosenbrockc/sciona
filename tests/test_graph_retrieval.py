"""Tests for CDG subgraph retrieval (pure Python, no live Memgraph)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from ageom.architect.graph_retrieval import (
    CDGSubgraphRetriever,
    ExampleChild,
    ExampleDecomposition,
    ExampleEdge,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(
    node_id: str = "node_1",
    parent_id: str | None = None,
    concept_type: ConceptType = ConceptType.SORTING,
    n_inputs: int = 1,
    n_outputs: int = 1,
) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id,
        parent_id=parent_id,
        name=f"Test {node_id}",
        description="test description",
        concept_type=concept_type,
        inputs=[IOSpec(name=f"in{i}", type_desc="any") for i in range(n_inputs)],
        outputs=[IOSpec(name=f"out{i}", type_desc="any") for i in range(n_outputs)],
        depth=1,
        status=NodeStatus.PENDING,
    )


def _make_record(
    fqn: str = "repo.node_1",
    name: str = "Example Node",
    n_children: int = 3,
    jaccard_score: float = 0.0,
) -> dict:
    children = [
        {
            "node_id": f"child_{i}",
            "name": f"Child {i}",
            "description": f"child {i} desc",
            "concept_type": "sorting",
            "status": "atomic",
            "n_inputs": 1,
            "n_outputs": 1,
            "type_signature": "A -> A",
        }
        for i in range(n_children)
    ]
    edges = [
        {
            "source_id": f"child_{i}",
            "target_id": f"child_{i + 1}",
            "output_name": "result",
            "input_name": "data",
        }
        for i in range(n_children - 1)
    ]
    rec = {
        "fqn": fqn,
        "name": name,
        "description": "example desc",
        "concept_type": "sorting",
        "repo": "other_repo",
        "topo_hash": "abc123",
        "children": children,
        "edges": edges,
    }
    if jaccard_score:
        rec["jaccard_score"] = jaccard_score
    return rec


def _make_retriever(
    store: MagicMock,
    max_examples: int = 3,
    min_children: int = 2,
    timeout_ms: int = 1800,
) -> CDGSubgraphRetriever:
    return CDGSubgraphRetriever(
        store=store,
        timeout_ms=timeout_ms,
        max_examples=max_examples,
        min_children=min_children,
        exclude_repo="my_repo",
    )


# ---------------------------------------------------------------------------
# Tests: error handling & timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_empty_on_store_error():
    store = MagicMock()
    store.query_by_topo_hash = AsyncMock(side_effect=RuntimeError("connection lost"))
    store.query_by_structure = AsyncMock(side_effect=RuntimeError("connection lost"))
    store.query_jaccard_neighborhood = AsyncMock(side_effect=RuntimeError("connection lost"))

    retriever = _make_retriever(store)
    node = _make_node()
    result = await retriever.find_similar(node, [node], [])
    assert result == []


@pytest.mark.asyncio
async def test_returns_empty_on_timeout():
    async def slow_query(*args, **kwargs):
        await asyncio.sleep(10)
        return []

    store = MagicMock()
    store.query_by_topo_hash = slow_query
    store.query_by_structure = slow_query
    store.query_jaccard_neighborhood = slow_query

    retriever = _make_retriever(store, timeout_ms=50)
    node = _make_node()
    result = await retriever.find_similar(node, [node], [])
    assert result == []


# ---------------------------------------------------------------------------
# Tests: layer cascading
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_layer1_hit_stops_pipeline():
    """3 Layer 1 results → Layer 2 never called."""
    store = MagicMock()
    store.query_by_topo_hash = AsyncMock(
        return_value=[
            _make_record(fqn=f"repo.node_{i}") for i in range(3)
        ]
    )
    store.query_by_structure = AsyncMock(return_value=[])
    store.query_jaccard_neighborhood = AsyncMock(return_value=[])

    retriever = _make_retriever(store, max_examples=3)
    node = _make_node()
    result = await retriever.find_similar(node, [node], [])

    assert len(result) == 3
    assert all(ex.retrieval_layer == 1 for ex in result)
    store.query_by_structure.assert_not_called()
    store.query_jaccard_neighborhood.assert_not_called()


@pytest.mark.asyncio
async def test_layer2_called_when_layer1_miss():
    store = MagicMock()
    store.query_by_topo_hash = AsyncMock(return_value=[])
    store.query_by_structure = AsyncMock(
        return_value=[_make_record(fqn="repo.struct_1")]
    )
    store.query_jaccard_neighborhood = AsyncMock(return_value=[])

    retriever = _make_retriever(store)
    node = _make_node()
    result = await retriever.find_similar(node, [node], [])

    assert len(result) == 1
    assert result[0].retrieval_layer == 2
    store.query_by_structure.assert_called_once()


@pytest.mark.asyncio
async def test_layer3_called_as_fallback():
    store = MagicMock()
    store.query_by_topo_hash = AsyncMock(return_value=[])
    store.query_by_structure = AsyncMock(return_value=[])
    store.query_jaccard_neighborhood = AsyncMock(
        return_value=[_make_record(fqn="repo.jaccard_1", jaccard_score=0.6)]
    )

    retriever = _make_retriever(store)
    node = _make_node()
    result = await retriever.find_similar(node, [node], [])

    assert len(result) == 1
    assert result[0].retrieval_layer == 3
    store.query_jaccard_neighborhood.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: deduplication and filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deduplicates_by_fqn():
    """Same fqn from two layers → kept once (higher score wins)."""
    store = MagicMock()
    store.query_by_topo_hash = AsyncMock(
        return_value=[_make_record(fqn="repo.dup_node")]
    )
    store.query_by_structure = AsyncMock(
        return_value=[_make_record(fqn="repo.dup_node")]
    )
    store.query_jaccard_neighborhood = AsyncMock(return_value=[])

    retriever = _make_retriever(store, max_examples=5)
    node = _make_node()
    result = await retriever.find_similar(node, [node], [])

    fqns = [ex.fqn for ex in result]
    assert fqns.count("repo.dup_node") == 1


@pytest.mark.asyncio
async def test_min_children_filter():
    """1-child examples filtered out."""
    store = MagicMock()
    store.query_by_topo_hash = AsyncMock(
        return_value=[_make_record(fqn="repo.tiny", n_children=1)]
    )
    store.query_by_structure = AsyncMock(return_value=[])
    store.query_jaccard_neighborhood = AsyncMock(return_value=[])

    retriever = _make_retriever(store, min_children=2)
    node = _make_node()
    result = await retriever.find_similar(node, [node], [])

    assert result == []


@pytest.mark.asyncio
async def test_top_n_respected():
    """10 results → max_examples=3 returned."""
    store = MagicMock()
    store.query_by_topo_hash = AsyncMock(
        return_value=[_make_record(fqn=f"repo.node_{i}") for i in range(10)]
    )
    store.query_by_structure = AsyncMock(return_value=[])
    store.query_jaccard_neighborhood = AsyncMock(return_value=[])

    retriever = _make_retriever(store, max_examples=3)
    node = _make_node()
    result = await retriever.find_similar(node, [node], [])

    assert len(result) == 3


@pytest.mark.asyncio
async def test_score_ordering():
    """Layer 1 (1.0) > Layer 2 (0.7) > Layer 3 (0.5)."""
    store = MagicMock()
    store.query_by_topo_hash = AsyncMock(
        return_value=[_make_record(fqn="repo.l1")]
    )
    store.query_by_structure = AsyncMock(
        return_value=[_make_record(fqn="repo.l2")]
    )
    store.query_jaccard_neighborhood = AsyncMock(
        return_value=[_make_record(fqn="repo.l3", jaccard_score=1.0)]
    )

    retriever = _make_retriever(store, max_examples=10)
    node = _make_node()
    result = await retriever.find_similar(node, [node], [])

    scores = [ex.score for ex in result]
    assert scores == sorted(scores, reverse=True)
    assert result[0].retrieval_layer == 1
    assert result[0].score == 1.0


# ---------------------------------------------------------------------------
# Tests: format_examples_for_prompt
# ---------------------------------------------------------------------------


def test_format_empty():
    assert format_examples_for_prompt([]) == ""


def test_format_single():
    ex = ExampleDecomposition(
        fqn="biosppy.ecg_filter",
        name="ECG Filter",
        description="Filter ECG signal",
        concept_type="signal_filter",
        repo="biosppy",
        topo_hash="abc123",
        children=[
            ExampleChild(
                node_id="c1",
                name="Bandpass",
                description="bandpass filter",
                concept_type="signal_filter",
                status="atomic",
                n_inputs=1,
                n_outputs=1,
                type_signature="ndarray -> ndarray",
            ),
            ExampleChild(
                node_id="c2",
                name="Normalize",
                description="normalize",
                concept_type="arithmetic",
                status="atomic",
                n_inputs=1,
                n_outputs=1,
                type_signature="ndarray -> ndarray",
            ),
        ],
        edges=[
            ExampleEdge(
                source_id="c1",
                target_id="c2",
                output_name="filtered",
                input_name="signal",
            )
        ],
        retrieval_layer=1,
        score=1.0,
    )
    output = format_examples_for_prompt([ex])
    assert "ECG Filter" in output
    assert "Bandpass" in output
    assert "Normalize" in output
    assert "filtered" in output
    assert "biosppy" in output


# ---------------------------------------------------------------------------
# Tests: make_retriever factory
# ---------------------------------------------------------------------------


def test_make_retriever_disabled():
    config = MagicMock()
    config.graph_retrieval_enabled = False
    store = MagicMock()
    assert make_retriever(config, store) is None


def test_make_retriever_enabled():
    config = MagicMock()
    config.graph_retrieval_enabled = True
    config.graph_retrieval_timeout_ms = 1800
    config.graph_retrieval_max_examples = 3
    config.graph_retrieval_min_children = 2
    store = MagicMock()
    result = make_retriever(config, store, current_repo="test_repo")
    assert isinstance(result, CDGSubgraphRetriever)


def test_make_retriever_no_store():
    config = MagicMock()
    config.graph_retrieval_enabled = True
    assert make_retriever(config, None) is None
