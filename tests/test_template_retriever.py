"""Tests for the unified TemplateRetriever (pure Python, no live Memgraph)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from sciona.architect.graph_alignment import GraphAlignmentScorer
from sciona.architect.graph_retrieval import ExampleChild, ExampleDecomposition, ExampleEdge
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.architect.template_retriever import TemplateMatch, TemplateRetriever


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


def _make_example(
    fqn: str = "repo.node_1",
    concept_type: str = "sorting",
    n_children: int = 3,
    score: float = 0.7,
) -> ExampleDecomposition:
    children = [
        ExampleChild(
            node_id=f"child_{i}",
            name=f"Child {i}",
            description=f"child {i}",
            concept_type=concept_type,
            status="atomic",
            n_inputs=1,
            n_outputs=1,
            type_signature="",
        )
        for i in range(n_children)
    ]
    return ExampleDecomposition(
        fqn=fqn,
        name="Example",
        description="example desc",
        concept_type=concept_type,
        repo="other_repo",
        topo_hash="abc123",
        children=children,
        edges=[],
        retrieval_layer=2,
        score=score,
    )


@pytest.mark.asyncio
async def test_returns_empty_when_no_store():
    scorer = GraphAlignmentScorer()
    retriever = TemplateRetriever(store=None, scorer=scorer)
    node = _make_node()
    result = await retriever.find_templates(node, [node], [])
    assert result == []


@pytest.mark.asyncio
async def test_returns_matches_above_threshold():
    store = MagicMock()
    store.query_by_topo_hash = AsyncMock(return_value=[])
    store.query_by_structure = AsyncMock(return_value=[])
    store.query_jaccard_neighborhood = AsyncMock(return_value=[])

    scorer = GraphAlignmentScorer()
    retriever = TemplateRetriever(
        store=store, scorer=scorer, confidence_threshold=0.2
    )

    # Mock the internal retriever to return candidates
    example = _make_example(concept_type="sorting")
    retriever._retriever = MagicMock()
    retriever._retriever.find_similar = AsyncMock(return_value=[example])

    node = _make_node(concept_type=ConceptType.SORTING)
    result = await retriever.find_templates(node, [node], [])

    assert len(result) >= 1
    assert all(isinstance(m, TemplateMatch) for m in result)
    assert all(m.confidence >= 0.2 for m in result)


@pytest.mark.asyncio
async def test_filters_below_threshold():
    scorer = GraphAlignmentScorer()
    retriever = TemplateRetriever(
        store=MagicMock(), scorer=scorer, confidence_threshold=0.99
    )

    # Return a candidate that will score low
    example = _make_example(concept_type="geometry")
    retriever._retriever = MagicMock()
    retriever._retriever.find_similar = AsyncMock(return_value=[example])

    node = _make_node(concept_type=ConceptType.SORTING)
    result = await retriever.find_templates(node, [node], [])

    assert result == []


@pytest.mark.asyncio
async def test_respects_max_results():
    scorer = GraphAlignmentScorer()
    retriever = TemplateRetriever(
        store=MagicMock(), scorer=scorer, max_results=2, confidence_threshold=0.1
    )

    examples = [_make_example(fqn=f"repo.n{i}") for i in range(5)]
    retriever._retriever = MagicMock()
    retriever._retriever.find_similar = AsyncMock(return_value=examples)

    node = _make_node()
    result = await retriever.find_templates(node, [node], [])

    assert len(result) <= 2


@pytest.mark.asyncio
async def test_returns_empty_on_timeout():
    async def slow_find(*args, **kwargs):
        import asyncio
        await asyncio.sleep(10)
        return []

    scorer = GraphAlignmentScorer()
    retriever = TemplateRetriever(
        store=MagicMock(), scorer=scorer, timeout_ms=50
    )
    retriever._retriever = MagicMock()
    retriever._retriever.find_similar = slow_find

    node = _make_node()
    result = await retriever.find_templates(node, [node], [])
    assert result == []


@pytest.mark.asyncio
async def test_sorted_by_confidence():
    scorer = GraphAlignmentScorer()
    retriever = TemplateRetriever(
        store=MagicMock(), scorer=scorer, confidence_threshold=0.1, max_results=5
    )

    examples = [
        _make_example(fqn=f"repo.n{i}", concept_type="sorting")
        for i in range(3)
    ]
    retriever._retriever = MagicMock()
    retriever._retriever.find_similar = AsyncMock(return_value=examples)

    node = _make_node(concept_type=ConceptType.SORTING)
    result = await retriever.find_templates(node, [node], [])

    confidences = [m.confidence for m in result]
    assert confidences == sorted(confidences, reverse=True)


@pytest.mark.asyncio
async def test_find_refinement_templates_no_store():
    scorer = GraphAlignmentScorer()
    retriever = TemplateRetriever(store=None, scorer=scorer)
    node = _make_node()
    result = await retriever.find_refinement_templates(node, {})
    assert result == []


@pytest.mark.asyncio
async def test_find_refinement_templates_returns_verified():
    store = MagicMock()
    store.query_verified_exemplars = AsyncMock(
        side_effect=[
            [
                {
                    "fqn": "repo.good_node",
                    "repo": "repo",
                    "verified_leaf_coverage": 0.9,
                    "topo_hash": "x",
                    "concept_type": "sorting",
                    "n_inputs": 1,
                    "n_outputs": 1,
                }
            ],
            [],
        ]
    )

    scorer = GraphAlignmentScorer()
    retriever = TemplateRetriever(store=store, scorer=scorer, confidence_threshold=0.5)
    # Need to set _store directly since constructor creates _retriever
    retriever._store = store

    node = _make_node()
    result = await retriever.find_refinement_templates(node, {})

    assert len(result) == 1
    assert result[0].source == "verified_exemplar_same_family"
    assert result[0].confidence >= 0.5
    assert store.query_verified_exemplars.await_count == 2


@pytest.mark.asyncio
async def test_find_refinement_templates_falls_back_to_cross_family_structure() -> None:
    store = MagicMock()
    store.query_verified_exemplars = AsyncMock(
        side_effect=[
            [],
            [
                {
                    "fqn": "repo.foreign_node",
                    "repo": "repo",
                    "verified_leaf_coverage": 0.92,
                    "topo_hash": "y",
                    "concept_type": "signal_filter",
                    "n_inputs": 1,
                    "n_outputs": 1,
                }
            ],
        ]
    )

    scorer = GraphAlignmentScorer()
    retriever = TemplateRetriever(store=store, scorer=scorer, confidence_threshold=0.5)
    retriever._store = store

    node = _make_node(concept_type=ConceptType.SORTING)
    result = await retriever.find_refinement_templates(node, {})

    assert len(result) == 1
    assert result[0].source == "verified_exemplar_cross_family"
    assert result[0].example.concept_type == "signal_filter"
    assert result[0].alignment.concept_type_match == 0.0
    assert result[0].alignment.io_arity_match == 1.0
