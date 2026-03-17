"""Unified template retriever: coarse Memgraph candidates → GraphAlignmentScorer rerank.

Replaces the weak topo_hash + Jaccard cascade with scored structural alignment
for finding reusable decomposition templates.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ageom.architect.graph_alignment import AlignmentScore, GraphAlignmentScorer
from ageom.architect.graph_retrieval import (
    CDGSubgraphRetriever,
    ExampleDecomposition,
)

if TYPE_CHECKING:
    from ageom.architect.models import AlgorithmicNode, DependencyEdge
    from ageom.graph_store import GraphStore

logger = logging.getLogger(__name__)


@dataclass
class TemplateMatch:
    """A scored template match from the retriever."""

    example: ExampleDecomposition
    alignment: AlignmentScore
    confidence: float
    source: str  # which retrieval layer produced the candidate


class TemplateRetriever:
    """Orchestrates candidate generation, alignment scoring, and confidence thresholds.

    Flow: Memgraph fetches coarse candidates (existing 3-layer cascade) →
    GraphAlignmentScorer reranks → return top results above confidence threshold.
    """

    def __init__(
        self,
        store: GraphStore | None,
        scorer: GraphAlignmentScorer,
        *,
        confidence_threshold: float = 0.6,
        max_candidates: int = 50,
        max_results: int = 3,
        exclude_repo: str = "",
        timeout_ms: int = 2000,
    ) -> None:
        self._store = store
        self._scorer = scorer
        self._confidence_threshold = confidence_threshold
        self._max_candidates = max_candidates
        self._max_results = max_results
        self._exclude_repo = exclude_repo
        self._timeout_ms = timeout_ms
        # Create the underlying CDG retriever for coarse candidates
        self._retriever: CDGSubgraphRetriever | None = None
        if store is not None:
            self._retriever = CDGSubgraphRetriever(
                store=store,
                timeout_ms=timeout_ms,
                max_examples=max_candidates,
                min_children=2,
                exclude_repo=exclude_repo,
            )

    async def find_templates(
        self,
        node: AlgorithmicNode,
        all_nodes: list[AlgorithmicNode],
        all_edges: list[DependencyEdge],
    ) -> list[TemplateMatch]:
        """Find the best-matching templates for a decomposition query.

        Returns up to max_results matches above confidence_threshold,
        sorted by confidence descending.
        """
        if self._retriever is None:
            return []

        try:
            timeout_s = self._timeout_ms / 1000.0
            return await asyncio.wait_for(
                self._run(node, all_nodes, all_edges),
                timeout=timeout_s,
            )
        except Exception:
            logger.debug("template_retriever: failed or timed out", exc_info=True)
            return []

    async def _run(
        self,
        node: AlgorithmicNode,
        all_nodes: list[AlgorithmicNode],
        all_edges: list[DependencyEdge],
    ) -> list[TemplateMatch]:
        assert self._retriever is not None

        # Step 1: Get coarse candidates from the 3-layer cascade
        candidates = await self._retriever.find_similar(node, all_nodes, all_edges)
        if not candidates:
            return []

        # Step 2: Collect query children and edges
        query_children = [n for n in all_nodes if n.parent_id == node.node_id]
        query_edges = [
            e for e in all_edges
            if e.source_id in {c.node_id for c in query_children}
            and e.target_id in {c.node_id for c in query_children}
        ]

        # Step 3: Score each candidate with GraphAlignmentScorer
        matches: list[TemplateMatch] = []
        for candidate in candidates:
            alignment = self._scorer.score(
                node, query_children, query_edges, candidate
            )
            confidence = alignment.total
            source = f"layer_{candidate.retrieval_layer}"
            if confidence >= self._confidence_threshold:
                matches.append(
                    TemplateMatch(
                        example=candidate,
                        alignment=alignment,
                        confidence=confidence,
                        source=source,
                    )
                )

        # Step 4: Sort by confidence and return top N
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches[: self._max_results]

    async def find_refinement_templates(
        self,
        failed_node: AlgorithmicNode,
        failure_context: dict[str, Any],
    ) -> list[TemplateMatch]:
        """Search for previously resolved nodes similar to a failed one.

        Uses the same retriever but looks for candidates with
        verified_leaf_coverage > 0 or matched_primitive set.
        """
        if self._store is None:
            return []

        try:
            concept_type = failed_node.concept_type.value
            records = await self._store.query_verified_exemplars(
                concept_type=concept_type,
                min_coverage=0.5,
                limit=10,
            )
            if not records:
                return []

            # For refinement, we don't have full subgraph info,
            # so return basic matches with confidence from coverage
            matches: list[TemplateMatch] = []
            for rec in records:
                coverage = float(rec.get("verified_leaf_coverage", 0.0))
                confidence = min(0.95, coverage)
                if confidence >= self._confidence_threshold:
                    # Create a minimal ExampleDecomposition for reference
                    example = ExampleDecomposition(
                        fqn=rec.get("fqn", ""),
                        name="",
                        description="",
                        concept_type=concept_type,
                        repo=rec.get("repo", ""),
                        topo_hash=rec.get("topo_hash", ""),
                        children=[],
                        edges=[],
                        retrieval_layer=0,
                        score=coverage,
                    )
                    alignment = AlignmentScore(
                        total=confidence,
                        concept_type_match=1.0,
                        io_arity_match=0.0,
                        child_concept_overlap=0.0,
                        topo_match=0.0,
                        type_class_match=0.0,
                        witness_type_match=0.0,
                    )
                    matches.append(
                        TemplateMatch(
                            example=example,
                            alignment=alignment,
                            confidence=confidence,
                            source="verified_exemplar",
                        )
                    )

            matches.sort(key=lambda m: m.confidence, reverse=True)
            return matches[: self._max_results]
        except Exception:
            logger.debug("template_retriever: refinement search failed", exc_info=True)
            return []
