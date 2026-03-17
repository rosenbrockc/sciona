"""CDG subgraph retrieval for the architect decomposition loop.

Queries Memgraph for structurally and semantically similar decomposition
subgraphs from other repos, providing the LLM with concrete precedents.

Three cascading layers:
  Layer 1: topo_hash exact match (fastest, highest confidence)
  Layer 2: Cypher structural match (concept_type + port arity)
  Layer 3: Jaccard neighbourhood similarity (MAGE)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ageom.architect.models import AlgorithmicNode, DependencyEdge
    from ageom.config import AgeomConfig
    from ageom.graph_store import GraphStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ExampleChild:
    node_id: str
    name: str
    description: str
    concept_type: str
    status: str
    n_inputs: int
    n_outputs: int
    type_signature: str
    abstract_type_class: str = ""
    matched_primitive: str = ""
    witness_input_types: list[str] = field(default_factory=list)
    witness_output_types: list[str] = field(default_factory=list)


@dataclass
class ExampleEdge:
    source_id: str
    target_id: str
    output_name: str
    input_name: str


@dataclass
class ExampleDecomposition:
    fqn: str
    name: str
    description: str
    concept_type: str
    repo: str
    topo_hash: str
    children: list[ExampleChild]
    edges: list[ExampleEdge]
    retrieval_layer: int
    score: float
    jaccard_score: float = 0.0
    abstract_type_class: str = ""
    n_inputs: int = 0
    n_outputs: int = 0


# ---------------------------------------------------------------------------
# Record parsing
# ---------------------------------------------------------------------------


def _parse_record(record: dict[str, Any], layer: int, score: float) -> ExampleDecomposition:
    """Convert a Cypher result record into an ExampleDecomposition."""
    children = []
    for c in record.get("children", []) or []:
        wit_in = c.get("witness_input_types") or []
        wit_out = c.get("witness_output_types") or []
        children.append(
            ExampleChild(
                node_id=c.get("node_id", ""),
                name=c.get("name", ""),
                description=c.get("description", ""),
                concept_type=c.get("concept_type", ""),
                status=c.get("status", ""),
                n_inputs=int(c.get("n_inputs", 0) or 0),
                n_outputs=int(c.get("n_outputs", 0) or 0),
                type_signature=c.get("type_signature", ""),
                abstract_type_class=c.get("abstract_type_class", ""),
                matched_primitive=c.get("matched_primitive", ""),
                witness_input_types=wit_in if isinstance(wit_in, list) else [],
                witness_output_types=wit_out if isinstance(wit_out, list) else [],
            )
        )
    edges = []
    for e in record.get("edges", []) or []:
        edges.append(
            ExampleEdge(
                source_id=e.get("source_id", ""),
                target_id=e.get("target_id", ""),
                output_name=e.get("output_name", ""),
                input_name=e.get("input_name", ""),
            )
        )
    return ExampleDecomposition(
        fqn=record.get("fqn", ""),
        name=record.get("name", ""),
        description=record.get("description", ""),
        concept_type=record.get("concept_type", ""),
        repo=record.get("repo", ""),
        topo_hash=record.get("topo_hash", ""),
        children=children,
        edges=edges,
        retrieval_layer=layer,
        score=score,
        jaccard_score=record.get("jaccard_score", 0.0),
        abstract_type_class=record.get("p_abstract_type_class", ""),
        n_inputs=int(record.get("p_n_inputs", 0) or 0),
        n_outputs=int(record.get("p_n_outputs", 0) or 0),
    )


# ---------------------------------------------------------------------------
# CDGSubgraphRetriever
# ---------------------------------------------------------------------------


class CDGSubgraphRetriever:
    """Multi-layer retriever that finds similar CDG subgraphs from Memgraph."""

    def __init__(
        self,
        store: GraphStore,
        timeout_ms: int = 1800,
        max_examples: int = 3,
        min_children: int = 2,
        exclude_repo: str = "",
    ) -> None:
        self._store = store
        self._timeout_ms = timeout_ms
        self._max_examples = max_examples
        self._min_children = min_children
        self._exclude_repo = exclude_repo

    async def find_similar(
        self,
        node: AlgorithmicNode,
        all_nodes: list[AlgorithmicNode],
        all_edges: list[DependencyEdge],
    ) -> list[ExampleDecomposition]:
        """Find similar decomposition subgraphs, returning up to max_examples.

        Wrapped in asyncio.wait_for with blanket exception handling —
        returns [] on any failure.
        """
        try:
            timeout_s = self._timeout_ms / 1000.0
            return await asyncio.wait_for(
                self._run_layers(node, all_nodes, all_edges),
                timeout=timeout_s,
            )
        except Exception:
            logger.debug("graph_retrieval: failed or timed out", exc_info=True)
            return []

    async def _run_layers(
        self,
        node: AlgorithmicNode,
        all_nodes: list[AlgorithmicNode],
        all_edges: list[DependencyEdge],
    ) -> list[ExampleDecomposition]:
        results: dict[str, ExampleDecomposition] = {}

        # Layer 1: topo_hash exact match
        topo_hash = self._compute_topo_hash(node, all_nodes, all_edges)
        if topo_hash:
            try:
                records = await self._store.query_by_topo_hash(
                    topo_hash, self._exclude_repo, limit=self._max_examples
                )
                for rec in records:
                    ex = _parse_record(rec, layer=1, score=1.0)
                    if len(ex.children) >= self._min_children and ex.fqn not in results:
                        results[ex.fqn] = ex
                logger.debug("graph_retrieval: layer1 returned %d results", len(records))
            except Exception:
                logger.debug("graph_retrieval: layer1 failed", exc_info=True)

        if len(results) >= self._max_examples:
            return self._top_n(results)

        # Layer 2: structural match
        try:
            records = await self._store.query_by_structure(
                concept_type=node.concept_type.value,
                n_inputs=len(node.inputs),
                n_outputs=len(node.outputs),
                exclude_repo=self._exclude_repo,
                min_children=self._min_children,
            )
            for rec in records:
                io_match = self._io_match_factor(node, rec)
                score = 0.7 * io_match
                ex = _parse_record(rec, layer=2, score=score)
                if ex.fqn not in results:
                    results[ex.fqn] = ex
            logger.debug("graph_retrieval: layer2 returned %d results", len(records))
        except Exception:
            logger.debug("graph_retrieval: layer2 failed", exc_info=True)

        if len(results) >= self._max_examples:
            return self._top_n(results)

        # Layer 3: Jaccard neighbourhood (only if exclude_repo is set)
        if self._exclude_repo:
            try:
                fqn = f"{self._exclude_repo}.{node.node_id}"
                records = await self._store.query_jaccard_neighborhood(
                    fqn, self._exclude_repo, limit=self._max_examples
                )
                for rec in records:
                    jaccard = float(rec.get("jaccard_score", 0.0))
                    score = 0.5 * jaccard
                    ex = _parse_record(rec, layer=3, score=score)
                    if len(ex.children) >= self._min_children and ex.fqn not in results:
                        results[ex.fqn] = ex
                logger.debug("graph_retrieval: layer3 returned %d results", len(records))
            except Exception:
                logger.debug("graph_retrieval: layer3 failed", exc_info=True)

        return self._top_n(results)

    def _top_n(self, results: dict[str, ExampleDecomposition]) -> list[ExampleDecomposition]:
        ranked = sorted(results.values(), key=lambda x: x.score, reverse=True)
        return ranked[: self._max_examples]

    @staticmethod
    def _compute_topo_hash(
        node: AlgorithmicNode,
        all_nodes: list[AlgorithmicNode],
        all_edges: list[DependencyEdge],
    ) -> str:
        """Compute topo_hash using the same function as graph_store."""
        from ageom.graph_store import _topo_hash

        nodes_dicts = [
            {"node_id": n.node_id, "parent_id": n.parent_id}
            for n in all_nodes
        ]
        edges_dicts = [
            {"source_id": e.source_id, "target_id": e.target_id}
            for e in all_edges
        ]
        return _topo_hash(nodes_dicts, edges_dicts, node.node_id)

    @staticmethod
    def _io_match_factor(node: AlgorithmicNode, record: dict[str, Any]) -> float:
        """Compute IO-match factor: 1.0 for exact, lower for ±1 mismatch."""
        rec_in = record.get("p_n_inputs", len(node.inputs))
        rec_out = record.get("p_n_outputs", len(node.outputs))
        delta = abs(len(node.inputs) - rec_in) + abs(len(node.outputs) - rec_out)
        return max(0.3, 1.0 - 0.15 * delta)


# ---------------------------------------------------------------------------
# Prompt formatter
# ---------------------------------------------------------------------------


def format_examples_for_prompt(examples: list[ExampleDecomposition]) -> str:
    """Serialize examples into a prompt-injectable string.

    Returns "" when the list is empty — the prompt is unchanged.
    """
    if not examples:
        return ""

    lines: list[str] = []
    lines.append(
        "Example decompositions from similar problems (use as structural "
        "inspiration, but do not copy names verbatim):"
    )

    for i, ex in enumerate(examples, 1):
        lines.append(f"\n  Example {i} (from repo '{ex.repo}'):")
        lines.append(f"    Parent: {ex.name}")
        if ex.description:
            lines.append(f"    Description: {ex.description[:120]}")
        lines.append(f"    Concept type: {ex.concept_type}")
        lines.append(f"    Children ({len(ex.children)}):")
        for child in ex.children:
            arity = f"in={child.n_inputs}, out={child.n_outputs}"
            line = f"      - {child.name} [{child.concept_type}] ({arity})"
            if child.type_signature:
                line += f"  type: {child.type_signature[:50]}"
            lines.append(line)

        # Cap edges at 6 to avoid prompt bloat
        visible_edges = ex.edges[:6]
        if visible_edges:
            lines.append(f"    Data-flow edges ({len(ex.edges)} total):")
            for edge in visible_edges:
                lines.append(
                    f"      {edge.output_name} -> {edge.input_name}"
                )
            if len(ex.edges) > 6:
                lines.append(f"      ... and {len(ex.edges) - 6} more")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_retriever(
    config: AgeomConfig,
    store: GraphStore | None,
    current_repo: str = "",
) -> CDGSubgraphRetriever | None:
    """Create a retriever if graph retrieval is enabled and a store is available."""
    if not config.graph_retrieval_enabled:
        return None
    if store is None:
        return None
    return CDGSubgraphRetriever(
        store=store,
        timeout_ms=config.graph_retrieval_timeout_ms,
        max_examples=config.graph_retrieval_max_examples,
        min_children=config.graph_retrieval_min_children,
        exclude_repo=current_repo,
    )
