"""Scored structural alignment between a CDG query node and a retrieved ExampleDecomposition.

Replaces the weak topo_hash + Jaccard cascade with a multi-factor
alignment score that accounts for concept type, IO arity, child overlap,
topology, abstract type class, and witness types.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from sciona.architect.graph_retrieval import ExampleDecomposition, ExampleEdge
    from sciona.architect.models import AlgorithmicNode, DependencyEdge


# ---------------------------------------------------------------------------
# Score container
# ---------------------------------------------------------------------------


@dataclass
class AlignmentScore:
    """Multi-factor alignment score between a query node and a candidate."""

    total: float  # weighted sum, 0.0-1.0
    concept_type_match: float  # weight 0.25
    io_arity_match: float  # weight 0.15
    child_concept_overlap: float  # weight 0.25
    topo_match: float  # weight 0.15
    type_class_match: float  # weight 0.10
    witness_type_match: float  # weight 0.10


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

_WEIGHTS = {
    "concept_type_match": 0.25,
    "io_arity_match": 0.15,
    "child_concept_overlap": 0.25,
    "topo_match": 0.15,
    "type_class_match": 0.10,
    "witness_type_match": 0.10,
}


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class GraphAlignmentScorer:
    """Score how well a query CDG node matches a candidate ExampleDecomposition."""

    def score(
        self,
        query_node: AlgorithmicNode,
        query_children: Sequence[AlgorithmicNode],
        query_edges: Sequence[DependencyEdge],
        candidate: ExampleDecomposition,
    ) -> AlignmentScore:
        """Compute a multi-factor alignment score."""
        concept = self._concept_type_match(query_node, candidate)
        io = self._io_arity_match(query_node, candidate)
        child = self._child_concept_overlap(query_children, candidate)
        topo = self._topo_match(query_edges, candidate.edges)
        tc = self._type_class_match(query_node, candidate)
        wit = self._witness_type_match(query_children, candidate)

        total = (
            _WEIGHTS["concept_type_match"] * concept
            + _WEIGHTS["io_arity_match"] * io
            + _WEIGHTS["child_concept_overlap"] * child
            + _WEIGHTS["topo_match"] * topo
            + _WEIGHTS["type_class_match"] * tc
            + _WEIGHTS["witness_type_match"] * wit
        )

        return AlignmentScore(
            total=total,
            concept_type_match=concept,
            io_arity_match=io,
            child_concept_overlap=child,
            topo_match=topo,
            type_class_match=tc,
            witness_type_match=wit,
        )

    # -- Individual factors --------------------------------------------------

    @staticmethod
    def _concept_type_match(
        query_node: AlgorithmicNode,
        candidate: ExampleDecomposition,
    ) -> float:
        return 1.0 if query_node.concept_type.value == candidate.concept_type else 0.0

    @staticmethod
    def _io_arity_match(
        query_node: AlgorithmicNode,
        candidate: ExampleDecomposition,
    ) -> float:
        # Fall back to 1.0 when candidate has no IO metadata.
        if candidate.n_inputs == 0 and candidate.n_outputs == 0:
            return 1.0
        delta = abs(len(query_node.inputs) - candidate.n_inputs) + abs(
            len(query_node.outputs) - candidate.n_outputs
        )
        return max(0.0, 1.0 - 0.15 * delta)

    @staticmethod
    def _child_concept_overlap(
        query_children: Sequence[AlgorithmicNode],
        candidate: ExampleDecomposition,
    ) -> float:
        """Jaccard similarity of child concept_type multisets."""
        if not query_children and not candidate.children:
            return 1.0
        if not query_children or not candidate.children:
            return 0.0
        q_set = {c.concept_type.value for c in query_children}
        c_set = {c.concept_type for c in candidate.children}
        intersection = q_set & c_set
        union = q_set | c_set
        if not union:
            return 1.0
        return len(intersection) / len(union)

    @staticmethod
    def _topo_match(
        query_edges: Sequence[DependencyEdge],
        candidate_edges: Sequence[ExampleEdge],
    ) -> float:
        """Compare sorted (in_degree, out_degree) sequences.

        Returns 1.0 if identical, partial credit based on overlap.
        """
        if not query_edges and not candidate_edges:
            return 1.0
        if not query_edges or not candidate_edges:
            return 0.0

        def degree_seq(
            edges: Sequence, source_attr: str = "source_id", target_attr: str = "target_id"
        ) -> list[tuple[int, int]]:
            in_deg: Counter[str] = Counter()
            out_deg: Counter[str] = Counter()
            nodes: set[str] = set()
            for e in edges:
                src = getattr(e, source_attr)
                tgt = getattr(e, target_attr)
                out_deg[src] += 1
                in_deg[tgt] += 1
                nodes.add(src)
                nodes.add(tgt)
            seq = sorted((in_deg.get(n, 0), out_deg.get(n, 0)) for n in nodes)
            return seq

        q_seq = degree_seq(query_edges)
        c_seq = degree_seq(candidate_edges)

        if q_seq == c_seq:
            return 1.0

        # Partial credit: multiset overlap / max size
        q_counter: Counter[tuple[int, int]] = Counter(q_seq)
        c_counter: Counter[tuple[int, int]] = Counter(c_seq)
        overlap = sum((q_counter & c_counter).values())
        total = max(len(q_seq), len(c_seq))
        return overlap / total if total else 1.0

    @staticmethod
    def _type_class_match(
        query_node: AlgorithmicNode,
        candidate: ExampleDecomposition,
    ) -> float:
        q_tc = getattr(query_node, "abstract_type_class", "")
        if not q_tc:
            return 1.0
        return 1.0 if q_tc == candidate.abstract_type_class else 0.0

    @staticmethod
    def _witness_type_match(
        query_children: Sequence[AlgorithmicNode],
        candidate: ExampleDecomposition,
    ) -> float:
        """Check overlap between candidate witness types and query child type_signatures."""
        # Collect all witness types from candidate children.
        has_witness = False
        candidate_witness_types: set[str] = set()
        for cc in candidate.children:
            if cc.witness_input_types or cc.witness_output_types:
                has_witness = True
                candidate_witness_types.update(cc.witness_input_types)
                candidate_witness_types.update(cc.witness_output_types)

        if not has_witness:
            return 1.0

        if not query_children:
            return 0.0

        # For each query child, check if its type_signature overlaps with witness types.
        matches = 0
        for qc in query_children:
            sig = qc.type_signature
            if sig and sig in candidate_witness_types:
                matches += 1

        return matches / len(query_children)
