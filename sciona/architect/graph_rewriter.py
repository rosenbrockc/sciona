"""DPO (Double Pushout) graph rewriting engine for CDG topology expansion.

Implements formal graph transformation using the algebraic DPO approach:
a RewriteRule is a span  L ← K → R  where L is the pattern to match,
K is the preserved interface, and R is the replacement.

The engine supports three topological operations:
  1. Edge interposition — insert a node on an existing edge
  2. Parallel branch insertion — add a new data-flow branch
  3. Node replacement with expansion — replace one node with several
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from pydantic import BaseModel

from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    NodeStatus,
)

logger = logging.getLogger(__name__)

G = TypeVar("G", bound=CDGExport)
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edge_key(source_id: str, target_id: str) -> str:
    """Deterministic edge identifier from endpoint IDs."""
    return f"{source_id}->{target_id}"


def _edge_key_from_edge(edge: DependencyEdge) -> str:
    return _edge_key(edge.source_id, edge.target_id)


def _node_matches_pattern(
    pattern: AlgorithmicNode, candidate: AlgorithmicNode
) -> bool:
    """Check whether *candidate* satisfies the *pattern* node's constraints.

    Matching semantics (checked in order):
      1. If pattern.matched_primitive is set → exact primitive name match.
      2. Elif pattern.concept_type is not CUSTOM → concept-type match.
      3. Else → wildcard (matches any node).
    """
    if pattern.matched_primitive:
        return candidate.matched_primitive == pattern.matched_primitive
    if pattern.concept_type != ConceptType.CUSTOM:
        return candidate.concept_type == pattern.concept_type
    return True  # wildcard


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

class Morphism(BaseModel):
    """A graph morphism mapping node/edge IDs from one graph to another."""

    node_map: dict[str, str]  # source node ID → target node ID
    edge_map: dict[str, str]  # source edge key → target edge key


@dataclass(frozen=True)
class RewriteRule:
    """A DPO rewrite rule defined by the span  L ← K → R."""

    name: str
    lhs: CDGExport  # pattern to find (L)
    rhs: CDGExport  # replacement (R)
    interface: CDGExport  # preserved boundary (K)

    # Morphisms defining the span
    l_morphism: Morphism  # K → L (inclusion into pattern)
    r_morphism: Morphism  # K → R (inclusion into replacement)

    priority: int = 0
    anchor_type: str | None = None  # optimisation hint: only match at these primitives


class GraphState(Generic[G]):
    """Result-monad for immutable graph transitions."""

    def __init__(self, graph: G, error: str | None = None):
        self._graph = graph
        self._error = error

    @property
    def is_failure(self) -> bool:
        return self._error is not None

    @property
    def error(self) -> str | None:
        return self._error

    def bind(self, transform: Callable[[G], "GraphState[G]"]) -> "GraphState[G]":
        if self.is_failure:
            return self  # type: ignore[return-value]
        return transform(self._graph)

    def unwrap(self) -> G:
        if self._error:
            raise RuntimeError(f"Graph Transformation Failed: {self._error}")
        return self._graph

    @classmethod
    def success(cls, graph: G) -> "GraphState[G]":
        return cls(graph)

    @classmethod
    def failure(cls, message: str) -> "GraphState[G]":
        return cls(None, message)  # type: ignore[arg-type]


class PriorityStrategy:
    """Sort rules by descending priority."""

    def sort_rules(self, rules: list[RewriteRule]) -> list[RewriteRule]:
        return sorted(rules, key=lambda r: r.priority, reverse=True)


# ---------------------------------------------------------------------------
# DPO Graph Rewriter
# ---------------------------------------------------------------------------

class GraphRewriter:
    """Formal graph-transformation engine using Double-Pushout (DPO) logic.

    Given a rule  L ← K → R  and a host graph G:
      1. Find a match  m: L → G  (subgraph isomorphism).
      2. Verify the gluing condition (identification + dangling).
      3. Construct context graph  D = G − m(L \\ K).
      4. Construct result  G' = D + (R \\ K)  glued along K.
    """

    def __init__(self, strategy: PriorityStrategy | None = None):
        self.strategy = strategy or PriorityStrategy()

    def apply_rule(
        self, rule: RewriteRule, graph: CDGExport
    ) -> GraphState[CDGExport]:
        """Apply a single DPO rule to *graph*.  Returns success or failure."""
        match = self._find_match(rule, graph)
        if not match:
            return GraphState.failure(f"Rule '{rule.name}' found no match.")

        if not self._check_gluing_condition(rule, match, graph):
            return GraphState.failure(
                f"Rule '{rule.name}' violates gluing condition."
            )

        try:
            d_graph = self._remove_lhs_minus_k(rule, match, graph)
        except Exception as e:
            return GraphState.failure(
                f"Deletion phase failed for '{rule.name}': {e}"
            )

        try:
            g_prime = self._add_rhs_minus_k(rule, match, d_graph)
        except Exception as e:
            return GraphState.failure(
                f"Addition phase failed for '{rule.name}': {e}"
            )

        return GraphState.success(g_prime)

    # ------------------------------------------------------------------
    # Phase 1: Match  m: L → G
    # ------------------------------------------------------------------

    def _find_match(
        self, rule: RewriteRule, graph: CDGExport
    ) -> Morphism | None:
        """Find an injective morphism from *rule.lhs* into *graph*.

        Uses backtracking search with forward-checking on edge constraints.
        LHS nodes with a concrete ``matched_primitive`` are searched first
        (anchor optimisation) to prune the search space early.

        Complexity: O(|V_G|^k) where k = |V_L|.  Since k ≤ 3 in practice,
        this is linear-to-cubic in the graph size.
        """
        lhs_nodes = rule.lhs.nodes
        lhs_edges = rule.lhs.edges
        g_nodes = graph.nodes

        if not lhs_nodes:
            return Morphism(node_map={}, edge_map={})

        # Build G forward-adjacency for quick edge lookups.
        g_adj: dict[str, set[str]] = {}
        for n in g_nodes:
            g_adj.setdefault(n.node_id, set())
        for e in graph.edges:
            g_adj.setdefault(e.source_id, set()).add(e.target_id)

        # Candidate lists per LHS node.
        l_node_map = {n.node_id: n for n in lhs_nodes}
        candidates: dict[str, list[str]] = {}
        for l_node in lhs_nodes:
            cands = [
                g.node_id
                for g in g_nodes
                if _node_matches_pattern(l_node, g)
            ]
            if not cands:
                return None  # unsatisfiable
            candidates[l_node.node_id] = cands

        # Order: anchored nodes first (fewer candidates → faster pruning).
        l_ids = sorted(
            [n.node_id for n in lhs_nodes],
            key=lambda nid: (
                0 if l_node_map[nid].matched_primitive else 1,
                len(candidates[nid]),
            ),
        )

        assignment: dict[str, str] = {}
        used: set[str] = set()

        def _backtrack(idx: int) -> bool:
            if idx == len(l_ids):
                return True

            l_id = l_ids[idx]
            for g_id in candidates[l_id]:
                if g_id in used:
                    continue

                # Forward-check: verify edges to already-assigned LHS nodes.
                ok = True
                for e in lhs_edges:
                    if e.source_id == l_id and e.target_id in assignment:
                        if assignment[e.target_id] not in g_adj.get(g_id, set()):
                            ok = False
                            break
                    if e.target_id == l_id and e.source_id in assignment:
                        if g_id not in g_adj.get(assignment[e.source_id], set()):
                            ok = False
                            break
                if not ok:
                    continue

                assignment[l_id] = g_id
                used.add(g_id)
                if _backtrack(idx + 1):
                    return True
                del assignment[l_id]
                used.discard(g_id)
            return False

        if not _backtrack(0):
            return None

        # Derive edge map from the node assignment.
        edge_map: dict[str, str] = {}
        for e in lhs_edges:
            g_src = assignment[e.source_id]
            g_tgt = assignment[e.target_id]
            edge_map[_edge_key(e.source_id, e.target_id)] = _edge_key(g_src, g_tgt)

        return Morphism(node_map=dict(assignment), edge_map=edge_map)

    # ------------------------------------------------------------------
    # Phase 2: Gluing condition
    # ------------------------------------------------------------------

    def _check_gluing_condition(
        self,
        rule: RewriteRule,
        match: Morphism,
        graph: CDGExport,
    ) -> bool:
        r"""Verify that removing m(L \\ K) from G leaves no dangling edges.

        For pure-insertion rules (L \\ K is empty) this is trivially true.
        For rules that delete nodes, every edge incident to a deleted node
        must itself be inside the matched region m(L).
        """
        k_images_in_l = set(rule.l_morphism.node_map.values())
        lk_l_node_ids = {
            n.node_id
            for n in rule.lhs.nodes
            if n.node_id not in k_images_in_l
        }

        if not lk_l_node_ids:
            return True  # pure insertion — trivially satisfied

        lk_g_node_ids = {match.node_map[l_id] for l_id in lk_l_node_ids}
        matched_g_nodes = set(match.node_map.values())

        for g_id in lk_g_node_ids:
            for e in graph.edges:
                if e.source_id == g_id and e.target_id not in matched_g_nodes:
                    logger.warning(
                        "Gluing violation: deleted node %s has outgoing edge "
                        "to context node %s",
                        g_id,
                        e.target_id,
                    )
                    return False
                if e.target_id == g_id and e.source_id not in matched_g_nodes:
                    logger.warning(
                        "Gluing violation: deleted node %s has incoming edge "
                        "from context node %s",
                        g_id,
                        e.source_id,
                    )
                    return False
        return True

    # ------------------------------------------------------------------
    # Phase 3: Context graph  D = G − m(L \ K)
    # ------------------------------------------------------------------

    def _remove_lhs_minus_k(
        self,
        rule: RewriteRule,
        match: Morphism,
        graph: CDGExport,
    ) -> CDGExport:
        r"""Remove m(L \\ K) nodes and edges from *graph* to produce D."""
        k_images_in_l = set(rule.l_morphism.node_map.values())
        k_edge_images_in_l = set(rule.l_morphism.edge_map.values())

        # L \ K node IDs  →  their images in G
        lk_l_node_ids = {
            n.node_id
            for n in rule.lhs.nodes
            if n.node_id not in k_images_in_l
        }
        lk_g_node_ids = {match.node_map[l_id] for l_id in lk_l_node_ids}

        # L \ K edge keys  →  their images in G
        lk_l_edge_keys: set[str] = set()
        for e in rule.lhs.edges:
            ek = _edge_key(e.source_id, e.target_id)
            if ek not in k_edge_images_in_l:
                lk_l_edge_keys.add(ek)
        lk_g_edge_keys = {
            match.edge_map[lk] for lk in lk_l_edge_keys if lk in match.edge_map
        }

        new_nodes = [n for n in graph.nodes if n.node_id not in lk_g_node_ids]
        new_edges = [
            e
            for e in graph.edges
            if _edge_key_from_edge(e) not in lk_g_edge_keys
        ]

        return graph.model_copy(update={"nodes": new_nodes, "edges": new_edges})

    # ------------------------------------------------------------------
    # Phase 4: Result graph  G' = D + (R \ K)
    # ------------------------------------------------------------------

    def _add_rhs_minus_k(
        self,
        rule: RewriteRule,
        match: Morphism,
        context_graph: CDGExport,
    ) -> CDGExport:
        r"""Glue R \\ K onto the context graph D, producing G'.

        For each K-image node in R, the corresponding G node ID is found by
        composing  r_morphism⁻¹ · l_morphism · match.  New (R \\ K) nodes
        get fresh UUIDs.
        """
        # Build the R-node-ID → G-node-ID mapping.
        r_to_g: dict[str, str] = {}

        # K-image nodes in R:  K →(l_morphism) L →(match) G
        for k_id, r_id in rule.r_morphism.node_map.items():
            l_id = rule.l_morphism.node_map[k_id]
            g_id = match.node_map[l_id]
            r_to_g[r_id] = g_id

        # R \ K nodes: fresh UUIDs.
        rk_r_node_ids: set[str] = set()
        for node in rule.rhs.nodes:
            if node.node_id not in r_to_g:
                fresh = str(uuid.uuid4())
                r_to_g[node.node_id] = fresh
                rk_r_node_ids.add(node.node_id)

        # Append new nodes to the context graph.
        new_nodes = list(context_graph.nodes)
        for node in rule.rhs.nodes:
            if node.node_id in rk_r_node_ids:
                new_nodes.append(
                    node.model_copy(update={"node_id": r_to_g[node.node_id]})
                )

        # R \ K edges: edges in R whose keys are NOT in r_morphism's image.
        k_edge_images_in_r = set(rule.r_morphism.edge_map.values())
        new_edges = list(context_graph.edges)
        for edge in rule.rhs.edges:
            ek = _edge_key(edge.source_id, edge.target_id)
            if ek not in k_edge_images_in_r:
                new_edges.append(
                    edge.model_copy(
                        update={
                            "source_id": r_to_g[edge.source_id],
                            "target_id": r_to_g[edge.target_id],
                        }
                    )
                )

        return context_graph.model_copy(
            update={"nodes": new_nodes, "edges": new_edges}
        )
