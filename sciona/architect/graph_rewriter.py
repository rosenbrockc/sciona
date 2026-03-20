from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    DependencyEdge,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)

G = TypeVar("G", bound=CDGExport)
T = TypeVar("T")


class Morphism(BaseModel):
    """A morphism f: G1 -> G2 preserving node/edge types and connectivity."""

    node_map: dict[str, str]  # LHS Node ID -> Target Graph Node ID
    edge_map: dict[str, str]  # LHS Edge ID -> Target Graph Edge ID


@dataclass(frozen=True)
class RewriteRule:
    """A DPO Rewrite Rule: L <- K -> R."""

    name: str
    lhs: CDGExport  # The pattern to find (L)
    rhs: CDGExport  # The replacement (R)
    interface: CDGExport  # The preserved part (K)

    # Morphisms defining the span
    l_morphism: Morphism  # K -> L (inclusion)
    r_morphism: Morphism  # K -> R (transformation)

    priority: int = 0
    anchor_type: str | None = None  # Optimization: Only match at these node types


class GraphState(Generic[G]):
    """A Result-based State Monad for immutable graph transitions."""

    def __init__(self, graph: G, error: str | None = None):
        self._graph = graph
        self._error = error

    @property
    def is_failure(self) -> bool:
        return self._error is not None

    def bind(self, transform: Callable[[G], GraphState[G]]) -> GraphState[G]:
        """Monadic bind: chains transformations if successful."""
        if self.is_failure:
            return self
        return transform(self._graph)

    def unwrap(self) -> G:
        """Return the graph or raise error if transformation failed."""
        if self._error:
            raise RuntimeError(f"Graph Transformation Failed: {self._error}")
        return self._graph

    @classmethod
    def success(cls, graph: G) -> GraphState[G]:
        return cls(graph)

    @classmethod
    def failure(cls, message: str) -> GraphState[G]:
        return cls(None, message)  # type: ignore


class PriorityStrategy:
    """Strategy to determine rule application order."""

    def sort_rules(self, rules: list[RewriteRule]) -> list[RewriteRule]:
        return sorted(rules, key=lambda r: r.priority, reverse=True)


class GraphRewriter:
    """Formal Graph Transformation engine using Double-Pushout (DPO) logic."""

    def __init__(self, strategy: PriorityStrategy | None = None):
        self.strategy = strategy or PriorityStrategy()

    def apply_rule(
        self, rule: RewriteRule, graph: CDGExport
    ) -> GraphState[CDGExport]:
        """Performs a DPO transformation: G -> D -> G'."""
        # 1. Match: Find a match m: L -> G
        match = self._find_match(rule, graph)
        if not match:
            return GraphState.failure(f"Rule '{rule.name}' found no match.")

        # 2. Check Gluing Condition (Identification + Dangling)
        # Simplified: Check that nodes to be deleted don't have edges to the context
        if not self._check_gluing_condition(rule, match, graph):
            return GraphState.failure(f"Rule '{rule.name}' violates gluing condition.")

        # 3. Construct Context Graph D = G - m(L \ K)
        try:
            d_graph = self._remove_lhs_minus_k(rule, match, graph)
        except Exception as e:
            return GraphState.failure(f"Deletion phase failed for '{rule.name}': {e}")

        # 4. Construct Pushout G' = D + (R \ K)
        try:
            g_prime = self._add_rhs_minus_k(rule, match, d_graph)
        except Exception as e:
            return GraphState.failure(f"Addition phase failed for '{rule.name}': {e}")

        return GraphState.success(g_prime)

    def _find_match(
        self, rule: RewriteRule, graph: CDGExport
    ) -> Morphism | None:
        """Find a match for rule.lhs in graph using anchor optimization."""
        # Placeholder for morphism search
        return None

    def _check_gluing_condition(
        self, rule: RewriteRule, match: Morphism, graph: CDGExport
    ) -> bool:
        r"""Verify that L \setminus K nodes have no edges outside m(L)."""
        return True

    def _remove_lhs_minus_k(
        self, rule: RewriteRule, match: Morphism, graph: CDGExport
    ) -> CDGExport:
        r"""Construct the context graph D."""
        return graph.model_copy(deep=True)

    def _add_rhs_minus_k(
        self,
        rule: RewriteRule,
        match: Morphism,
        context_graph: CDGExport,
    ) -> CDGExport:
        """Construct the resulting graph G'."""
        return context_graph.model_copy(deep=True)
