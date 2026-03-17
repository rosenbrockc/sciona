"""Decomposition engine state and dependency types for LangGraph."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated

from typing_extensions import TypedDict

from ageom.architect.catalog import PrimitiveCatalog
from ageom.architect.embedder import SkillIndex
from ageom.architect.models import AlgorithmicNode, DependencyEdge
from ageom.hunter.llm import LLMClient
from ageom.shared_context import SharedContextMetrics, SharedContextStore

if TYPE_CHECKING:
    from ageom.architect.graph_retrieval import CDGSubgraphRetriever
    from ageom.architect.template_retriever import TemplateRetriever


def _merge_nodes(
    existing: list[AlgorithmicNode], updates: list[AlgorithmicNode]
) -> list[AlgorithmicNode]:
    """Custom reducer: latest entry per node_id wins.

    This lets node functions update status (PENDING -> DECOMPOSED, PENDING -> REJECTED)
    by returning a copy of the node. Standard operator.add would create duplicates.
    """
    by_id: dict[str, AlgorithmicNode] = {}
    for node in existing:
        by_id[node.node_id] = node
    for node in updates:
        by_id[node.node_id] = node
    return list(by_id.values())


class DecompositionState(TypedDict):
    """LangGraph state for the decomposition cycle."""

    # Immutable inputs
    goal: str
    max_depth: int

    # CDG accumulation
    nodes: Annotated[list[AlgorithmicNode], _merge_nodes]
    edges: Annotated[list[DependencyEdge], operator.add]
    history: Annotated[list[dict], operator.add]

    # Per-iteration (overwrite)
    pending_node_ids: list[str]
    current_node_id: str
    paradigm: str
    skeleton_instantiated: bool

    # Critique state (overwrite per-iteration)
    critique_passed: bool
    critique_reason: str
    critique_retries: int

    # Termination flags
    done: bool
    error: str


@dataclass
class DecompositionDeps:
    """Dependencies injected into the decomposition graph via config."""

    catalog: PrimitiveCatalog
    skill_index: SkillIndex
    llm: LLMClient
    graph_retriever: CDGSubgraphRetriever | None = field(default=None)
    template_retriever: TemplateRetriever | None = field(default=None)
    shared_context: SharedContextStore | None = field(default=None)
    shared_context_metrics: SharedContextMetrics | None = field(default=None)
    context_namespace: str = field(default="")
    context_budget_chars: int = field(default=900)
    use_monadic_rewriter: bool = field(default=False)
    architect_critique_llm_enabled: bool = field(default=True)
