"""Expansion rules for agent simulation, search, and planning pipelines."""

from __future__ import annotations

import re

from sciona.architect.graph_rewriter import GraphState, Morphism, RewriteRule
from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.principal.expansion import ExpansionContext, ExpansionDiagnostic


_DOMAIN = "agent_simulation_search_planning"


def _node(
    node_id: str,
    name: str,
    concept_type: ConceptType,
    *,
    matched_primitive: str = "",
    description: str = "",
    inputs: list[IOSpec] | None = None,
    outputs: list[IOSpec] | None = None,
) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id,
        name=name,
        description=description or name,
        concept_type=concept_type,
        status=NodeStatus.ATOMIC,
        matched_primitive=matched_primitive,
        inputs=inputs or [],
        outputs=outputs or [],
        type_signature=f"{name} -> result",
    )


def _edge(
    source_id: str,
    target_id: str,
    output_name: str = "out",
    input_name: str = "in",
    type_desc: str = "PlanningArtifact",
) -> DependencyEdge:
    return DependencyEdge(
        source_id=source_id,
        target_id=target_id,
        output_name=output_name,
        input_name=input_name,
        source_type=type_desc,
        target_type=type_desc,
    )


def _normalized_label(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _node_matches_any_label(node: AlgorithmicNode, labels: set[str]) -> bool:
    return (
        _normalized_label(node.node_id) in labels
        or _normalized_label(node.name) in labels
    )


def _fresh_node_id(base: str, used: set[str]) -> str:
    candidate = base
    ordinal = 2
    while candidate in used:
        candidate = f"{base}_{ordinal}"
        ordinal += 1
    used.add(candidate)
    return candidate


def _insert_after_rollout(graph: CDGExport, node: AlgorithmicNode) -> GraphState[CDGExport]:
    source_labels = {
        "search_or_rollout",
        "search",
        "rollout",
        "simulation_rollout",
        "search_or_rollout",
    }
    target_labels = {"constraint_repair", "plan_selection", "policy_value_scoring"}
    source_ids = {
        candidate.node_id
        for candidate in graph.nodes
        if _node_matches_any_label(candidate, source_labels)
    }
    target_ids = {
        candidate.node_id
        for candidate in graph.nodes
        if _node_matches_any_label(candidate, target_labels)
    }
    selected_edge = next(
        (
            edge
            for edge in graph.edges
            if edge.source_id in source_ids and edge.target_id in target_ids
        ),
        None,
    )
    if selected_edge is None:
        return GraphState.failure("No rollout-to-planning edge found for planning refinement insertion")

    node_by_id = {candidate.node_id: candidate for candidate in graph.nodes}
    source_node = node_by_id.get(selected_edge.source_id)
    target_node = node_by_id.get(selected_edge.target_id)
    parent_id = (source_node.parent_id if source_node else None) or (
        target_node.parent_id if target_node else None
    )
    depth = min(
        value
        for value in (
            source_node.depth if source_node else None,
            target_node.depth if target_node else None,
        )
        if value is not None
    )
    new_id = _fresh_node_id(node.node_id, {candidate.node_id for candidate in graph.nodes})
    added = node.model_copy(update={"node_id": new_id, "parent_id": parent_id, "depth": depth})
    retained_edges = [
        edge
        for edge in graph.edges
        if not (
            edge.source_id == selected_edge.source_id
            and edge.target_id == selected_edge.target_id
        )
    ]
    return GraphState.success(
        graph.model_copy(
            update={
                "nodes": [*graph.nodes, added],
                "edges": [
                    *retained_edges,
                    selected_edge.model_copy(update={"target_id": new_id}),
                    selected_edge.model_copy(update={"source_id": new_id}),
                ],
            }
        )
    )


def _semantic_rule(name: str, node: AlgorithmicNode) -> RewriteRule:
    sentinel = _node(
        f"semantic_only_{name}",
        f"semantic_only_{name}",
        ConceptType.CUSTOM,
        matched_primitive=f"__semantic_only__.{name}",
    )
    graph = CDGExport(nodes=[sentinel], edges=[])
    return RewriteRule(
        name=name,
        lhs=graph,
        rhs=graph,
        interface=CDGExport(nodes=[], edges=[]),
        l_morphism=Morphism(node_map={}, edge_map={}),
        r_morphism=Morphism(node_map={}, edge_map={}),
        priority=3,
        semantic_apply=lambda host: _insert_after_rollout(host, node),
    )


def _build_insert_ppo_policy_optimization_after_rollout() -> RewriteRule:
    ppo = _node(
        "ppo_policy_optimization",
        "PPO Policy Optimization",
        ConceptType.OPTIMIZATION,
        matched_primitive="ppo_policy_optimization",
        description=(
            "Refine a simulation or search policy with PPO or multi-agent PPO "
            "before final feasibility repair and plan selection."
        ),
        inputs=[IOSpec(name="rollout_trajectories", type_desc="TrajectoryBatch")],
        outputs=[IOSpec(name="optimized_policy_candidates", type_desc="ScoredCandidateSet")],
    )
    return _semantic_rule("insert_ppo_policy_optimization_after_rollout", ppo)


def _build_insert_mcts_backtracking_search_after_rollout() -> RewriteRule:
    search = _node(
        "mcts_backtracking_search",
        "MCTS Backtracking Search",
        ConceptType.SEARCHING,
        matched_primitive="monte_carlo_tree_search_backtracking",
        description=(
            "Explore candidate reasoning, ranking, game, or planning states with "
            "Monte Carlo Tree Search and deterministic backtracking before final "
            "repair or plan selection."
        ),
        inputs=[IOSpec(name="candidate_states", type_desc="SearchStateSet")],
        outputs=[IOSpec(name="scored_search_paths", type_desc="ScoredCandidateSet")],
    )
    return _semantic_rule("insert_mcts_backtracking_search_after_rollout", search)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
        "required",
        "recommended",
    }


def _planning_text(context: ExpansionContext) -> str:
    artifact = context.planning_artifact or {}
    return str(artifact).lower() if isinstance(artifact, dict) else ""


def _requires_ppo(context: ExpansionContext) -> bool:
    intermediates = context.intermediates or {}
    if any(
        _truthy(intermediates.get(key))
        for key in (
            "agent_simulation.requires_ppo",
            "agent_simulation.use_ppo",
            "agent_simulation.use_multi_agent_ppo",
            "requires_ppo",
            "use_ppo",
            "use_multi_agent_ppo",
        )
    ):
        return True
    text = _planning_text(context)
    return bool(
        re.search(r"\bppo\b", text)
        or "proximal policy optimization" in text
        or "multi-agent ppo" in text
    )


def _requires_mcts_backtracking(context: ExpansionContext) -> bool:
    intermediates = context.intermediates or {}
    if any(
        _truthy(intermediates.get(key))
        for key in (
            "agent_simulation.requires_mcts_backtracking_search",
            "agent_simulation.use_mcts_backtracking_search",
            "requires_mcts_backtracking_search",
            "use_mcts_backtracking_search",
            "requires_mcts",
            "use_mcts",
            "requires_backtracking_search",
            "use_backtracking_search",
        )
    ):
        return True
    text = _planning_text(context)
    return bool(
        "monte carlo tree search" in text
        or re.search(r"\bmcts\b", text)
        or "backtracking mcts" in text
        or "backtracking search" in text
        or "tree search with backtracking" in text
    )


def _diagnose_ppo_policy_optimization(
    cdg: CDGExport,
    context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    if not _requires_ppo(context):
        return None
    return ExpansionDiagnostic(
        rule_name="insert_ppo_policy_optimization_after_rollout",
        severity=0.80,
        evidence="PPO or multi-agent PPO is required for policy optimization inside a planning loop.",
        metric_name="requires_ppo_policy_optimization",
        metric_value=1.0,
        threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_mcts_backtracking_search(
    cdg: CDGExport,
    context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    if not _requires_mcts_backtracking(context):
        return None
    return ExpansionDiagnostic(
        rule_name="insert_mcts_backtracking_search_after_rollout",
        severity=0.80,
        evidence="MCTS or backtracking tree search is required inside the planning loop.",
        metric_name="requires_mcts_backtracking_search",
        metric_value=1.0,
        threshold=0.0,
        source_domain=_DOMAIN,
    )


class AgentSimulationSearchPlanningRuleSet:
    """Expansion rules for planning pipelines with learned policy optimization."""

    @property
    def name(self) -> str:
        return _DOMAIN

    @property
    def domain(self) -> str:
        return _DOMAIN

    def rules(self) -> list[RewriteRule]:
        return [
            _build_insert_ppo_policy_optimization_after_rollout(),
            _build_insert_mcts_backtracking_search_after_rollout(),
        ]

    def diagnose(
        self,
        cdg: CDGExport,
        context: ExpansionContext,
    ) -> list[ExpansionDiagnostic]:
        diagnostics = [
            diagnostic
            for diagnostic in (
                _diagnose_ppo_policy_optimization(cdg, context),
                _diagnose_mcts_backtracking_search(cdg, context),
            )
            if diagnostic is not None
        ]
        return diagnostics
