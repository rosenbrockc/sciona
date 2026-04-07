"""Node functions and routing for the LangGraph decomposition cycle.

Each node function: async def fn(state, config) -> dict
Routing functions: def fn(state) -> str
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from langchain_core.runnables import RunnableConfig

from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.architect.deterministic_decompose import (
    DeterministicRewriteError,
    build_deterministic_decomposition,
)
from sciona.architect.proposal_models import (
    ProposalType,
    proposal_from_primitive,
    proposal_from_template_match,
)
from sciona.architect.proposal_ranking import ScoredProposal, rank_proposals
from sciona.architect.skeleton_proposals import generate_skeleton_proposals
from sciona.architect.structural_critic import structural_critique_issues
from sciona.architect.planning_contract import (
    PlanningConstraint,
    PlanningConstraintCategory,
    build_planning_artifact,
    render_planning_artifact_block,
    summarize_planning_artifact,
)
from sciona.architect.prompts import (
    CRITIQUE_SYSTEM,
    CRITIQUE_USER,
    DECOMPOSE_NODE_SYSTEM,
    DECOMPOSE_NODE_USER,
    SELECT_STRATEGY_SYSTEM,
    SELECT_STRATEGY_USER,
)
from sciona.architect.skeletons import (
    SKELETON_TEMPLATES,
    get_skeleton,
    infer_boundary_ports,
    instantiate_skeleton,
)
from sciona.architect.state import DecompositionDeps, DecompositionState
from sciona.llm_router import (
    ARCHITECT_CRITIQUE,
    ARCHITECT_DECOMPOSE,
    ARCHITECT_STRATEGY,
    select_llm,
)
from sciona.shared_context import format_context_block
from sciona.telemetry import (
    get_current_stage,
    increment_run_metadata_counter,
    log_event,
    merge_run_metadata,
    update_stage,
)

# ---------------------------------------------------------------------------
# Conjugate prior/likelihood pair detection
# ---------------------------------------------------------------------------

_SKELETON_ACCEPTANCE_MARGIN = 0.15

# Each entry maps a canonical pair name to recognizable keywords (in lower-case)
# that appear in goal text or IO specifications, along with the sufficient
# statistic description, hyperparameter update rule, and result distribution.
_CONJUGATE_PAIRS: dict[str, dict] = {
    "beta_bernoulli": {
        "keywords": [
            "beta-bernoulli",
            "beta bernoulli",
            "coin flip",
            "binomial with beta prior",
            "bernoulli with beta",
        ],
        "library_hints": [
            "conjugatepriorslib",
            "conjugatepriors.jl",
            "bayes crate",
            "conjugate_update",
            "analytical update",
        ],
        "sufficient_stat": "Count successes k and total trials n from data",
        "hyperparameter_update": "alpha_post = alpha_prior + k, beta_post = beta_prior + (n - k)",
        "result_distribution": "Beta(alpha_post, beta_post)",
        "type_sig_ingest": "ndarray -> tuple[int, int]",
        "type_sig_update": "tuple[float, float] -> tuple[int, int] -> tuple[float, float]",
        "type_sig_construct": "tuple[float, float] -> BetaDistribution",
    },
    "normal_normal": {
        "keywords": [
            "normal-normal",
            "normal normal",
            "gaussian conjugate",
            "gaussian with known variance",
            "normal with normal prior",
        ],
        "library_hints": [
            "conjugatepriorslib",
            "conjugatepriors.jl",
            "bayes crate",
            "conjugate_update",
            "analytical update",
        ],
        "sufficient_stat": "Compute sample mean x_bar and sample count n from data",
        "hyperparameter_update": (
            "mu_post = (mu_prior / sigma_prior^2 + n * x_bar / sigma_obs^2) "
            "/ (1/sigma_prior^2 + n/sigma_obs^2); "
            "sigma_post^2 = 1 / (1/sigma_prior^2 + n/sigma_obs^2)"
        ),
        "result_distribution": "Normal(mu_post, sigma_post)",
        "type_sig_ingest": "ndarray -> tuple[float, int]",
        "type_sig_update": "tuple[float, float] -> tuple[float, int] -> tuple[float, float] -> tuple[float, float]",
        "type_sig_construct": "tuple[float, float] -> NormalDistribution",
    },
    "gamma_poisson": {
        "keywords": ["gamma-poisson", "gamma poisson", "poisson with gamma prior"],
        "library_hints": [
            "conjugatepriorslib",
            "conjugatepriors.jl",
            "bayes crate",
            "conjugate_update",
            "analytical update",
        ],
        "sufficient_stat": "Sum all observations s and count n from data",
        "hyperparameter_update": "alpha_post = alpha_prior + s, beta_post = beta_prior + n",
        "result_distribution": "Gamma(alpha_post, beta_post)",
        "type_sig_ingest": "ndarray -> tuple[float, int]",
        "type_sig_update": "tuple[float, float] -> tuple[float, int] -> tuple[float, float]",
        "type_sig_construct": "tuple[float, float] -> GammaDistribution",
    },
    "dirichlet_categorical": {
        "keywords": [
            "dirichlet-categorical",
            "dirichlet categorical",
            "dirichlet multinomial",
            "categorical with dirichlet",
        ],
        "library_hints": [
            "conjugatepriorslib",
            "conjugatepriors.jl",
            "bayes crate",
            "conjugate_update",
            "analytical update",
        ],
        "sufficient_stat": "Count occurrences of each category from data",
        "hyperparameter_update": "alpha_post_k = alpha_prior_k + count_k for each category k",
        "result_distribution": "Dirichlet(alpha_post)",
        "type_sig_ingest": "ndarray -> ndarray",
        "type_sig_update": "ndarray -> ndarray -> ndarray",
        "type_sig_construct": "ndarray -> DirichletDistribution",
    },
}


def _detect_conjugate_pair(goal: str) -> dict | None:
    """Deterministic pre-scan for conjugate prior/likelihood pairs.

    Checks the goal text for known conjugate pair keywords and library hints.
    Returns the pair spec dict if detected, None otherwise.
    """
    goal_lower = goal.lower()
    for pair_name, spec in _CONJUGATE_PAIRS.items():
        for kw in spec["keywords"]:
            if kw in goal_lower:
                return {**spec, "pair_name": pair_name}
        for hint in spec["library_hints"]:
            if hint in goal_lower:
                # Library hint alone is weaker — require *some* probabilistic keyword too
                if any(
                    w in goal_lower
                    for w in [
                        "prior",
                        "posterior",
                        "conjugate",
                        "bayesian",
                        "update",
                        "inference",
                        "likelihood",
                    ]
                ):
                    return {**spec, "pair_name": pair_name}
    return None


def _get_deps(config: RunnableConfig) -> DecompositionDeps:
    """Extract DecompositionDeps from LangGraph config."""
    return config["configurable"]["deps"]


def _find_node(nodes: list[AlgorithmicNode], node_id: str) -> AlgorithmicNode | None:
    """Find a node by ID in the nodes list."""
    for n in nodes:
        if n.node_id == node_id:
            return n
    return None


def _descendant_ids(nodes: list[AlgorithmicNode], root_id: str) -> set[str]:
    """Return all descendant node IDs under a parent."""
    pending = [root_id]
    descendants: set[str] = set()
    while pending:
        current = pending.pop()
        for node in nodes:
            if node.parent_id != current or node.node_id in descendants:
                continue
            descendants.add(node.node_id)
            pending.append(node.node_id)
    return descendants


def _pending_under_blocked(nodes: list[AlgorithmicNode]) -> list[AlgorithmicNode]:
    """Detect invalid pending descendants beneath blocked parents."""
    blocked_ids = {
        node.node_id for node in nodes if node.status in {NodeStatus.BLOCKED, NodeStatus.HIGH_RISK}
    }
    if not blocked_ids:
        return []
    by_id = {node.node_id: node for node in nodes}
    offenders: list[AlgorithmicNode] = []
    for node in nodes:
        if node.status != NodeStatus.PENDING:
            continue
        parent_id = node.parent_id
        while parent_id:
            if parent_id in blocked_ids:
                offenders.append(node)
                break
            parent = by_id.get(parent_id)
            parent_id = parent.parent_id if parent is not None else None
    return offenders


def _critique_reason_category(reason: str) -> str:
    text = reason.lower()
    if "any" in text or "type" in text:
        return "type_compatibility"
    if "path" in text or "missing step" in text or "complete" in text:
        return "semantic_completeness"
    if "atomic" in text or "catalog" in text:
        return "atomicity"
    if "depth" in text:
        return "depth_limit"
    if "parse" in text or "schema" in text:
        return "llm_output"
    return "other"


def _architect_snapshot(
    nodes: list[AlgorithmicNode],
    edges: list[DependencyEdge],
) -> dict[str, Any]:
    active_nodes = [node for node in nodes if node.status != NodeStatus.REJECTED]
    active_ids = {node.node_id for node in active_nodes}
    active_edges = [
        edge
        for edge in edges
        if edge.source_id in active_ids and edge.target_id in active_ids
    ]
    parent_ids = {edge.source_id for edge in active_edges}
    leaves = [
        node
        for node in active_nodes
        if not node.children and node.node_id not in parent_ids
    ]
    non_atomic_leaves = [
        node for node in leaves if node.status != NodeStatus.ATOMIC
    ]
    total_ports = sum(len(node.inputs) + len(node.outputs) for node in active_nodes)
    any_ports = sum(
        1
        for node in active_nodes
        for io in (node.inputs + node.outputs)
        if _is_any_type(io.type_desc)
    )
    total_edges = len(active_edges)
    any_edges = sum(
        1
        for edge in active_edges
        if _is_any_type(edge.source_type) or _is_any_type(edge.target_type)
    )
    status_counts: dict[str, int] = {}
    for node in active_nodes:
        key = node.status.value
        status_counts[key] = status_counts.get(key, 0) + 1
    return {
        "node_status_counts": status_counts,
        "unresolved_leaf_count": len(non_atomic_leaves),
        "blocked_node_names": [
            node.name for node in active_nodes if node.status == NodeStatus.BLOCKED
        ],
        "any_port_count": any_ports,
        "total_port_count": total_ports,
        "any_port_pct": (any_ports / total_ports) if total_ports else 0.0,
        "any_edge_count": any_edges,
        "total_edge_count": total_edges,
        "any_edge_pct": (any_edges / total_edges) if total_edges else 0.0,
    }


def _publish_architect_snapshot(
    *,
    phase: str,
    node_id: str,
    nodes: list[AlgorithmicNode],
    edges: list[DependencyEdge],
    extra: dict[str, Any] | None = None,
) -> None:
    snapshot = _architect_snapshot(nodes, edges)
    if extra:
        snapshot.update(extra)
    merge_run_metadata({"architect_metrics": snapshot})
    log_event(
        "architect",
        phase,
        "ARCHITECT_SNAPSHOT",
        node_id=node_id,
        payload=snapshot,
    )


def _format_io(specs: list[IOSpec]) -> str:
    """Format IOSpec list for prompt display."""
    if not specs:
        return "none"
    parts: list[str] = []
    for spec in specs:
        rendered = f"{spec.name}: {spec.type_desc}"
        if not spec.required:
            rendered += " (optional"
            if spec.default_value_repr:
                rendered += f", default={spec.default_value_repr}"
            rendered += ")"
        parts.append(rendered)
    return ", ".join(parts)


def _format_primitives(prims: list) -> str:
    """Format primitives list for prompt display."""
    if not prims:
        return "No relevant primitives found."
    lines = []
    for p in prims[:10]:
        line = f"- {p.name} [{p.category.value}]: {p.description[:100]}"
        required_inputs = [inp.name for inp in p.inputs if inp.required]
        optional_inputs = [inp.name for inp in p.inputs if not inp.required]
        if required_inputs or optional_inputs:
            line += "  (inputs:"
            if required_inputs:
                line += f" required={required_inputs}"
            if optional_inputs:
                line += f" optional={optional_inputs}"
            line += ")"
        if p.type_signature:
            line += f"  (type: {p.type_signature[:60]})"
        lines.append(line)
    return "\n".join(lines)


def _is_any_type(type_desc: str) -> bool:
    return type_desc.strip() in {"", "Any"}


def _tokenize(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9_]+", text.lower()) if len(tok) >= 3}


def _instantiate_template(
    parent: AlgorithmicNode,
    example: "ExampleDecomposition",
    all_nodes: list[AlgorithmicNode],
) -> tuple[list[AlgorithmicNode], list[DependencyEdge]]:
    """Instantiate a template from a retrieved example decomposition."""
    from sciona.architect.graph_retrieval import ExampleDecomposition

    parent_copy = parent.model_copy()
    parent_copy.status = NodeStatus.DECOMPOSED
    parent_copy.children = []

    new_nodes: list[AlgorithmicNode] = [parent_copy]
    new_edges: list[DependencyEdge] = []

    for i, child in enumerate(example.children):
        child_id = f"{parent.node_id}_t{i}"
        try:
            ct = ConceptType(child.concept_type)
        except ValueError:
            ct = parent.concept_type

        child_node = AlgorithmicNode(
            node_id=child_id,
            parent_id=parent.node_id,
            name=child.name,
            description=child.description,
            concept_type=ct,
            status=NodeStatus.ATOMIC,
            depth=parent.depth + 1,
            type_signature=child.type_signature,
            matched_primitive=child.matched_primitive or None,
        )
        new_nodes.append(child_node)
        parent_copy.children.append(child_id)

    # Re-create edges with new child IDs
    child_id_map = {
        old_child.node_id: f"{parent.node_id}_t{i}"
        for i, old_child in enumerate(example.children)
    }
    for edge in example.edges:
        new_source = child_id_map.get(edge.source_id)
        new_target = child_id_map.get(edge.target_id)
        if new_source and new_target:
            new_edges.append(
                DependencyEdge(
                    source_id=new_source,
                    target_id=new_target,
                    output_name=edge.output_name,
                    input_name=edge.input_name,
                    source_type="",
                    target_type="",
                )
            )

    return new_nodes, new_edges


def _instantiate_skeleton_for_node(
    parent: AlgorithmicNode,
    *,
    skeleton_name: str,
) -> tuple[list[AlgorithmicNode], list[DependencyEdge]]:
    """Instantiate a named skeleton as a conservative child decomposition."""
    skeleton = get_skeleton(parent.concept_type, variant=skeleton_name)
    if skeleton is None:
        raise ValueError(f"Unknown skeleton proposal: {skeleton_name}")

    parent_copy = parent.model_copy()
    parent_copy.status = NodeStatus.DECOMPOSED
    parent_copy.children = []

    child_nodes, child_edges = instantiate_skeleton(
        skeleton,
        parent.description or parent.name,
        parent_id=parent.node_id,
        base_depth=parent.depth + 1,
    )
    parent_copy.children.extend(child.node_id for child in child_nodes)
    return [parent_copy, *child_nodes], child_edges


def _select_live_skeleton_proposal(
    ranked_proposals: list[ScoredProposal],
) -> tuple[ScoredProposal | None, str, float]:
    """Select a live skeleton proposal only when it clears a strict margin."""
    if not ranked_proposals:
        return None, "no_ranked_proposals", 0.0

    top = ranked_proposals[0]
    if top.proposal.proposal_type != ProposalType.SKELETON:
        return None, "top_not_skeleton", 0.0
    if top.score <= 0.0:
        return None, "non_positive_score", 0.0

    lower_complexity = [
        candidate
        for candidate in ranked_proposals[1:]
        if candidate.complexity_penalty < top.complexity_penalty
    ]
    if not lower_complexity:
        return top, "accepted_no_lower_complexity_alternative", 0.0

    best_alternative = max(lower_complexity, key=lambda candidate: candidate.score)
    score_margin = top.score - best_alternative.score
    if score_margin < _SKELETON_ACCEPTANCE_MARGIN:
        return None, "insufficient_margin_over_simpler_alternative", score_margin
    return top, "accepted_margin_over_simpler_alternative", score_margin


def _lexical_primitive_fallback(
    node: AlgorithmicNode,
    deps: DecompositionDeps,
    *,
    k: int = 5,
) -> list:
    """Fallback primitive retrieval when semantic/category retrieval is empty."""
    catalog_prims = deps.catalog.all_primitives()
    if not catalog_prims:
        return []

    query_parts = [
        node.name,
        node.description,
        node.concept_type.value,
        " ".join(io.name for io in node.inputs),
        " ".join(io.type_desc for io in node.inputs),
        " ".join(io.name for io in node.outputs),
        " ".join(io.type_desc for io in node.outputs),
    ]
    query_tokens = _tokenize(" ".join(query_parts))
    if not query_tokens:
        return catalog_prims[:k]

    scored: list[tuple[float, Any]] = []
    for prim in catalog_prims:
        prim_text = " ".join(
            [
                prim.name,
                prim.description,
                prim.category.value,
                " ".join(f"{io.name} {io.type_desc}" for io in prim.inputs),
                " ".join(f"{io.name} {io.type_desc}" for io in prim.outputs),
            ]
        )
        prim_tokens = _tokenize(prim_text)
        overlap = len(query_tokens & prim_tokens)
        if overlap <= 0:
            continue
        score = float(overlap)
        if prim.category == node.concept_type:
            score += 2.0
        scored.append((score, prim))

    if not scored:
        return []
    scored.sort(key=lambda row: row[0], reverse=True)
    return [prim for _score, prim in scored[:k]]


def _parse_json(text: str) -> dict | None:
    """Try to parse JSON from LLM output, stripping markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Drop first and last lines (fences)
        lines = [line for line in lines[1:] if not line.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _context_namespace(config: RunnableConfig, deps: DecompositionDeps) -> str:
    base = deps.context_namespace.strip()
    if not base:
        return ""
    run_id = str(config.get("configurable", {}).get("run_id", "")).strip()
    return f"{base}/{run_id}" if run_id else base


def _template_namespace(config: RunnableConfig, deps: DecompositionDeps) -> str:
    base = _context_namespace(config, deps).strip().strip("/")
    if not base:
        return "architect/templates"
    if "/" in base:
        base = base.rsplit("/", 1)[0]
    return f"{base}/templates"


def _failure_namespace(config: RunnableConfig, deps: DecompositionDeps) -> str:
    base = _context_namespace(config, deps).strip().strip("/")
    if not base:
        return "architect/failures"
    if "/" in base:
        base = base.rsplit("/", 1)[0]
    return f"{base}/failures"


def _planning_artifact_block(state: DecompositionState) -> str:
    artifact = state.get("planning_artifact")
    if not artifact:
        return ""
    return render_planning_artifact_block(artifact)


async def _search_context(
    deps: DecompositionDeps,
    config: RunnableConfig,
    *,
    channel: str,
    query: str,
    limit: int = 3,
) -> str:
    store = deps.shared_context
    if store is None:
        return ""
    ns = _context_namespace(config, deps)
    if not ns:
        return ""
    try:
        records = await store.search(f"{ns}/{channel}", query, limit=limit)
        return format_context_block(
            "Shared Context",
            records,
            max_chars=deps.context_budget_chars,
            metrics=deps.shared_context_metrics,
        )
    except Exception:
        return ""


async def _search_template_context(
    deps: DecompositionDeps,
    config: RunnableConfig,
    *,
    query: str,
    limit: int = 3,
) -> str:
    store = deps.shared_context
    if store is None:
        return ""
    ns = _template_namespace(config, deps)
    if not ns:
        return ""
    try:
        records = await store.search(ns, query, limit=limit)
        if deps.shared_context_metrics is not None:
            deps.shared_context_metrics.record_template_search(hits=len(records))
        block = format_context_block(
            "Prior Decomposition Templates",
            records,
            max_chars=deps.context_budget_chars,
            metrics=deps.shared_context_metrics,
        )
        if deps.shared_context_metrics is not None:
            deps.shared_context_metrics.record_template_injection(
                chars=len(block),
                records=len(records),
            )
        return block
    except Exception:
        return ""


async def _search_failure_context(
    deps: DecompositionDeps,
    config: RunnableConfig,
    *,
    query: str,
    limit: int = 3,
) -> str:
    store = deps.shared_context
    if store is None:
        return ""
    ns = _failure_namespace(config, deps)
    if not ns:
        return ""
    try:
        records = await store.search(ns, query, limit=limit)
        if deps.shared_context_metrics is not None:
            deps.shared_context_metrics.record_failure_search(hits=len(records))
        block = format_context_block(
            "Prior Failure Patterns",
            records,
            max_chars=deps.context_budget_chars,
            metrics=deps.shared_context_metrics,
        )
        if deps.shared_context_metrics is not None:
            deps.shared_context_metrics.record_failure_injection(
                chars=len(block),
                records=len(records),
            )
        return block
    except Exception:
        return ""


async def _put_context(
    deps: DecompositionDeps,
    config: RunnableConfig,
    *,
    channel: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    store = deps.shared_context
    if store is None:
        return
    ns = _context_namespace(config, deps)
    if not ns:
        return
    try:
        await store.put(f"{ns}/{channel}", text, metadata=metadata)
    except Exception:
        return


async def _put_template_context(
    deps: DecompositionDeps,
    config: RunnableConfig,
    *,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    store = deps.shared_context
    if store is None:
        return
    ns = _template_namespace(config, deps)
    if not ns:
        return
    payload = dict(metadata or {})
    payload.setdefault("confidence", 0.95)
    try:
        await store.put(ns, text, metadata=payload)
        if deps.shared_context_metrics is not None:
            deps.shared_context_metrics.record_template_put()
    except Exception:
        return


async def _put_failure_context(
    deps: DecompositionDeps,
    config: RunnableConfig,
    *,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    store = deps.shared_context
    if store is None:
        return
    ns = _failure_namespace(config, deps)
    if not ns:
        return
    payload = dict(metadata or {})
    payload.setdefault("confidence", 0.9)
    try:
        await store.put(ns, text, metadata=payload)
        if deps.shared_context_metrics is not None:
            deps.shared_context_metrics.record_failure_put()
    except Exception:
        return


# ---------------------------------------------------------------------------
# Node: select_strategy
# ---------------------------------------------------------------------------


async def select_strategy(
    state: DecompositionState, config: RunnableConfig
) -> dict[str, Any]:
    """Entry point: LLM picks a paradigm and bootstraps the CDG via skeleton.

    Before calling the LLM, performs a deterministic pre-scan for known
    conjugate prior/likelihood pairs.  When one is found the graph can
    short-circuit the iterative decompose/critique loop entirely — see
    ``route_after_strategy`` and ``advance_conjugate_node``.
    """
    deps = _get_deps(config)
    goal = state["goal"]

    # ------------------------------------------------------------------
    # Conjugate pre-scan (deterministic, free — runs before LLM call)
    # ------------------------------------------------------------------
    conjugate_spec = _detect_conjugate_pair(goal)
    if conjugate_spec is not None:
        await _put_context(
            deps,
            config,
            channel="strategy",
            text=(
                f"Goal: {goal}\n"
                f"Paradigm: conjugate_update\n"
                f"Conjugate pair: {conjugate_spec['pair_name']}"
            ),
            metadata={"paradigm": "conjugate_update"},
        )
        # Signal the router to short-circuit into advance_conjugate_node
        root_id = f"root_{uuid.uuid4().hex[:8]}"
        root = AlgorithmicNode(
            node_id=root_id,
            name=goal,
            description=goal,
            concept_type=ConceptType.CONJUGATE_UPDATE,
            status=NodeStatus.DECOMPOSED,
            depth=0,
        )
        planning_artifact = build_planning_artifact(
            goal=goal,
            thread_id=str(config.get("configurable", {}).get("thread_id", "")),
            paradigm="conjugate_update",
            family_hint="conjugate_update",
            strategy_rationale="Deterministic conjugate-pair short circuit.",
            skeleton_instantiated=False,
            root_inputs=[],
            root_outputs=[],
            assumptions=[f"Detected conjugate pair: {conjugate_spec['pair_name']}"],
            unresolved_questions=[
                "Conjugate fast-path should still preserve the planning contract."
            ],
            extra_constraints=[
                PlanningConstraint(
                    category=PlanningConstraintCategory.STAGE,
                    subject="conjugate_update",
                    statement="Short-circuit directly into conjugate update flow.",
                    source_stage="strategy",
                ),
                PlanningConstraint(
                    category=PlanningConstraintCategory.ADMISSIBILITY,
                    subject="conjugate_update",
                    statement=(
                        "Only apply the short-circuit when the detected pair is "
                        "unambiguously conjugate."
                    ),
                    source_stage="strategy",
                ),
            ],
        )
        planning_artifact_data = planning_artifact.model_dump(mode="json")
        history_entry = {
            "step": "select_strategy",
            "paradigm": "conjugate_update",
            "conjugate_pair": conjugate_spec["pair_name"],
            "short_circuit": True,
            "planning_artifact": summarize_planning_artifact(planning_artifact_data),
        }
        _publish_architect_snapshot(
            phase="strategy",
            node_id=root_id,
            nodes=[root],
            edges=[],
            extra={"planning_artifact": summarize_planning_artifact(planning_artifact_data)},
        )
        return {
            "nodes": [root],
            "edges": [],
            "history": [history_entry],
            "planning_artifact": planning_artifact_data,
            "pending_node_ids": [],
            "current_node_id": root_id,
            "paradigm": "conjugate_update",
            "skeleton_instantiated": False,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
            # Stash the spec for advance_conjugate_node to consume.
            # This key is ignored by the state reducer (it's not in the
            # TypedDict), but it is present in the dict returned by the
            # node and lands in the state snapshot for the *next* node
            # via config forwarding.  We also pass it through the
            # ``_conjugate_spec`` field that the router will read.
        }

    # ------------------------------------------------------------------
    # Standard LLM-based strategy selection
    # ------------------------------------------------------------------
    available = list(SKELETON_TEMPLATES.keys())
    available_str = "\n".join(f"  - {ct.value}" for ct in available)
    user_prompt = SELECT_STRATEGY_USER.format(goal=goal)
    shared_block = await _search_context(
        deps,
        config,
        channel="strategy",
        query=goal,
        limit=3,
    )
    if shared_block:
        user_prompt += f"\n\n{shared_block}"

    response = await select_llm(deps.llm, ARCHITECT_STRATEGY).complete(
        SELECT_STRATEGY_SYSTEM.format(available_paradigms=available_str),
        user_prompt,
    )

    parsed = _parse_json(response)

    # Parse paradigm or fall back to CUSTOM
    paradigm = ConceptType.CUSTOM
    variant_hint = ""
    if parsed:
        paradigm_str = parsed.get("paradigm", "")
        for ct in ConceptType:
            if ct.value == paradigm_str:
                paradigm = ct
                break
        variant_hint = parsed.get("variant_hint", "")

    # Create root node
    root_id = f"root_{uuid.uuid4().hex[:8]}"
    root = AlgorithmicNode(
        node_id=root_id,
        name=goal,
        description=goal,
        concept_type=paradigm,
        status=NodeStatus.PENDING,
        depth=0,
    )

    nodes: list[AlgorithmicNode] = [root]
    edges: list[DependencyEdge] = []
    skeleton_instantiated = False

    # Try to instantiate skeleton
    skeleton = get_skeleton(paradigm, variant=variant_hint or None)
    if skeleton:
        skel_nodes, skel_edges = instantiate_skeleton(
            skeleton, goal, parent_id=root_id, base_depth=0
        )
        root_inputs, root_outputs = infer_boundary_ports(skel_nodes, skel_edges)
        root = root.model_copy(
            update={
                "status": NodeStatus.DECOMPOSED,
                "children": [n.node_id for n in skel_nodes],
                "inputs": root_inputs,
                "outputs": root_outputs,
            }
        )
        nodes = [root] + skel_nodes
        edges = skel_edges
        skeleton_instantiated = True

    # Check which skeleton nodes are already atomic
    for i, node in enumerate(nodes):
        if node.node_id == root_id:
            continue
        if deps.catalog.is_atomic(node):
            # Find the matching primitive name
            prim_name = node.matched_primitive
            if not prim_name:
                primitive = deps.catalog.get(node.name)
                if primitive is not None:
                    prim_name = primitive.name
            nodes[i] = node.model_copy(
                update={
                    "status": NodeStatus.ATOMIC,
                    "matched_primitive": prim_name or node.name,
                }
            )

    # Build pending queue. Without a skeleton, the root itself must be decomposed.
    pending = [n.node_id for n in nodes if n.status == NodeStatus.PENDING]

    current_node_id = pending[0] if pending else ""
    planning_artifact = build_planning_artifact(
        goal=goal,
        thread_id=str(config.get("configurable", {}).get("thread_id", "")),
        paradigm=paradigm.value,
        family_hint=paradigm.value,
        strategy_rationale=str(parsed.get("rationale", "") if parsed else ""),
        variant_hint=variant_hint,
        skeleton_instantiated=skeleton_instantiated,
        root_inputs=root.inputs,
        root_outputs=root.outputs,
        assumptions=[
            "Root boundary contracts were derived from the instantiated skeleton.",
        ]
        if skeleton_instantiated
        else [
            "Root boundary contracts are still provisional until decomposition instantiates them.",
        ],
        unresolved_questions=[
            "Family-specific refinements should preserve declared planning constraints."
        ],
    )
    planning_artifact_data = planning_artifact.model_dump(mode="json")
    planning_summary = summarize_planning_artifact(planning_artifact_data)

    history_entry = {
        "step": "select_strategy",
        "paradigm": paradigm.value,
        "variant_hint": variant_hint,
        "skeleton_instantiated": skeleton_instantiated,
        "num_nodes": len(nodes),
        "num_pending": len(pending),
        "planning_artifact": planning_summary,
    }

    await _put_context(
        deps,
        config,
        channel="strategy",
        text=(
            f"Goal: {goal}\n"
            f"Paradigm: {paradigm.value}\n"
            f"Variant: {variant_hint or '(none)'}\n"
            f"Skeleton: {skeleton_instantiated}"
        ),
        metadata={"paradigm": paradigm.value},
    )
    _publish_architect_snapshot(
        phase="strategy",
        node_id=current_node_id or root_id,
        nodes=nodes,
        edges=edges,
        extra={"planning_artifact": planning_summary},
    )

    return {
        "nodes": nodes,
        "edges": edges,
        "history": [history_entry],
        "planning_artifact": planning_artifact_data,
        "pending_node_ids": pending,
        "current_node_id": current_node_id,
        "paradigm": paradigm.value,
        "skeleton_instantiated": skeleton_instantiated,
        "critique_passed": False,
        "critique_reason": "",
        "critique_retries": 0,
        "done": len(pending) == 0,
        "error": "",
    }


# ---------------------------------------------------------------------------
# Node: decompose_node
# ---------------------------------------------------------------------------


async def decompose_node(
    state: DecompositionState, config: RunnableConfig
) -> dict[str, Any]:
    """Core LLM decomposition: break current_node_id into sub-nodes + edges."""
    deps = _get_deps(config)
    current_id = state["current_node_id"]
    all_nodes = state["nodes"]
    max_depth = state["max_depth"]

    node = _find_node(all_nodes, current_id)
    if node is None:
        return {
            "error": f"Node {current_id} not found",
            "done": True,
            "history": [
                {"step": "decompose_node", "error": f"Node {current_id} not found"}
            ],
        }

    # Gather relevant primitives
    catalog_prims = deps.catalog.find_matching_primitives(node, k=5)
    lexical_prims = _lexical_primitive_fallback(node, deps, k=5)
    try:
        skill_prims = deps.skill_index.search(f"{node.name} {node.description}", k=5)
    except Exception:
        skill_prims = []

    # Deduplicate by name
    seen_names: set[str] = set()
    all_prims = []
    for p in lexical_prims + catalog_prims + skill_prims:
        if p.name not in seen_names:
            all_prims.append(p)
            seen_names.add(p.name)
    primitive_proposals = [
        proposal_from_primitive(primitive) for primitive in all_prims
    ]
    skeleton_proposals = generate_skeleton_proposals(node)

    # Build retry context
    retry_context = ""
    retries = state.get("critique_retries", 0)
    if retries > 0:
        reason = state.get("critique_reason", "")
        retry_context = (
            f"IMPORTANT: This is retry #{retries}. "
            f"Previous decomposition was rejected: {reason}\n"
            "Please fix the issues and try again."
        )
    planning_context = _planning_artifact_block(state) or "(none)"

    # Try template retriever for high-confidence instantiation (skip LLM)
    template_retriever = getattr(deps, "template_retriever", None)
    template_matches = []
    template_proposals = []
    if template_retriever is not None:
        try:
            template_matches = await template_retriever.find_templates(
                node, all_nodes, state["edges"]
            )
            template_proposals = [
                proposal_from_template_match(match) for match in template_matches[:3]
            ]
        except Exception:
            logger.debug("template_retriever failed in decompose_node", exc_info=True)
            template_proposals = []

    ranked_proposals = rank_proposals(
        primitive_proposals + template_proposals + skeleton_proposals,
        preferred_family=node.concept_type.value,
    )
    accepted_skeleton, skeleton_acceptance_reason, skeleton_margin = (
        _select_live_skeleton_proposal(ranked_proposals)
    )
    if accepted_skeleton is not None:
        skeleton_name = str(accepted_skeleton.proposal.skeleton_name or "")
        log_event(
            "architect",
            "decompose",
            "SKELETON_PROPOSAL_USED",
            payload={
                "node_id": node.node_id,
                "skeleton_name": skeleton_name,
                "score": round(accepted_skeleton.score, 3),
                "margin": round(skeleton_margin, 3),
                "source_family": accepted_skeleton.proposal.source_family,
            },
        )
        new_nodes, new_edges = _instantiate_skeleton_for_node(
            node,
            skeleton_name=skeleton_name,
        )
        return {
            "nodes": new_nodes,
            "edges": new_edges,
            "history": [
                {
                    "step": "decompose_node",
                    "node_id": node.node_id,
                    "selected_proposal_type": accepted_skeleton.proposal.proposal_type.value,
                    "selected_skeleton_name": skeleton_name,
                    "selected_proposal_score": round(accepted_skeleton.score, 3),
                    "skeleton_acceptance_reason": skeleton_acceptance_reason,
                    "skeleton_acceptance_margin": round(skeleton_margin, 3),
                    "primitive_proposal_count": len(primitive_proposals),
                    "template_proposal_count": len(template_proposals),
                    "skeleton_proposal_count": len(skeleton_proposals),
                    "top_ranked_proposal_type": (
                        ranked_proposals[0].proposal.proposal_type.value
                        if ranked_proposals
                        else ""
                    ),
                    "top_ranked_proposal_score": (
                        round(ranked_proposals[0].score, 3)
                        if ranked_proposals
                        else 0.0
                    ),
                    "ranked_proposal_types": [
                        scored.proposal.proposal_type.value
                        for scored in ranked_proposals[:3]
                    ],
                }
            ],
        }

    if template_matches and template_matches[0].confidence >= 0.8:
        match = template_matches[0]
        template_proposal = proposal_from_template_match(match)
        log_event(
            "architect",
            "decompose",
            "TEMPLATE_USED",
            payload={
                "node_id": node.node_id,
                "template_fqn": match.example.fqn,
                "confidence": round(match.confidence, 3),
                "source": match.source,
            },
        )
        new_nodes, new_edges = _instantiate_template(node, match.example, all_nodes)
        return {
            "nodes": new_nodes,
            "edges": new_edges,
            "history": [
                {
                    "step": "decompose_node",
                    "node_id": node.node_id,
                    "template_used": match.example.fqn,
                    "confidence": round(match.confidence, 3),
                    "selected_proposal_type": template_proposal.proposal_type.value,
                    "skeleton_acceptance_reason": skeleton_acceptance_reason,
                    "skeleton_acceptance_margin": round(skeleton_margin, 3),
                    "primitive_proposal_count": len(primitive_proposals),
                    "template_proposal_count": len(template_proposals),
                    "skeleton_proposal_count": len(skeleton_proposals),
                    "top_ranked_proposal_type": (
                        ranked_proposals[0].proposal.proposal_type.value
                        if ranked_proposals
                        else ""
                    ),
                    "top_ranked_proposal_score": (
                        round(ranked_proposals[0].score, 3)
                        if ranked_proposals
                        else 0.0
                    ),
                    "ranked_proposal_types": [
                        scored.proposal.proposal_type.value
                        for scored in ranked_proposals[:3]
                    ],
                }
            ],
        }
    if template_matches:
        log_event(
            "architect",
            "decompose",
            "TEMPLATE_FALLBACK_TO_LLM",
            payload={
                "node_id": node.node_id,
                "best_confidence": round(template_matches[0].confidence, 3),
                "skeleton_acceptance_reason": skeleton_acceptance_reason,
            },
        )

    # Retrieve similar decomposition examples from Memgraph
    example_decompositions = ""
    graph_retriever = getattr(deps, "graph_retriever", None)
    if graph_retriever is not None:
        from sciona.architect.graph_retrieval import format_examples_for_prompt

        examples = await graph_retriever.find_similar(node, all_nodes, state["edges"])
        if examples:
            example_decompositions = format_examples_for_prompt(examples)

    user_prompt = DECOMPOSE_NODE_USER.format(
        node_name=node.name,
        node_description=node.description,
        concept_type=node.concept_type.value,
        inputs=_format_io(node.inputs),
        outputs=_format_io(node.outputs),
        depth=node.depth,
        max_depth=max_depth,
        planning_context=planning_context,
        primitives=_format_primitives(all_prims),
        example_decompositions=example_decompositions,
        retry_context=retry_context,
    )
    shared_block = await _search_context(
        deps,
        config,
        channel="decompose",
        query=f"{node.name} {node.description} {node.concept_type.value}",
        limit=3,
    )
    if shared_block:
        user_prompt += f"\n\n{shared_block}"
    template_block = await _search_template_context(
        deps,
        config,
        query=f"{node.name} {node.description} {node.concept_type.value}",
        limit=2,
    )
    if template_block:
        user_prompt += f"\n\n{template_block}"
    failure_block = await _search_failure_context(
        deps,
        config,
        query=f"{node.name} {node.description} {node.concept_type.value}",
        limit=2,
    )
    if failure_block:
        user_prompt += f"\n\n{failure_block}"

    response = await select_llm(deps.llm, ARCHITECT_DECOMPOSE).complete(
        DECOMPOSE_NODE_SYSTEM,
        user_prompt,
    )

    parsed = _parse_json(response)
    if not parsed:
        # Fallback: empty decomposition → critic rejects → retry
        return {
            "nodes": [],
            "edges": [],
            "history": [
                {"step": "decompose_node", "node_id": current_id, "parse_error": True}
            ],
        }

    progress_updates_raw = parsed.get("progress_updates", [])
    progress_updates: list[str] = []
    if isinstance(progress_updates_raw, list):
        for item in progress_updates_raw:
            text = str(item).strip()
            if text:
                progress_updates.append(text)
    if progress_updates:
        stage = get_current_stage() or "architect_decompose"
        first_update = progress_updates[0][:180]
        update_stage(
            stage=stage,
            status="running",
            message=f"{node.name}: {first_update}",
        )
        log_event(
            "architect",
            "decompose",
            "DECOMPOSE_PROGRESS",
            node_id=current_id,
            payload={"progress_updates": progress_updates[:6]},
        )
        await _put_context(
            deps,
            config,
            channel="decompose_progress",
            text=(
                f"Node: {node.name}\n"
                f"Progress: {' | '.join(progress_updates[:6])}"
            ),
            metadata={"node_id": current_id, "node_name": node.name},
        )

    use_monadic_rewriter = False
    if config and config.get("configurable") and config["configurable"].get("deps"):
        deps_obj = config["configurable"]["deps"]
        use_monadic_rewriter = getattr(deps_obj, "use_monadic_rewriter", False)

    try:
        built = build_deterministic_decomposition(
            parsed=parsed,
            parent=node,
            catalog=deps.catalog,
            use_monadic_rewriter=use_monadic_rewriter,
        )
    except DeterministicRewriteError as exc:
        return {
            "nodes": [],
            "edges": [],
            "error": str(exc),
            "history": [
                {
                    "step": "decompose_node",
                    "node_id": current_id,
                    "rewrite_error": str(exc),
                }
            ],
        }
    new_nodes = built.nodes
    new_edges = built.edges
    rewrite_actions = list(getattr(built, "rewrite_actions", []))

    history_entry = {
        "step": "decompose_node",
        "node_id": current_id,
        "num_sub_nodes": len(new_nodes),
        "num_edges": len(new_edges),
        "rewrite_actions": rewrite_actions,
        "skeleton_acceptance_reason": skeleton_acceptance_reason,
        "skeleton_acceptance_margin": round(skeleton_margin, 3),
        "primitive_proposal_count": len(primitive_proposals),
        "template_proposal_count": len(template_proposals),
        "skeleton_proposal_count": len(skeleton_proposals),
        "top_ranked_proposal_type": (
            ranked_proposals[0].proposal.proposal_type.value
            if ranked_proposals
            else ""
        ),
        "top_ranked_proposal_score": (
            round(ranked_proposals[0].score, 3)
            if ranked_proposals
            else 0.0
        ),
        "ranked_proposal_types": [
            scored.proposal.proposal_type.value for scored in ranked_proposals[:3]
        ],
    }

    await _put_context(
        deps,
        config,
        channel="decompose",
        text=(
            f"Parent: {node.name}\n"
            f"Children: {', '.join(n.name for n in new_nodes) or '(none)'}\n"
            f"Edges: {len(new_edges)}\n"
            f"Rewrite actions: {len(rewrite_actions)}\n"
            f"Retry: {retries}"
        ),
        metadata={"node_id": current_id, "node_name": node.name},
    )

    _publish_architect_snapshot(
        phase="decompose",
        node_id=current_id,
        nodes=all_nodes + new_nodes,
        edges=state["edges"] + new_edges,
        extra={"last_node_name": node.name, "rewrite_actions": rewrite_actions},
    )

    return {
        "nodes": new_nodes,
        "edges": new_edges,
        "history": [history_entry],
    }


# ---------------------------------------------------------------------------
# Node: critique_decomposition
# ---------------------------------------------------------------------------


async def critique_decomposition(
    state: DecompositionState, config: RunnableConfig
) -> dict[str, Any]:
    """Two-phase validation: deterministic checks first, then LLM critique."""
    deps = _get_deps(config)
    current_id = state["current_node_id"]
    all_nodes = state["nodes"]
    max_depth = state["max_depth"]

    parent = _find_node(all_nodes, current_id)
    if parent is None:
        return {
            "critique_passed": False,
            "critique_reason": f"Parent node {current_id} not found",
            "history": [{"step": "critique", "error": "parent not found"}],
        }

    # Find children of current node
    children = [
        n
        for n in all_nodes
        if n.parent_id == current_id and n.status != NodeStatus.REJECTED
    ]
    child_edges = [
        e
        for e in state["edges"]
        if e.source_id in {c.node_id for c in children}
        or e.target_id in {c.node_id for c in children}
    ]

    # ------------------------------------------------------------------
    # Phase A: Deterministic checks (fast, free)
    # ------------------------------------------------------------------
    issues = structural_critique_issues(
        parent,
        children,
        child_edges,
        max_depth=max_depth,
        catalog=deps.catalog,
    )

    if issues:
        reason = "Deterministic checks failed: " + "; ".join(issues)
        category = _critique_reason_category(reason)
        increment_run_metadata_counter(
            "architect_metrics",
            "critique_reject_counts_by_category",
            category,
        )
        log_event(
            "architect",
            "critique",
            "ARCHITECT_CRITIQUE_REJECTED",
            node_id=current_id,
            payload={"category": category, "reason": reason, "phase": "deterministic"},
        )
        await _put_context(
            deps,
            config,
            channel="critique",
            text=(
                f"Parent: {parent.name}\n"
                f"Approved: False\n"
                f"Reason: {reason}"
            ),
            metadata={"node_id": current_id, "phase": "deterministic"},
        )
        await _put_failure_context(
            deps,
            config,
            text=(
                f"Parent: {parent.name}\n"
                f"Description: {parent.description}\n"
                f"Category: {category}\n"
                f"Reason: {reason}"
            ),
            metadata={
                "node_id": current_id,
                "node_name": parent.name,
                "phase": "deterministic",
                "category": category,
            },
        )
        return {
            "critique_passed": False,
            "critique_reason": reason,
            "history": [
                {"step": "critique", "phase": "deterministic", "issues": issues}
            ],
        }

    if not getattr(deps, "architect_critique_llm_enabled", True):
        reason = "Deterministic structural checks passed"
        await _put_context(
            deps,
            config,
            channel="critique",
            text=(
                f"Parent: {parent.name}\n"
                "Approved: True\n"
                f"Reason: {reason}"
            ),
            metadata={"node_id": current_id, "phase": "deterministic_only"},
        )
        return {
            "critique_passed": True,
            "critique_reason": reason,
            "history": [
                {
                    "step": "critique",
                    "phase": "deterministic_only",
                    "approved": True,
                    "reason": reason,
                }
            ],
        }

    # ------------------------------------------------------------------
    # Phase B: LLM critique (only if enabled and Phase A passes)
    # ------------------------------------------------------------------
    sub_nodes_str = "\n".join(
        f"  - {c.name} [{c.concept_type.value}] "
        f"(inputs: {_format_io(c.inputs)}, outputs: {_format_io(c.outputs)}, "
        f"status: {c.status.value}, matched_primitive: {c.matched_primitive or '(none)'})"
        for c in children
    )
    edges_str = "\n".join(
        f"  - {e.source_id[:12]}... -> {e.target_id[:12]}...: "
        f"{e.output_name} -> {e.input_name} ({e.source_type})"
        for e in child_edges
    )

    catalog_prims = deps.catalog.find_matching_primitives(parent, k=5)
    child_prims = [
        deps.catalog.get(child.matched_primitive or "")
        for child in children
        if child.matched_primitive
    ]
    seen_primitive_names: set[str] = set()
    critique_primitives = []
    for primitive in [*catalog_prims, *child_prims]:
        if primitive is None or primitive.name in seen_primitive_names:
            continue
        seen_primitive_names.add(primitive.name)
        critique_primitives.append(primitive)
    critique_prompt = CRITIQUE_USER.format(
        parent_name=parent.name,
        parent_description=parent.description,
        parent_inputs=_format_io(parent.inputs),
        parent_outputs=_format_io(parent.outputs),
        planning_context=_planning_artifact_block(state) or "(none)",
        sub_nodes=sub_nodes_str,
        edges=edges_str or "  (no edges)",
        current_depth=parent.depth,
        max_depth=max_depth,
        primitives=_format_primitives(critique_primitives),
    )
    shared_block = await _search_context(
        deps,
        config,
        channel="critique",
        query=f"{parent.name} {parent.description}",
        limit=3,
    )
    if shared_block:
        critique_prompt += f"\n\n{shared_block}"

    response = await select_llm(deps.llm, ARCHITECT_CRITIQUE).complete(
        CRITIQUE_SYSTEM,
        critique_prompt,
    )

    parsed = _parse_json(response)

    if parsed is None:
        # Deterministic checks already passed, so malformed critique should not
        # block progress solely due LLM formatting.
        await _put_context(
            deps,
            config,
            channel="critique",
            text=(
                f"Parent: {parent.name}\n"
                "Approved: True\n"
                "Reason: LLM critique parse failed; accepted by deterministic checks"
            ),
            metadata={"node_id": current_id, "phase": "llm", "parse_error": True},
        )
        return {
            "critique_passed": True,
            "critique_reason": "LLM critique parse failed; accepted by deterministic checks",
            "history": [{"step": "critique", "phase": "llm", "parse_error": True}],
        }

    approved_raw = parsed.get("approved")
    if not isinstance(approved_raw, bool):
        # Same fail-open behavior for wrong JSON shape (e.g., missing "approved").
        await _put_context(
            deps,
            config,
            channel="critique",
            text=(
                f"Parent: {parent.name}\n"
                "Approved: True\n"
                "Reason: LLM critique had invalid schema; accepted by deterministic checks"
            ),
            metadata={"node_id": current_id, "phase": "llm", "schema_error": True},
        )
        return {
            "critique_passed": True,
            "critique_reason": "LLM critique had invalid schema; accepted by deterministic checks",
            "history": [
                {
                    "step": "critique",
                    "phase": "llm",
                    "schema_error": True,
                    "keys": sorted(parsed.keys()),
                }
            ],
        }

    approved = approved_raw
    reason = parsed.get("reason", "")
    flagged = parsed.get("flagged_nodes", [])
    if not approved:
        category = _critique_reason_category(reason)
        increment_run_metadata_counter(
            "architect_metrics",
            "critique_reject_counts_by_category",
            category,
        )
        log_event(
            "architect",
            "critique",
            "ARCHITECT_CRITIQUE_REJECTED",
            node_id=current_id,
            payload={"category": category, "reason": reason, "phase": "llm"},
        )

    # Only rejected critiques should downgrade children. Some models emit
    # advisory flagged_nodes even when approved=true; treating those as hard
    # downgrades leaves primitive-bound leaves unresolved.
    flagged_updates: list[AlgorithmicNode] = []
    if flagged and not approved:
        for child in children:
            if child.name in flagged:
                flagged_updates.append(
                    child.model_copy(update={"status": NodeStatus.HIGH_RISK})
                )

    result: dict[str, Any] = {
        "critique_passed": approved,
        "critique_reason": reason,
        "history": [
            {"step": "critique", "phase": "llm", "approved": approved, "reason": reason}
        ],
    }
    if flagged_updates:
        result["nodes"] = flagged_updates

    await _put_context(
        deps,
        config,
        channel="critique",
        text=(
            f"Parent: {parent.name}\n"
            f"Approved: {approved}\n"
            f"Reason: {reason or '(none)'}\n"
            f"Flagged: {', '.join(flagged) if flagged else '(none)'}"
        ),
        metadata={"node_id": current_id, "phase": "llm", "approved": approved},
    )
    if not approved:
        category = _critique_reason_category(reason)
        await _put_failure_context(
            deps,
            config,
            text=(
                f"Parent: {parent.name}\n"
                f"Description: {parent.description}\n"
                f"Category: {category}\n"
                f"Reason: {reason or '(none)'}\n"
                f"Flagged: {', '.join(flagged) if flagged else '(none)'}"
            ),
            metadata={
                "node_id": current_id,
                "node_name": parent.name,
                "phase": "llm",
                "category": category,
                "approved": False,
            },
        )

    return result


# ---------------------------------------------------------------------------
# Node: advance_node
# ---------------------------------------------------------------------------


async def advance_node(
    state: DecompositionState, config: RunnableConfig
) -> dict[str, Any]:
    """After approved critique or max-retries exhaustion: move to next node."""
    deps = _get_deps(config)
    current_id = state["current_node_id"]
    all_nodes = state["nodes"]
    pending = list(state["pending_node_ids"])
    critique_passed = state.get("critique_passed", False)

    parent = _find_node(all_nodes, current_id)
    updated_nodes: list[AlgorithmicNode] = []

    if critique_passed and parent:
        # Update parent: mark as DECOMPOSED with children
        children = [
            n
            for n in all_nodes
            if n.parent_id == current_id and n.status != NodeStatus.REJECTED
        ]
        child_ids = [c.node_id for c in children]
        updated_nodes.append(
            parent.model_copy(
                update={
                    "status": NodeStatus.DECOMPOSED,
                    "children": child_ids,
                }
            )
        )

        # Add non-atomic children to pending
        new_pending = [c.node_id for c in children if c.status == NodeStatus.PENDING]
        child_summaries = [
            (
                f"{child.name}"
                f"{f' [{child.matched_primitive}]' if child.matched_primitive else ''}"
            )
            for child in children
        ]
        await _put_template_context(
            deps,
            config,
            text=(
                f"Parent: {parent.name}\n"
                f"Description: {parent.description}\n"
                f"Children: {', '.join(child_summaries) or '(none)'}\n"
                f"Outputs: {', '.join(port.name for port in parent.outputs) or '(none)'}"
            ),
            metadata={
                "node_id": current_id,
                "node_name": parent.name,
                "child_count": len(children),
                "approved": True,
                "channel": "templates",
            },
        )
    else:
        new_pending = []

    # Remove current from pending
    if current_id in pending:
        pending.remove(current_id)

    # Add new pending nodes
    pending.extend(new_pending)

    # Pick next
    next_id = pending[0] if pending else ""
    merged_nodes = all_nodes + updated_nodes
    blocked_pending = _pending_under_blocked(merged_nodes)
    done = len(pending) == 0
    error = ""
    if blocked_pending:
        names = ", ".join(node.name for node in blocked_pending[:5])
        error = f"Invalid architect state: pending nodes under blocked parent ({names})"
        done = True

    _publish_architect_snapshot(
        phase="advance",
        node_id=current_id,
        nodes=merged_nodes,
        edges=state["edges"],
        extra={"last_node_name": parent.name if parent else current_id},
    )

    return {
        "nodes": updated_nodes,
        "pending_node_ids": pending,
        "current_node_id": next_id,
        "critique_retries": 0,
        "critique_passed": False,
        "critique_reason": "",
        "done": done,
        "error": error,
        "history": [
            {
                "step": "advance_node",
                "from_node": current_id,
                "next_node": next_id,
                "done": done,
                "error": error,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Node: prepare_retry
# ---------------------------------------------------------------------------


async def prepare_retry(
    state: DecompositionState, config: RunnableConfig
) -> dict[str, Any]:
    """After rejected critique: increment retries, mark children REJECTED, loop back."""
    current_id = state["current_node_id"]
    all_nodes = state["nodes"]
    retries = state.get("critique_retries", 0)

    # Mark all current children rejected so retries replace prior attempts.
    rejected_updates: list[AlgorithmicNode] = []
    for node in all_nodes:
        if node.parent_id == current_id and node.status != NodeStatus.REJECTED:
            rejected_updates.append(
                node.model_copy(update={"status": NodeStatus.REJECTED})
            )

    parent = _find_node(all_nodes, current_id)
    if parent is not None:
        increment_run_metadata_counter(
            "architect_metrics",
            "retry_counts_by_node",
            parent.name,
        )

    return {
        "nodes": rejected_updates,
        "critique_retries": retries + 1,
        "history": [
            {
                "step": "prepare_retry",
                "node_id": current_id,
                "retry_num": retries + 1,
                "num_rejected": len(rejected_updates),
            }
        ],
    }


async def block_node(
    state: DecompositionState, config: RunnableConfig
) -> dict[str, Any]:
    """Terminate decomposition when critique retries are exhausted."""
    current_id = state["current_node_id"]
    all_nodes = state["nodes"]
    reason = state.get("critique_reason", "").strip() or "Critique retries exhausted"
    parent = _find_node(all_nodes, current_id)

    descendant_ids = _descendant_ids(all_nodes, current_id)
    blocked_updates: list[AlgorithmicNode] = []
    for node in all_nodes:
        if node.node_id == current_id and parent is not None:
            blocked_updates.append(
                parent.model_copy(
                    update={
                        "status": NodeStatus.BLOCKED,
                        "critic_notes": reason,
                    }
                )
            )
            continue
        if node.node_id in descendant_ids and node.status != NodeStatus.REJECTED:
            blocked_updates.append(
                node.model_copy(
                    update={
                        "status": NodeStatus.REJECTED,
                        "critic_notes": f"Discarded after parent blocked: {reason}",
                    }
                )
            )

    merged_nodes = all_nodes + blocked_updates
    _publish_architect_snapshot(
        phase="block",
        node_id=current_id,
        nodes=merged_nodes,
        edges=state["edges"],
        extra={
            "last_node_name": parent.name if parent else current_id,
            "blocked_reason": reason,
        },
    )
    log_event(
        "architect",
        "block",
        "ARCHITECT_BLOCKED",
        node_id=current_id,
        payload={"reason": reason, "node_name": parent.name if parent else current_id},
    )

    return {
        "nodes": blocked_updates,
        "pending_node_ids": [],
        "current_node_id": "",
        "critique_retries": 0,
        "critique_passed": False,
        "done": True,
        "error": f"Architect decomposition blocked at '{parent.name if parent else current_id}': {reason}",
        "history": [
            {
                "step": "block_node",
                "node_id": current_id,
                "reason": reason,
                "num_discarded": max(len(blocked_updates) - 1, 0),
            }
        ],
    }


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def route_after_critic(state: DecompositionState) -> str:
    """Conditional edge after critique_decomposition."""
    passed = state.get("critique_passed", False)
    retries = state.get("critique_retries", 0)

    if not passed and retries < 3:
        return "retry_decompose"
    if not passed:
        return "block_node"
    return "next_node"


def route_after_strategy(state: DecompositionState) -> str:
    """Conditional edge after select_strategy.

    If a conjugate pair was detected, skip the iterative decompose/critique
    loop and jump straight to the short-circuit conjugate path.
    """
    if state.get("paradigm") == "conjugate_update":
        return "conjugate"
    if state.get("skeleton_instantiated") and not state.get("current_node_id"):
        return "advance"
    return "decompose"


def route_after_advance(state: DecompositionState) -> str:
    """Conditional edge after advance_node."""
    if state.get("done", False):
        return "end"
    pending = state.get("pending_node_ids", [])
    if not pending:
        return "end"
    return "decompose"


# ---------------------------------------------------------------------------
# Node: advance_conjugate_node  (short-circuit path)
# ---------------------------------------------------------------------------


async def advance_conjugate_node(
    state: DecompositionState, config: RunnableConfig
) -> dict[str, Any]:
    """Emit a fully ATOMIC 3-node CDG for a conjugate Bayesian update.

    This node is reached **only** when ``_detect_conjugate_pair`` found a
    known conjugate pair during ``select_strategy``.  It bypasses the
    iterative decompose → critique → advance loop entirely.

    The resulting CDG contains:
        1. Data_Ingestion (Sufficient Statistics)
        2. Hyperparameter_Update
        3. Distribution_Construction

    All three nodes are marked ``NodeStatus.ATOMIC`` so they hand off
    directly to the Hunter (Round 2).
    """
    goal = state["goal"]
    root_nodes = state["nodes"]
    root_id = state["current_node_id"]

    # Re-detect the conjugate pair from the goal (deterministic, pure).
    conjugate_spec = _detect_conjugate_pair(goal)
    if conjugate_spec is None:
        # Defensive: should never happen since route_after_strategy
        # only sends us here when paradigm == conjugate_update.
        return {
            "error": "Conjugate spec not found on re-scan",
            "done": True,
            "history": [
                {"step": "advance_conjugate_node", "error": "no conjugate pair"}
            ],
        }

    pair_name = conjugate_spec["pair_name"]
    log_event("architect", "conjugate", "CONJUGATE_FAST_PATH_FIRED", payload={"pair": conjugate_spec["pair_name"]})

    # -- Node 1: Data Ingestion (Sufficient Statistics) -------------------
    ingest_id = f"conj_ingest_{uuid.uuid4().hex[:8]}"
    ingest_node = AlgorithmicNode(
        node_id=ingest_id,
        parent_id=root_id,
        name="Data Ingestion (Sufficient Statistics)",
        description=(
            f"[{pair_name}] {conjugate_spec['sufficient_stat']}. "
            "Stateless reduction of raw observations to sufficient statistics."
        ),
        concept_type=ConceptType.CONJUGATE_UPDATE,
        inputs=[IOSpec(name="data", type_desc="ndarray", constraints="observed data")],
        outputs=[
            IOSpec(
                name="sufficient_stats",
                type_desc="tuple",
                constraints="sufficient statistics for conjugate update",
            )
        ],
        status=NodeStatus.ATOMIC,
        depth=1,
        type_signature=conjugate_spec["type_sig_ingest"],
        matched_primitive=f"conjugate_sufficient_stats_{pair_name}",
    )

    # -- Node 2: Hyperparameter Update ------------------------------------
    update_id = f"conj_update_{uuid.uuid4().hex[:8]}"
    update_node = AlgorithmicNode(
        node_id=update_id,
        parent_id=root_id,
        name="Hyperparameter Update",
        description=(
            f"[{pair_name}] {conjugate_spec['hyperparameter_update']}. "
            "Pure function: prior hyperparameters + sufficient stats → posterior "
            "hyperparameters (State Decoupling enforced — no hidden state)."
        ),
        concept_type=ConceptType.CONJUGATE_UPDATE,
        inputs=[
            IOSpec(
                name="prior_hyperparams",
                type_desc="tuple",
                constraints="prior distribution hyperparameters",
            ),
            IOSpec(
                name="sufficient_stats",
                type_desc="tuple",
                constraints="sufficient statistics from data ingestion",
            ),
        ],
        outputs=[
            IOSpec(
                name="posterior_hyperparams",
                type_desc="tuple",
                constraints="updated posterior hyperparameters",
            )
        ],
        status=NodeStatus.ATOMIC,
        depth=1,
        type_signature=conjugate_spec["type_sig_update"],
        matched_primitive=f"conjugate_hyperparameter_update_{pair_name}",
    )

    # -- Node 3: Distribution Construction --------------------------------
    construct_id = f"conj_dist_{uuid.uuid4().hex[:8]}"
    construct_node = AlgorithmicNode(
        node_id=construct_id,
        parent_id=root_id,
        name="Distribution Construction",
        description=(
            f"[{pair_name}] Construct {conjugate_spec['result_distribution']} "
            "from posterior hyperparameters. Pure constructor — no side effects."
        ),
        concept_type=ConceptType.CONJUGATE_UPDATE,
        inputs=[
            IOSpec(
                name="posterior_hyperparams",
                type_desc="tuple",
                constraints="posterior hyperparameters",
            )
        ],
        outputs=[
            IOSpec(
                name="posterior_distribution",
                type_desc="Distribution",
                constraints=conjugate_spec["result_distribution"],
            )
        ],
        status=NodeStatus.ATOMIC,
        depth=1,
        type_signature=conjugate_spec["type_sig_construct"],
        matched_primitive=f"conjugate_construct_{pair_name}",
    )

    # -- Edges: linear chain ingest → update → construct ------------------
    edges = [
        DependencyEdge(
            source_id=ingest_id,
            target_id=update_id,
            output_name="sufficient_stats",
            input_name="sufficient_stats",
            source_type="tuple",
            target_type="tuple",
        ),
        DependencyEdge(
            source_id=update_id,
            target_id=construct_id,
            output_name="posterior_hyperparams",
            input_name="posterior_hyperparams",
            source_type="tuple",
            target_type="tuple",
        ),
    ]

    # Update root to list children
    root = _find_node(root_nodes, root_id)
    updated_root: list[AlgorithmicNode] = []
    if root:
        updated_root = [
            root.model_copy(
                update={
                    "children": [ingest_id, update_id, construct_id],
                    "status": NodeStatus.DECOMPOSED,
                }
            )
        ]

    return {
        "nodes": updated_root + [ingest_node, update_node, construct_node],
        "edges": edges,
        "pending_node_ids": [],
        "current_node_id": "",
        "done": True,
        "history": [
            {
                "step": "advance_conjugate_node",
                "pair_name": pair_name,
                "nodes_emitted": 3,
                "all_atomic": True,
            }
        ],
    }
