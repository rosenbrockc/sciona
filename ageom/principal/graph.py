"""Principal state machine: Forward -> Evaluate -> Backward -> Update loop.

Orchestrates NAS-style optimisation over the existing synthesis pipeline,
using the Architect's checkpointer for O(1) time-travel coordinate descent.
"""

from __future__ import annotations

import logging
import inspect
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from ageom.architect.catalog import PrimitiveCatalog
from ageom.architect.graph import DecompositionAgent
from ageom.architect.handoff import CDGExport
from ageom.architect.models import NodeStatus
from ageom.principal.atom_ledger import AtomLedger, compute_slot_signature
from ageom.principal.backprop import CreditAssigner
from ageom.principal.evaluator import ExecutionSandbox
from ageom.principal.hpo import OptunaManager, TrialPrunedEarly
from ageom.principal.models import (
    BenchmarkResult,
    NodeGradient,
    OptimizationMetric,
)
from ageom.principal.reference_attribution import (
    compute_reference_loss_gradients,
    is_reference_loss_objective,
)
from ageom.principal.structure_summary import summarize_trial_structure
from ageom.principal.structure_objective import benchmark_from_ghost_report
from ageom.principal.variant_mutation import maybe_apply_bottleneck_variant
from ageom.synthesizer.ghost_sim import GhostSimReport, run_ghost_simulation
from ageom.synthesizer.models import ExportBundle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class PrincipalState:
    """Mutable state threaded through the Principal graph."""

    goal: str = ""
    metric: OptimizationMetric = OptimizationMetric.LATENCY
    dataset_path: str = ""
    max_trials: int = 50
    current_trial: int = 0
    best_loss: float = float("inf")

    # Pipeline artefacts
    thread_id: str = ""
    cdg: CDGExport | None = None
    export_bundle: ExportBundle | None = None
    ghost_report: GhostSimReport = field(default_factory=GhostSimReport)
    benchmark: BenchmarkResult | None = None
    match_results: list[Any] = field(default_factory=list)

    # Gradient
    top_gradient: NodeGradient | None = None
    bottleneck_node_id: str = ""
    bottleneck_reason: str = ""

    # Hyperparameter assignments (node_id -> {param_name: value})
    node_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    best_node_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Bookkeeping
    done: bool = False
    error: str = ""
    trial_history: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


async def seed_population(state: PrincipalState, config: RunnableConfig) -> dict:
    """Use OptunaManager to suggest initial Architect paradigms and run decomposition."""
    deps: PrincipalDeps = config["configurable"]["deps"]

    state.current_trial += 1
    thread_id = uuid.uuid4().hex
    state.thread_id = thread_id

    logger.info("Trial %d: decomposing '%s'", state.current_trial, state.goal)

    cdg = await deps.architect.decompose(state.goal, thread_id=thread_id)
    state.cdg = cdg

    return {"cdg": cdg, "thread_id": thread_id, "current_trial": state.current_trial}


async def execute_forward(state: PrincipalState, config: RunnableConfig) -> dict:
    """Run the Orchestrator / Synthesizer pipeline and produce an ExportBundle."""
    deps: PrincipalDeps = config["configurable"]["deps"]

    # Increment trial counter for time-travel re-entries (seed handles the
    # first trial; subsequent trials arrive here via time_travel → forward).
    trials_so_far = len(state.trial_history)
    if trials_so_far >= state.current_trial:
        state.current_trial = trials_so_far + 1

    if state.cdg is None:
        return {"error": "No CDG available", "done": True}

    # Ghost simulation for early pruning and precision gradients
    match_results = deps.match_results_fn(state.cdg) if deps.match_results_fn else []
    if inspect.isawaitable(match_results):
        match_results = await match_results
    state.match_results = list(match_results)
    ghost_report = run_ghost_simulation(state.cdg, match_results)
    state.ghost_report = ghost_report

    # Early prune via Optuna if ghost sim shows structural failure
    try:
        OptunaManager.check_early_prune(ghost_report)
    except TrialPrunedEarly as exc:
        logger.warning("Trial %d pruned early: %s", state.current_trial, exc)
        return {"error": str(exc), "done": False}

    # The actual synthesis is delegated to a caller-provided function
    # that returns an ExportBundle (keeps the graph deterministic).
    if deps.synthesize_fn is not None:
        bundle = await deps.synthesize_fn(state.cdg, match_results)
        if state.node_params:
            bundle.parameter_assignments = dict(state.node_params)
        state.export_bundle = bundle

    return {
        "ghost_report": ghost_report,
        "export_bundle": state.export_bundle,
        "current_trial": state.current_trial,
        "match_results": state.match_results,
    }


async def evaluate_run(state: PrincipalState, config: RunnableConfig) -> dict:
    """Execute the instrumented artifact and gather telemetry."""
    deps: PrincipalDeps = config["configurable"]["deps"]

    if state.export_bundle is None:
        return {"error": "No export bundle to evaluate", "done": True}

    if state.metric == OptimizationMetric.STRUCTURE:
        benchmark = benchmark_from_ghost_report(state.ghost_report)
    else:
        sandbox = deps.sandbox
        if state.dataset_path.endswith((".yml", ".yaml")):
            benchmark = await sandbox.evaluate_adapter(
                state.export_bundle,
                state.dataset_path,
                state.metric,
                varset=deps.dataset_varset,
                evaluation_spec=deps.evaluation_spec,
            )
        else:
            benchmark = await sandbox.evaluate(
                state.export_bundle,
                state.dataset_path,
                state.metric,
                evaluation_spec=deps.evaluation_spec,
            )
    state.benchmark = benchmark

    # Track best
    if benchmark.global_loss < state.best_loss:
        state.best_loss = benchmark.global_loss
        logger.info(
            "Trial %d: new best loss = %.6f",
            state.current_trial,
            benchmark.global_loss,
        )

    structure = (
        summarize_trial_structure(
            state.cdg,
            ghost_report=state.ghost_report,
            match_results=state.match_results,
        )
        if state.cdg is not None
        else {}
    )
    previous_structure = (
        state.trial_history[-1].get("structure", {}) if state.trial_history else {}
    )
    if structure:
        structure["topology_changed"] = (
            bool(previous_structure)
            and structure.get("topo_hash") != previous_structure.get("topo_hash")
        )
        structure["primitive_assignment_changed"] = (
            bool(previous_structure)
            and structure.get("primitive_signature")
            != previous_structure.get("primitive_signature")
        )
        structure["node_count_delta"] = int(structure.get("node_count", 0)) - int(
            previous_structure.get("node_count", 0) or 0
        )
        structure["edge_count_delta"] = int(structure.get("edge_count", 0)) - int(
            previous_structure.get("edge_count", 0) or 0
        )

    state.trial_history.append(
        {
            "trial": state.current_trial,
            "loss": benchmark.global_loss,
            "thread_id": state.thread_id,
            "structure": structure,
            "parameter_assignments": dict(state.node_params),
        }
    )

    # Track best params
    if benchmark.global_loss <= state.best_loss:
        state.best_node_params = dict(state.node_params)

    return {
        "benchmark": benchmark,
        "best_loss": state.best_loss,
        "trial_history": state.trial_history,
        "current_trial": state.current_trial,
    }


def _record_gradients_to_ledger(
    ledger: AtomLedger,
    cdg: CDGExport,
    gradients: list[NodeGradient],
    trial: int,
) -> None:
    """Record all gradient observations to the atom ledger."""
    node_map = {n.node_id: n for n in cdg.nodes}
    for grad in gradients:
        node = node_map.get(grad.node_id)
        if node is None or node.status != NodeStatus.ATOMIC:
            continue
        if not node.matched_primitive:
            continue
        parent = node_map.get(node.parent_id) if node.parent_id else None
        slot = compute_slot_signature(node, parent)
        ledger.record(slot, node.matched_primitive, grad.gradient_score, trial)


async def compute_gradients(state: PrincipalState, config: RunnableConfig) -> dict:
    """Call CreditAssigner to find the top bottleneck node."""
    deps: PrincipalDeps = config["configurable"]["deps"]
    if state.cdg is None or state.benchmark is None:
        return {"done": True, "error": "Missing CDG or benchmark"}

    if (
        state.export_bundle is not None
        and is_reference_loss_objective(state.metric, deps.evaluation_spec)[0] is not None
    ):
        gradients = await compute_reference_loss_gradients(
            state.cdg,
            state.export_bundle,
            state.dataset_path,
            deps.evaluation_spec,
            dataset_varset=deps.dataset_varset,
        )
        if gradients:
            if deps.atom_ledger is not None and state.cdg is not None:
                _record_gradients_to_ledger(
                    deps.atom_ledger, state.cdg, gradients, state.current_trial
                )
            top = gradients[0]
            state.top_gradient = top
            state.bottleneck_node_id = top.node_id
            state.bottleneck_reason = top.bottleneck_reason
            return {
                "top_gradient": top,
                "bottleneck_node_id": top.node_id,
                "bottleneck_reason": top.bottleneck_reason,
            }

    assigner = CreditAssigner()
    gradients = assigner.compute_gradients(
        state.cdg,
        state.benchmark,
        state.ghost_report,
        state.metric,
    )

    if not gradients:
        logger.info("No gradients computed; ending optimisation loop.")
        return {"done": True}

    if deps.atom_ledger is not None and state.cdg is not None:
        _record_gradients_to_ledger(
            deps.atom_ledger, state.cdg, gradients, state.current_trial
        )

    top = gradients[0]
    state.top_gradient = top
    state.bottleneck_node_id = top.node_id
    state.bottleneck_reason = top.bottleneck_reason

    logger.info(
        "Trial %d bottleneck: %s (%.1f%%)",
        state.current_trial,
        top.node_id,
        top.gradient_score,
    )
    return {
        "top_gradient": top,
        "bottleneck_node_id": top.node_id,
        "bottleneck_reason": top.bottleneck_reason,
    }


async def time_travel_update(state: PrincipalState, config: RunnableConfig) -> dict:
    """Fork the Architect graph at the bottleneck and re-decompose with a constraint."""
    deps: PrincipalDeps = config["configurable"]["deps"]

    if not state.bottleneck_node_id or state.cdg is None:
        return {"done": True}

    architect = deps.architect
    bottleneck_name = next(
        (node.name for node in state.cdg.nodes if node.node_id == state.bottleneck_node_id),
        None,
    )
    mutation = maybe_apply_bottleneck_variant(
        state.cdg,
        bottleneck_name=bottleneck_name,
        atom_ledger=deps.atom_ledger,
        catalog=deps.catalog,
    )
    if mutation.applied:
        state.cdg = mutation.cdg
        logger.info(
            "Applied variant family '%s' in place for '%s' -> %s",
            mutation.family or "unknown",
            bottleneck_name or state.bottleneck_node_id,
            mutation.variant_name or "variant",
        )
        return {"cdg": mutation.cdg}
    if mutation.family is not None and not mutation.allow_redecompose:
        logger.info(
            "Variant family '%s' has no further safe mutations for '%s'; ending loop.",
            mutation.family,
            bottleneck_name or state.bottleneck_node_id,
        )
        return {"done": True}

    # Find the checkpoint just before the bottleneck node was decomposed
    history = await architect.get_state_history(state.thread_id)
    target_cp: str | None = None
    for entry in history:
        vals = entry.get("values", {})
        node_ids = {n.node_id for n in vals.get("nodes", [])}
        if state.bottleneck_node_id not in node_ids:
            # This checkpoint is just before the bottleneck was created
            target_cp = entry.get("checkpoint_id")
            break

    if target_cp is None:
        # Fallback: use the earliest checkpoint
        if history:
            target_cp = history[-1].get("checkpoint_id")

    if target_cp is None:
        logger.warning("No checkpoint found for time-travel; ending loop.")
        return {"done": True}

    # Fork a new thread from that checkpoint
    new_thread_id = await architect.fork(state.thread_id, target_cp)
    state.thread_id = new_thread_id

    constraint = (
        f"The previous decomposition caused a bottleneck: "
        f"{state.bottleneck_reason}. Re-decompose more efficiently."
    )

    # Append bandit ranking hints for the bottleneck slot
    if deps.atom_ledger is not None and deps.catalog is not None and state.cdg is not None:
        bottleneck_node = next(
            (n for n in state.cdg.nodes if n.node_id == state.bottleneck_node_id),
            None,
        )
        if bottleneck_node is not None and bottleneck_node.matched_primitive:
            node_map = {n.node_id: n for n in state.cdg.nodes}
            parent = node_map.get(bottleneck_node.parent_id) if bottleneck_node.parent_id else None
            slot = compute_slot_signature(bottleneck_node, parent)
            same_category = [
                p.name for p in deps.catalog.search_by_category(bottleneck_node.concept_type)
            ]
            if len(same_category) > 1:
                ranked = deps.atom_ledger.rank_candidates(slot, same_category)
                top_3 = [
                    (name, f"{score:.2f}")
                    for name, score in ranked[:3]
                    if score != float("inf")
                ]
                if top_3:
                    constraint += (
                        f"\nATOM RANKINGS for '{bottleneck_name}': "
                        f"prefer {top_3}"
                    )

    constrained_goal = f"{state.goal}\n\nCONSTRAINT: {constraint}"
    cdg = await architect.decompose(constrained_goal, thread_id=new_thread_id)
    mutation = maybe_apply_bottleneck_variant(
        cdg,
        bottleneck_name=bottleneck_name,
        atom_ledger=deps.atom_ledger,
        catalog=deps.catalog,
    )
    state.cdg = mutation.cdg

    logger.info(
        "Time-travel: forked at checkpoint %s -> thread %s",
        target_cp,
        new_thread_id,
    )

    return {"cdg": mutation.cdg, "thread_id": new_thread_id}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_after_gradients(state: PrincipalState) -> str:
    """Decide whether to continue optimising or stop."""
    if state.done:
        return "end"
    if len(state.trial_history) >= state.max_trials:
        return "end"
    if state.error and "pruned" not in state.error.lower():
        return "end"
    return "time_travel"


def route_after_update(state: PrincipalState) -> str:
    """After time-travel update, loop back or stop."""
    if state.done:
        return "end"
    if len(state.trial_history) >= state.max_trials:
        return "end"
    return "forward"


def route_after_forward(state: PrincipalState) -> str:
    """After forward pass, evaluate or loop back (if pruned early)."""
    if state.done:
        return "end"
    if state.error:
        # Pruned early — skip evaluation, go straight to time-travel
        return "time_travel"
    return "evaluate"


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


@dataclass
class PrincipalDeps:
    """Injected dependencies for the Principal graph."""

    architect: DecompositionAgent
    sandbox: ExecutionSandbox
    match_results_fn: Any = None  # Callable[[CDGExport], list[MatchResult]]
    synthesize_fn: Any = None  # Callable[[CDGExport, list], Awaitable[ExportBundle]]
    evaluation_spec: Any = None
    dataset_varset: dict[str, str] | None = None
    atom_ledger: AtomLedger | None = None
    catalog: PrimitiveCatalog | None = None


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_principal_graph() -> StateGraph:
    """Construct the Principal's optimisation StateGraph."""
    graph = StateGraph(PrincipalState)

    graph.add_node("seed", seed_population)
    graph.add_node("forward", execute_forward)
    graph.add_node("evaluate", evaluate_run)
    graph.add_node("gradients", compute_gradients)
    graph.add_node("time_travel", time_travel_update)

    graph.set_entry_point("seed")
    graph.add_edge("seed", "forward")
    graph.add_conditional_edges(
        "forward",
        route_after_forward,
        {"evaluate": "evaluate", "time_travel": "time_travel", "end": END},
    )
    graph.add_edge("evaluate", "gradients")
    graph.add_conditional_edges(
        "gradients",
        route_after_gradients,
        {"time_travel": "time_travel", "end": END},
    )
    graph.add_conditional_edges(
        "time_travel",
        route_after_update,
        {"forward": "forward", "end": END},
    )

    return graph
