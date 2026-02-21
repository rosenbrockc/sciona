"""Principal state machine: Forward -> Evaluate -> Backward -> Update loop.

Orchestrates NAS-style optimisation over the existing synthesis pipeline,
using the Architect's checkpointer for O(1) time-travel coordinate descent.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import END, StateGraph

from ageom.architect.graph import DecompositionAgent
from ageom.architect.handoff import CDGExport
from ageom.principal.backprop import CreditAssigner
from ageom.principal.evaluator import ExecutionSandbox
from ageom.principal.hpo import OptunaManager, TrialPrunedEarly
from ageom.principal.models import (
    BenchmarkResult,
    NodeGradient,
    OptimizationMetric,
)
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

    # Gradient
    top_gradient: NodeGradient | None = None
    bottleneck_node_id: str = ""
    bottleneck_reason: str = ""

    # Bookkeeping
    done: bool = False
    error: str = ""
    trial_history: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


async def seed_population(state: PrincipalState, config: dict) -> dict:
    """Use OptunaManager to suggest initial Architect paradigms and run decomposition."""
    deps: PrincipalDeps = config["configurable"]["deps"]

    state.current_trial += 1
    thread_id = uuid.uuid4().hex
    state.thread_id = thread_id

    logger.info("Trial %d: decomposing '%s'", state.current_trial, state.goal)

    cdg = await deps.architect.decompose(state.goal, thread_id=thread_id)
    state.cdg = cdg

    return {"cdg": cdg, "thread_id": thread_id, "current_trial": state.current_trial}


async def execute_forward(state: PrincipalState, config: dict) -> dict:
    """Run the Orchestrator / Synthesizer pipeline and produce an ExportBundle."""
    deps: PrincipalDeps = config["configurable"]["deps"]

    if state.cdg is None:
        return {"error": "No CDG available", "done": True}

    # Ghost simulation for early pruning and precision gradients
    match_results = deps.match_results_fn(state.cdg) if deps.match_results_fn else []
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
        state.export_bundle = bundle

    return {"ghost_report": ghost_report, "export_bundle": state.export_bundle}


async def evaluate_run(state: PrincipalState, config: dict) -> dict:
    """Execute the instrumented artifact and gather telemetry."""
    deps: PrincipalDeps = config["configurable"]["deps"]

    if state.export_bundle is None:
        return {"error": "No export bundle to evaluate", "done": True}

    sandbox = deps.sandbox
    benchmark = await sandbox.evaluate(
        state.export_bundle,
        state.dataset_path,
        state.metric,
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

    state.trial_history.append(
        {
            "trial": state.current_trial,
            "loss": benchmark.global_loss,
            "thread_id": state.thread_id,
        }
    )

    return {"benchmark": benchmark}


async def compute_gradients(state: PrincipalState, config: dict) -> dict:
    """Call CreditAssigner to find the top bottleneck node."""
    if state.cdg is None or state.benchmark is None:
        return {"done": True, "error": "Missing CDG or benchmark"}

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


async def time_travel_update(state: PrincipalState, config: dict) -> dict:
    """Fork the Architect graph at the bottleneck and re-decompose with a constraint."""
    deps: PrincipalDeps = config["configurable"]["deps"]

    if not state.bottleneck_node_id or state.cdg is None:
        return {"done": True}

    architect = deps.architect

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

    # Inject the bottleneck constraint into the forked state
    constraint = (
        f"The previous decomposition caused a bottleneck: "
        f"{state.bottleneck_reason}. Re-decompose more efficiently."
    )
    fork_config = {"configurable": {"thread_id": new_thread_id}}
    current = await architect._graph.aget_state(fork_config)
    updated_values = dict(current.values)

    # Append constraint to the goal so the LLM sees it
    original_goal = updated_values.get("goal", state.goal)
    updated_values["goal"] = f"{original_goal}\n\nCONSTRAINT: {constraint}"

    # Reset decomposition progress to re-run from the fork point
    updated_values["done"] = False
    updated_values["error"] = ""

    await architect._graph.aupdate_state(fork_config, updated_values)

    # Re-run decomposition on the forked thread
    cdg = await architect.decompose(state.goal, thread_id=new_thread_id)
    state.cdg = cdg

    logger.info(
        "Time-travel: forked at checkpoint %s -> thread %s",
        target_cp,
        new_thread_id,
    )

    return {"cdg": cdg, "thread_id": new_thread_id}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_after_gradients(state: PrincipalState) -> str:
    """Decide whether to continue optimising or stop."""
    if state.done:
        return "end"
    if state.current_trial >= state.max_trials:
        return "end"
    if state.error and "pruned" not in state.error.lower():
        return "end"
    return "time_travel"


def route_after_update(state: PrincipalState) -> str:
    """After time-travel update, loop back or stop."""
    if state.done:
        return "end"
    if state.current_trial >= state.max_trials:
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
