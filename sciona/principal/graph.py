"""Principal state machine: Forward -> Evaluate -> Backward -> Update loop.

Orchestrates NAS-style optimisation over the existing synthesis pipeline,
using the Architect's checkpointer for O(1) time-travel coordinate descent.
"""

from __future__ import annotations

import logging
import inspect
import uuid
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from sciona.architect.catalog import PrimitiveCatalog
from sciona.architect.handoff import CDGExport
from sciona.architect.models import NodeStatus
from sciona.principal.atom_ledger import compute_slot_signature
from sciona.principal.backprop import CreditAssigner
from sciona.principal.evaluation_helpers import evaluate_bundle_for_metric
from sciona.principal.graph_routing import (
    route_after_forward,
    route_after_gradients,
    route_after_proposal,
    route_after_update,
)
from sciona.principal.graph_types import PrincipalDeps, PrincipalState
from sciona.principal.graph_utils import _param_signature, _structure_has_tunables
from sciona.principal.hpo import OptunaManager, SuggestedParams, TrialPrunedEarly
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules import default_rule_sets
from sciona.principal.models import (
    BenchmarkResult,
    NodeGradient,
    OptimizationMetric,
)
from sciona.principal.reference_attribution import (
    compute_reference_loss_gradients,
    is_reference_loss_objective,
)
from sciona.principal.proposal_helpers import (
    build_expansion_context,
    build_redecomposition_candidate,
    evaluate_proposal_candidate,
    summarize_expansion_context,
)
from sciona.principal.structure_summary import summarize_trial_structure
from sciona.principal.structure_objective import benchmark_from_ghost_report
from sciona.principal.variant_mutation import maybe_apply_bottleneck_variant
from sciona.synthesizer.ghost_sim import GhostSimReport, run_ghost_simulation
from sciona.synthesizer.models import ExportBundle

logger = logging.getLogger(__name__)


async def seed_population(state: PrincipalState, config: RunnableConfig) -> dict:
    """Use OptunaManager to suggest initial Architect paradigms and run decomposition."""
    deps: PrincipalDeps = config["configurable"]["deps"]

    state.current_trial += 1
    thread_id = uuid.uuid4().hex
    state.thread_id = thread_id

    logger.info("Trial %d: decomposing '%s'", state.current_trial, state.goal)

    cdg = await deps.architect.decompose(state.goal, thread_id=thread_id)
    state.cdg = cdg
    has_tunables = _structure_has_tunables(cdg, deps.catalog)
    state.pending_param_search = has_tunables and deps.param_trials_per_structure > 0
    state.param_trials_remaining = (
        deps.param_trials_per_structure if has_tunables else 0
    )

    return {
        "cdg": cdg,
        "thread_id": thread_id,
        "current_trial": state.current_trial,
        "pending_param_search": state.pending_param_search,
        "param_trials_remaining": state.param_trials_remaining,
    }

async def suggest_params(state: PrincipalState, config: RunnableConfig) -> dict:
    """Sample node-level hyperparameters for the current CDG signature."""
    deps: PrincipalDeps = config["configurable"]["deps"]
    if state.cdg is None or deps.catalog is None or deps.hpo_manager is None:
        state.node_params = {}
        state.param_signature = ""
        state.hpo_trial_number = None
        return {"node_params": {}, "param_signature": "", "hpo_trial_number": None}

    signature = _param_signature(state.cdg)
    suggested = deps.hpo_manager.suggest_node_params(
        signature=signature,
        cdg=state.cdg,
        catalog=deps.catalog,
    )
    state.node_params = suggested.assignments
    state.param_signature = suggested.signature
    state.hpo_trial_number = suggested.trial_number
    return {
        "node_params": suggested.assignments,
        "param_signature": suggested.signature,
        "hpo_trial_number": suggested.trial_number,
    }


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

    if (
        state.reuse_cached_evaluation
        and state.export_bundle is not None
        and state.benchmark is not None
    ):
        logger.info(
            "Trial %d reusing cached proposal evaluation for '%s'",
            state.current_trial,
            state.selected_proposal or "proposal",
        )
        return {
            "ghost_report": state.ghost_report,
            "export_bundle": state.export_bundle,
            "current_trial": state.current_trial,
            "match_results": state.match_results,
            "error": "",
            "selected_proposal": "",
            "reuse_cached_evaluation": True,
        }

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
        if deps.hpo_manager is not None:
            deps.hpo_manager.prune_trial(
                signature=state.param_signature,
                trial_number=state.hpo_trial_number,
            )
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
        "error": "",
        "selected_proposal": "",
        "reuse_cached_evaluation": False,
    }


async def evaluate_run(state: PrincipalState, config: RunnableConfig) -> dict:
    """Execute the instrumented artifact and gather telemetry."""
    deps: PrincipalDeps = config["configurable"]["deps"]

    if state.export_bundle is None:
        return {"error": "No export bundle to evaluate", "done": True}

    reused_cached_evaluation = bool(state.reuse_cached_evaluation)
    if reused_cached_evaluation and state.benchmark is not None:
        benchmark = state.benchmark
    elif state.metric == OptimizationMetric.STRUCTURE:
        benchmark = benchmark_from_ghost_report(state.ghost_report)
    else:
        benchmark = await evaluate_bundle_for_metric(
            deps.sandbox,
            state.export_bundle,
            state.dataset_path,
            state.metric,
            dataset_varset=deps.dataset_varset,
            evaluation_spec=deps.evaluation_spec,
        )
    state.benchmark = benchmark
    state.reuse_cached_evaluation = False

    if (
        deps.hpo_manager is not None
        and state.param_signature
        and state.hpo_trial_number is not None
    ):
        deps.hpo_manager.complete_trial(
            signature=state.param_signature,
            trial_number=state.hpo_trial_number,
            loss=benchmark.global_loss,
        )

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
            catalog=deps.catalog,
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
            "reused_cached_evaluation": reused_cached_evaluation,
            "expansion": {
                "applied": False,
                "rules_applied": [],
                "diagnostic_count": 0,
                "context_summary": {},
            },
            "rollback": {
                "applied": False,
                "reason": "",
                "restored_trial": None,
            },
        }
    )

    # Track best params
    if benchmark.global_loss <= state.best_loss:
        state.best_node_params = dict(state.node_params)

    if state.pending_param_search and state.param_trials_remaining > 0:
        state.param_trials_remaining -= 1
        if state.param_trials_remaining <= 0:
            state.pending_param_search = False

    return {
        "benchmark": benchmark,
        "best_loss": state.best_loss,
        "best_node_params": state.best_node_params,
        "trial_history": state.trial_history,
        "reuse_cached_evaluation": state.reuse_cached_evaluation,
        "current_trial": state.current_trial,
        "pending_param_search": state.pending_param_search,
        "param_trials_remaining": state.param_trials_remaining,
    }


async def select_proposal(state: PrincipalState, config: RunnableConfig) -> dict:
    """Compare sibling expansion and mutation proposals from the same baseline."""
    deps: PrincipalDeps = config["configurable"]["deps"]
    if (
        state.cdg is None
        or state.benchmark is None
        or not state.bottleneck_node_id
    ):
        state.selected_proposal = ""
        return {"selected_proposal": ""}

    baseline_cdg = state.cdg.model_copy(deep=True)
    baseline_loss = float(state.benchmark.global_loss)
    bottleneck_name = next(
        (node.name for node in baseline_cdg.nodes if node.node_id == state.bottleneck_node_id),
        None,
    )
    proposal_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    engine = deps.expansion_engine or ExpansionEngine(default_rule_sets())
    context = build_expansion_context(state)
    expansion = engine.expand(baseline_cdg, context)
    if expansion.expanded:
        loss, bundle, benchmark, match_results, ghost_report = await evaluate_proposal_candidate(
            state,
            deps,
            expansion.cdg,
        )
        candidates.append(
            {
                "label": "expansion",
                "loss": loss,
                "cdg": expansion.cdg,
                "bundle": bundle,
                "benchmark": benchmark,
                "match_results": match_results,
                "ghost_report": ghost_report,
                "rules_applied": list(expansion.applied_rules),
            }
        )
        proposal_rows.append(
            {
                "label": "expansion",
                "loss": loss,
                "improves_baseline": loss < baseline_loss,
                "rules_applied": list(expansion.applied_rules),
            }
        )

    mutation = maybe_apply_bottleneck_variant(
        baseline_cdg,
        bottleneck_name=bottleneck_name,
        atom_ledger=deps.atom_ledger,
        catalog=deps.catalog,
    )
    if mutation.applied:
        loss, bundle, benchmark, match_results, ghost_report = await evaluate_proposal_candidate(
            state,
            deps,
            mutation.cdg,
        )
        candidates.append(
            {
                "label": "local_mutation",
                "loss": loss,
                "cdg": mutation.cdg,
                "bundle": bundle,
                "benchmark": benchmark,
                "match_results": match_results,
                "ghost_report": ghost_report,
                "variant_name": mutation.variant_name or "",
                "family": mutation.family or "",
            }
        )
        proposal_rows.append(
            {
                "label": "local_mutation",
                "loss": loss,
                "improves_baseline": loss < baseline_loss,
                "variant_name": mutation.variant_name or "",
                "family": mutation.family or "",
            }
        )

    selected = None
    for candidate in sorted(candidates, key=lambda row: (row["loss"], row["label"])):
        if candidate["loss"] < baseline_loss:
            selected = candidate
            break

    if selected is None:
        redecomposition = await build_redecomposition_candidate(
            state,
            deps,
            bottleneck_name=bottleneck_name,
        )
        if redecomposition is not None:
            redecompose_cdg, redecompose_thread_id = redecomposition
            loss, bundle, benchmark, match_results, ghost_report = await evaluate_proposal_candidate(
                state,
                deps,
                redecompose_cdg,
            )
            candidates.append(
                {
                    "label": "redecompose",
                    "loss": loss,
                    "cdg": redecompose_cdg,
                    "bundle": bundle,
                    "benchmark": benchmark,
                    "match_results": match_results,
                    "ghost_report": ghost_report,
                    "thread_id": redecompose_thread_id,
                }
            )
            proposal_rows.append(
                {
                    "label": "redecompose",
                    "loss": loss,
                    "improves_baseline": loss < baseline_loss,
                }
            )
            if loss < baseline_loss:
                selected = candidates[-1]

    if state.trial_history:
        latest = dict(state.trial_history[-1])
        latest["proposal_selection"] = {
            "baseline_loss": baseline_loss,
            "candidates": proposal_rows,
            "selected": str(selected["label"]) if selected is not None else "",
        }
        latest["expansion"] = {
            "applied": selected is not None and selected["label"] == "expansion",
            "rules_applied": (
                list(selected.get("rules_applied", []))
                if selected is not None and selected["label"] == "expansion"
                else []
            ),
            "diagnostic_count": len(expansion.diagnostics),
            "diagnostic_rule_names": sorted(
                {diag.rule_name for diag in expansion.diagnostics}
            ),
            "context_summary": summarize_expansion_context(context),
        }
        state.trial_history[-1] = latest

    if selected is None:
        state.selected_proposal = ""
        state.reuse_cached_evaluation = False
        return {
            "selected_proposal": "",
            "trial_history": state.trial_history,
            "expansion_applied": False,
            "expansion_rules_applied": [],
            "reuse_cached_evaluation": False,
        }

    state.cdg = selected["cdg"]
    state.export_bundle = selected.get("bundle")
    state.benchmark = selected.get("benchmark")
    state.match_results = list(selected.get("match_results", []))
    state.ghost_report = selected["ghost_report"]
    state.expansion_applied = selected["label"] == "expansion"
    state.expansion_rules_applied = list(selected.get("rules_applied", []))
    state.thread_id = str(selected.get("thread_id", state.thread_id) or state.thread_id)
    state.node_params = {}
    state.param_signature = ""
    state.hpo_trial_number = None
    state.pending_param_search = _structure_has_tunables(state.cdg, deps.catalog) and (
        deps.param_trials_per_structure > 0
    )
    state.param_trials_remaining = (
        deps.param_trials_per_structure if state.pending_param_search else 0
    )
    state.selected_proposal = str(selected["label"])
    state.reuse_cached_evaluation = not state.pending_param_search
    logger.info(
        "Trial %d selected proposal '%s' (loss=%.6f vs baseline %.6f)",
        state.current_trial,
        state.selected_proposal,
        float(selected["loss"]),
        baseline_loss,
    )
    return {
        "cdg": state.cdg,
        "export_bundle": state.export_bundle,
        "benchmark": state.benchmark,
        "match_results": state.match_results,
        "ghost_report": state.ghost_report,
        "expansion_applied": state.expansion_applied,
        "expansion_rules_applied": state.expansion_rules_applied,
        "pending_param_search": state.pending_param_search,
        "param_trials_remaining": state.param_trials_remaining,
        "selected_proposal": state.selected_proposal,
        "thread_id": state.thread_id,
        "reuse_cached_evaluation": state.reuse_cached_evaluation,
        "trial_history": state.trial_history,
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

    bottleneck_name = next(
        (node.name for node in state.cdg.nodes if node.node_id == state.bottleneck_node_id),
        None,
    )
    candidate = await build_redecomposition_candidate(
        state,
        deps,
        bottleneck_name=bottleneck_name,
    )
    if candidate is None:
        logger.warning("No checkpoint found for time-travel; ending loop.")
        return {"done": True}

    cdg, new_thread_id = candidate
    state.thread_id = new_thread_id
    state.cdg = cdg
    has_tunables = _structure_has_tunables(cdg, deps.catalog)
    state.pending_param_search = has_tunables and deps.param_trials_per_structure > 0
    state.param_trials_remaining = deps.param_trials_per_structure if has_tunables else 0

    logger.info(
        "Time-travel: forked new thread %s for re-decomposition",
        new_thread_id,
    )

    return {
        "cdg": cdg,
        "thread_id": new_thread_id,
        "pending_param_search": state.pending_param_search,
        "param_trials_remaining": state.param_trials_remaining,
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_principal_graph() -> StateGraph:
    """Construct the Principal's optimisation StateGraph."""
    graph = StateGraph(PrincipalState)

    graph.add_node("seed", seed_population)
    graph.add_node("suggest_params", suggest_params)
    graph.add_node("forward", execute_forward)
    graph.add_node("evaluate", evaluate_run)
    graph.add_node("gradients", compute_gradients)
    graph.add_node("select_proposal", select_proposal)
    graph.add_node("time_travel", time_travel_update)

    graph.set_entry_point("seed")
    graph.add_edge("seed", "suggest_params")
    graph.add_edge("suggest_params", "forward")
    graph.add_conditional_edges(
        "forward",
        route_after_forward,
        {"evaluate": "evaluate", "time_travel": "time_travel", "end": END},
    )
    graph.add_edge("evaluate", "gradients")
    graph.add_conditional_edges(
        "gradients",
        route_after_gradients,
        {"suggest_params": "suggest_params", "select_proposal": "select_proposal", "end": END},
    )
    graph.add_conditional_edges(
        "select_proposal",
        route_after_proposal,
        {"suggest_params": "suggest_params", "time_travel": "time_travel", "end": END},
    )
    graph.add_conditional_edges(
        "time_travel",
        route_after_update,
        {"suggest_params": "suggest_params", "end": END},
    )

    return graph
