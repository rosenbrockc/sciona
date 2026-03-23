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

from sciona.architect.catalog import PrimitiveCatalog
from sciona.architect.graph import DecompositionAgent
from sciona.architect.handoff import CDGExport
from sciona.architect.models import NodeStatus
from sciona.principal.atom_ledger import AtomLedger, compute_slot_signature
from sciona.principal.backprop import CreditAssigner
from sciona.principal.evaluator import ExecutionSandbox
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
from sciona.principal.structure_summary import summarize_trial_structure
from sciona.principal.structure_objective import benchmark_from_ghost_report
from sciona.principal.variant_mutation import maybe_apply_bottleneck_variant
from sciona.synthesizer.ghost_sim import GhostSimReport, run_ghost_simulation
from sciona.synthesizer.models import ExportBundle

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
    param_signature: str = ""
    hpo_trial_number: int | None = None
    pending_param_search: bool = False
    param_trials_remaining: int = 0
    expansion_applied: bool = False
    expansion_rules_applied: list[str] = field(default_factory=list)
    expansion_candidate_active: bool = False
    expansion_baseline_trial: int | None = None
    expansion_baseline_loss: float | None = None
    expansion_baseline_thread_id: str = ""
    expansion_baseline_cdg: CDGExport | None = None
    expansion_baseline_export_bundle: ExportBundle | None = None
    expansion_baseline_benchmark: BenchmarkResult | None = None
    expansion_baseline_node_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    selected_proposal: str = ""

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


def _param_signature(cdg: CDGExport) -> str:
    """Return the scoped study signature for the current structure + primitives."""
    summary = summarize_trial_structure(cdg)
    return f"{summary.get('topo_hash', '')}:{summary.get('primitive_signature', '')}"


def _structure_has_tunables(cdg: CDGExport, catalog: PrimitiveCatalog | None) -> bool:
    """Return whether any atomic node in *cdg* exposes approved tunables."""
    if catalog is None:
        return False
    for node in cdg.nodes:
        if node.status != NodeStatus.ATOMIC:
            continue
        primitive_name = str(node.matched_primitive or "").strip()
        if not primitive_name:
            continue
        primitive = catalog.get(primitive_name)
        if primitive is not None and primitive.tunable_params:
            return True
    return False


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

    if deps.hpo_manager is not None:
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

    rollback_payload: dict[str, Any] = {}
    if (
        state.expansion_candidate_active
        and state.expansion_baseline_loss is not None
        and benchmark.global_loss > state.expansion_baseline_loss
        and state.expansion_baseline_cdg is not None
    ):
        restored_trial = state.expansion_baseline_trial
        state.cdg = state.expansion_baseline_cdg
        state.export_bundle = state.expansion_baseline_export_bundle
        state.benchmark = state.expansion_baseline_benchmark
        state.thread_id = state.expansion_baseline_thread_id
        state.node_params = dict(state.expansion_baseline_node_params)
        state.expansion_candidate_active = False
        state.expansion_applied = False
        state.expansion_rules_applied = []
        state.done = True
        latest = dict(state.trial_history[-1])
        latest["rollback"] = {
            "applied": True,
            "reason": "expanded structure increased objective loss",
            "restored_trial": restored_trial,
        }
        state.trial_history[-1] = latest
        _clear_expansion_baseline(state)
        rollback_payload = {
            "cdg": state.cdg,
            "export_bundle": state.export_bundle,
            "benchmark": state.benchmark,
            "node_params": state.node_params,
            "done": True,
            "trial_history": state.trial_history,
        }
    elif state.expansion_candidate_active:
        state.expansion_candidate_active = False
        _clear_expansion_baseline(state)

    return {
        "benchmark": benchmark,
        "best_loss": state.best_loss,
        "best_node_params": state.best_node_params,
        "trial_history": state.trial_history,
        "current_trial": state.current_trial,
        "pending_param_search": state.pending_param_search,
        "param_trials_remaining": state.param_trials_remaining,
        "expansion_candidate_active": state.expansion_candidate_active,
        "expansion_baseline_trial": state.expansion_baseline_trial,
        "expansion_baseline_loss": state.expansion_baseline_loss,
        "expansion_baseline_thread_id": state.expansion_baseline_thread_id,
        "expansion_baseline_cdg": state.expansion_baseline_cdg,
        "expansion_baseline_export_bundle": state.expansion_baseline_export_bundle,
        "expansion_baseline_benchmark": state.expansion_baseline_benchmark,
        "expansion_baseline_node_params": state.expansion_baseline_node_params,
        **rollback_payload,
    }


def _summarize_expansion_context(context: ExpansionContext) -> dict[str, Any]:
    """Return a compact telemetry summary for the runtime expansion context."""
    signal_data = context.signal_data or {}
    intermediates = context.intermediates or {}
    eval_result = context.eval_result or {}
    return {
        "signal_keys": sorted(signal_data.keys())[:12],
        "intermediate_keys": sorted(intermediates.keys())[:16],
        "has_eval_result": bool(eval_result),
        "eval_keys": (
            sorted(eval_result.keys())[:16]
            if isinstance(eval_result, dict)
            else []
        ),
    }


def _build_expansion_context(state: PrincipalState) -> ExpansionContext:
    """Construct a best-effort runtime context from the latest evaluation artifacts."""
    artifacts = (
        dict(state.benchmark.runtime_artifacts)
        if state.benchmark is not None
        else {}
    )
    stdout_payload = artifacts.get("stdout_payload", {})
    eval_result: dict[str, Any]
    if isinstance(stdout_payload, dict):
        eval_result = dict(stdout_payload)
    else:
        eval_result = {}
    if state.benchmark is not None:
        eval_result.setdefault("global_loss", state.benchmark.global_loss)
    intermediates = artifacts.get("intermediates", {})
    signal_data = artifacts.get("signal_data", {})
    if not isinstance(intermediates, dict):
        intermediates = {}
    if not isinstance(signal_data, dict):
        signal_data = {}
    return ExpansionContext(
        intermediates=dict(intermediates),
        eval_result=eval_result or None,
        signal_data=dict(signal_data) or None,
    )


async def apply_expansion(state: PrincipalState, config: RunnableConfig) -> dict:
    """Apply family-agnostic CDG expansion rules using runtime diagnostics."""
    deps: PrincipalDeps = config["configurable"]["deps"]
    if state.done:
        return {"done": True}
    if state.cdg is None or state.benchmark is None:
        state.expansion_applied = False
        state.expansion_rules_applied = []
        return {
            "expansion_applied": False,
            "expansion_rules_applied": [],
        }

    engine = deps.expansion_engine or ExpansionEngine(default_rule_sets())
    context = _build_expansion_context(state)
    result = engine.expand(state.cdg, context)
    state.expansion_applied = bool(result.expanded)
    state.expansion_rules_applied = list(result.applied_rules)

    if state.trial_history:
        latest = dict(state.trial_history[-1])
        latest["expansion"] = {
            "applied": bool(result.expanded),
            "rules_applied": list(result.applied_rules),
            "diagnostic_count": len(result.diagnostics),
            "diagnostic_rule_names": sorted(
                {diag.rule_name for diag in result.diagnostics}
            ),
            "context_summary": _summarize_expansion_context(context),
        }
        state.trial_history[-1] = latest

    if not result.expanded:
        return {
            "expansion_applied": False,
            "expansion_rules_applied": [],
            "trial_history": state.trial_history,
            "expansion_candidate_active": state.expansion_candidate_active,
            "expansion_baseline_trial": state.expansion_baseline_trial,
            "expansion_baseline_loss": state.expansion_baseline_loss,
            "expansion_baseline_thread_id": state.expansion_baseline_thread_id,
            "expansion_baseline_cdg": state.expansion_baseline_cdg,
            "expansion_baseline_export_bundle": state.expansion_baseline_export_bundle,
            "expansion_baseline_benchmark": state.expansion_baseline_benchmark,
            "expansion_baseline_node_params": state.expansion_baseline_node_params,
        }

    state.expansion_candidate_active = True
    state.expansion_baseline_trial = state.current_trial
    state.expansion_baseline_loss = (
        state.benchmark.global_loss if state.benchmark is not None else None
    )
    state.expansion_baseline_thread_id = state.thread_id
    state.expansion_baseline_cdg = state.cdg.model_copy(deep=True)
    state.expansion_baseline_export_bundle = (
        state.export_bundle.model_copy(deep=True)
        if state.export_bundle is not None
        else None
    )
    state.expansion_baseline_benchmark = (
        state.benchmark.model_copy(deep=True)
        if state.benchmark is not None
        else None
    )
    state.expansion_baseline_node_params = dict(state.node_params)
    state.cdg = result.cdg
    state.node_params = {}
    state.param_signature = ""
    state.hpo_trial_number = None
    has_tunables = _structure_has_tunables(result.cdg, deps.catalog)
    state.pending_param_search = has_tunables and deps.param_trials_per_structure > 0
    state.param_trials_remaining = (
        deps.param_trials_per_structure if has_tunables else 0
    )
    logger.info(
        "Trial %d expansion applied: %s",
        state.current_trial,
        ", ".join(result.applied_rules) or "(none)",
    )
    return {
        "cdg": result.cdg,
        "expansion_applied": True,
        "expansion_rules_applied": list(result.applied_rules),
        "trial_history": state.trial_history,
        "pending_param_search": state.pending_param_search,
        "param_trials_remaining": state.param_trials_remaining,
        "expansion_candidate_active": state.expansion_candidate_active,
        "expansion_baseline_trial": state.expansion_baseline_trial,
        "expansion_baseline_loss": state.expansion_baseline_loss,
        "expansion_baseline_thread_id": state.expansion_baseline_thread_id,
        "expansion_baseline_cdg": state.expansion_baseline_cdg,
        "expansion_baseline_export_bundle": state.expansion_baseline_export_bundle,
        "expansion_baseline_benchmark": state.expansion_baseline_benchmark,
        "expansion_baseline_node_params": state.expansion_baseline_node_params,
    }


def _clear_expansion_baseline(state: PrincipalState) -> None:
    """Clear any stored pre-expansion baseline snapshot."""
    state.expansion_baseline_trial = None
    state.expansion_baseline_loss = None
    state.expansion_baseline_thread_id = ""
    state.expansion_baseline_cdg = None
    state.expansion_baseline_export_bundle = None
    state.expansion_baseline_benchmark = None
    state.expansion_baseline_node_params = {}


async def _evaluate_proposal_candidate(
    state: PrincipalState,
    deps: "PrincipalDeps",
    cdg: CDGExport,
) -> tuple[float, ExportBundle | None, BenchmarkResult | None, list[Any], GhostSimReport]:
    """Evaluate a proposal candidate without mutating trial history."""
    match_results = deps.match_results_fn(cdg) if deps.match_results_fn else []
    if inspect.isawaitable(match_results):
        match_results = await match_results
    match_results = list(match_results)
    ghost_report = run_ghost_simulation(cdg, match_results)
    bundle: ExportBundle | None = None
    benchmark: BenchmarkResult | None = None
    if deps.synthesize_fn is not None:
        bundle = await deps.synthesize_fn(cdg, match_results)
    if state.metric == OptimizationMetric.STRUCTURE:
        benchmark = benchmark_from_ghost_report(ghost_report)
    elif bundle is not None:
        sandbox = deps.sandbox
        if state.dataset_path.endswith((".yml", ".yaml")):
            benchmark = await sandbox.evaluate_adapter(
                bundle,
                state.dataset_path,
                state.metric,
                varset=deps.dataset_varset,
                evaluation_spec=deps.evaluation_spec,
            )
        else:
            benchmark = await sandbox.evaluate(
                bundle,
                state.dataset_path,
                state.metric,
                evaluation_spec=deps.evaluation_spec,
            )
    loss = (
        float(benchmark.global_loss)
        if benchmark is not None
        else float("inf")
    )
    return loss, bundle, benchmark, match_results, ghost_report


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
    context = _build_expansion_context(state)
    expansion = engine.expand(baseline_cdg, context)
    if expansion.expanded:
        loss, bundle, benchmark, match_results, ghost_report = await _evaluate_proposal_candidate(
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
        loss, bundle, benchmark, match_results, ghost_report = await _evaluate_proposal_candidate(
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
            "context_summary": _summarize_expansion_context(context),
        }
        state.trial_history[-1] = latest

    if selected is None:
        state.selected_proposal = ""
        return {
            "selected_proposal": "",
            "trial_history": state.trial_history,
            "expansion_applied": False,
            "expansion_rules_applied": [],
        }

    state.cdg = selected["cdg"]
    state.export_bundle = selected.get("bundle")
    state.benchmark = selected.get("benchmark")
    state.match_results = list(selected.get("match_results", []))
    state.ghost_report = selected["ghost_report"]
    state.expansion_applied = selected["label"] == "expansion"
    state.expansion_rules_applied = list(selected.get("rules_applied", []))
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

    architect = deps.architect
    bottleneck_name = next(
        (node.name for node in state.cdg.nodes if node.node_id == state.bottleneck_node_id),
        None,
    )

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
    has_tunables = _structure_has_tunables(mutation.cdg, deps.catalog)
    state.pending_param_search = has_tunables and deps.param_trials_per_structure > 0
    state.param_trials_remaining = deps.param_trials_per_structure if has_tunables else 0

    logger.info(
        "Time-travel: forked at checkpoint %s -> thread %s",
        target_cp,
        new_thread_id,
    )

    return {
        "cdg": mutation.cdg,
        "thread_id": new_thread_id,
        "pending_param_search": state.pending_param_search,
        "param_trials_remaining": state.param_trials_remaining,
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_after_gradients(state: PrincipalState) -> str:
    """Decide whether to continue optimising or stop."""
    if state.done:
        return "end"
    if len(state.trial_history) >= state.max_trials and state.current_trial > 1:
        return "end"
    if state.error and "pruned" not in state.error.lower():
        return "end"
    if state.pending_param_search and state.param_trials_remaining > 0:
        return "suggest_params"
    return "select_proposal"


def route_after_expansion(state: PrincipalState) -> str:
    """Backward-compatible routing helper for legacy expansion tests."""
    if state.done:
        return "end"
    if len(state.trial_history) >= state.max_trials:
        return "end"
    if state.expansion_applied:
        return "suggest_params"
    return "gradients"


def route_after_proposal(state: PrincipalState) -> str:
    """After proposal comparison, either evaluate the chosen branch or continue."""
    if state.done:
        return "end"
    if len(state.trial_history) >= state.max_trials:
        return "end"
    if state.selected_proposal:
        return "suggest_params"
    return "time_travel"


def route_after_update(state: PrincipalState) -> str:
    """After time-travel update, loop back or stop."""
    if state.done:
        return "end"
    if len(state.trial_history) >= state.max_trials:
        return "end"
    return "suggest_params"


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
    hpo_manager: OptunaManager | None = None
    param_trials_per_structure: int = 1
    expansion_engine: ExpansionEngine | None = None


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
    graph.add_node("expand", apply_expansion)
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
