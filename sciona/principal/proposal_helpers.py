"""Shared helper routines for Principal proposal selection and time travel."""

from __future__ import annotations

import inspect
from typing import Any

from sciona.architect.handoff import CDGExport
from sciona.principal.atom_ledger import compute_slot_signature
from sciona.principal.evaluation_helpers import evaluate_bundle_for_metric
from sciona.principal.expansion import ExpansionContext
from sciona.principal.models import BenchmarkResult, OptimizationMetric
from sciona.principal.structure_objective import benchmark_from_ghost_report
from sciona.principal.variant_mutation import maybe_apply_bottleneck_variant
from sciona.synthesizer.ghost_sim import GhostSimReport, run_ghost_simulation
from sciona.synthesizer.models import ExportBundle


def summarize_expansion_context(context: ExpansionContext) -> dict[str, Any]:
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


def build_expansion_context(state: Any) -> ExpansionContext:
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


async def build_redecomposition_candidate(
    state: Any,
    deps: Any,
    *,
    bottleneck_name: str | None,
) -> tuple[CDGExport, str] | None:
    """Fork and re-decompose a candidate structure for the current bottleneck."""
    if not state.bottleneck_node_id or state.cdg is None:
        return None

    history = await deps.architect.get_state_history(state.thread_id)
    target_cp: str | None = None
    for entry in history:
        vals = entry.get("values", {})
        node_ids = {n.node_id for n in vals.get("nodes", [])}
        if state.bottleneck_node_id not in node_ids:
            target_cp = entry.get("checkpoint_id")
            break

    if target_cp is None and history:
        target_cp = history[-1].get("checkpoint_id")
    if target_cp is None:
        return None

    new_thread_id = await deps.architect.fork(state.thread_id, target_cp)
    constraint = (
        f"The previous decomposition caused a bottleneck: "
        f"{state.bottleneck_reason}. Re-decompose more efficiently."
    )

    if deps.atom_ledger is not None and deps.catalog is not None and state.cdg is not None:
        bottleneck_node = next(
            (n for n in state.cdg.nodes if n.node_id == state.bottleneck_node_id),
            None,
        )
        if bottleneck_node is not None and bottleneck_node.matched_primitive:
            node_map = {n.node_id: n for n in state.cdg.nodes}
            parent = (
                node_map.get(bottleneck_node.parent_id)
                if bottleneck_node.parent_id
                else None
            )
            slot = compute_slot_signature(bottleneck_node, parent)
            same_category = [
                p.name
                for p in deps.catalog.search_by_category(bottleneck_node.concept_type)
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
    cdg = await deps.architect.decompose(constrained_goal, thread_id=new_thread_id)
    mutation = maybe_apply_bottleneck_variant(
        cdg,
        bottleneck_name=bottleneck_name,
        atom_ledger=deps.atom_ledger,
        catalog=deps.catalog,
    )
    return mutation.cdg, new_thread_id


async def evaluate_proposal_candidate(
    state: Any,
    deps: Any,
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
        benchmark = await evaluate_bundle_for_metric(
            deps.sandbox,
            bundle,
            state.dataset_path,
            state.metric,
            dataset_varset=deps.dataset_varset,
            evaluation_spec=deps.evaluation_spec,
        )
    loss = float(benchmark.global_loss) if benchmark is not None else float("inf")
    return loss, bundle, benchmark, match_results, ghost_report
