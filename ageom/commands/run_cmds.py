"""Command for full orchestration: decompose -> match -> refine -> assemble."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ageom.commands._helpers import (
    _create_llm_router,
    _create_proof_env,
    _create_shared_context,
    _load_architect_catalog,
    _load_semantic_index,
    _load_skill_index_or_empty,
    _mode_feature_summary,
    _print_mode_summary,
    _print_prompt_routing_summary,
    _print_retrieval_policy,
    _print_shared_context_metrics,
    _resolve_retrieval_policy,
    _routing_metadata_summary,
    _shared_context_metadata,
    _summarize_prompt_routing,
    _warm_llm_if_supported,
    _write_shared_context_metrics_file,
)

if TYPE_CHECKING:
    from ageom.types import Prover


def _build_rapid_direct_cdg(
    goal: str,
    prover: "Prover",
    match_result: Any,
):
    """Build a minimal one-node CDG for rapid direct-match runs."""
    from ageom.services import build_direct_goal_cdg

    return build_direct_goal_cdg(
        goal,
        prover,
        match_result,
        execution_mode="rapid",
        conceptual_summary="Rapid-mode direct retrieval without architect decomposition.",
    )


async def _run_rapid_direct_match(
    goal: str,
    *,
    prover: "Prover",
    hunter: Any,
):
    """Run the rapid-mode direct Hunter path and wrap it in an orchestration result."""
    from ageom.services import HunterDirectMatchRequest, HunterService

    service = HunterService(hunter)
    match_result = await service.match_goal(
        HunterDirectMatchRequest(
            goal=goal,
            prover=prover,
            informal_desc="rapid direct baseline without decomposition",
            context={"execution_mode": "rapid", "rapid_direct_path": "true"},
        )
    )
    return service.direct_match_result(
        goal,
        prover,
        match_result,
        execution_mode="rapid",
        informal_desc="Rapid-mode direct retrieval without architect decomposition.",
        context={"execution_mode": "rapid", "rapid_direct_path": "true"},
    )


async def _run_structured_single_pass(
    cdg: Any,
    *,
    prover: "Prover",
    hunter: Any,
):
    """Run one Hunter pass over decomposed leaves without orchestration refinement."""
    from ageom.architect.handoff import to_pdg_nodes
    from ageom.orchestrator import OrchestratorResult
    from ageom.services import HunterBatchMatchRequest, HunterService

    pdg_nodes = to_pdg_nodes(cdg, prover=prover, strict=False)
    service = HunterService(hunter)
    batch_result = await service.match_batch(HunterBatchMatchRequest(pdg_nodes=pdg_nodes))
    return OrchestratorResult(
        cdg=cdg,
        match_results=batch_result.match_results,
        rounds_used=1,
        failures=batch_result.failures,
        ungroundable=batch_result.ungroundable,
    )


async def _cmd_run(args: argparse.Namespace) -> None:
    """Run the full orchestration loop: decompose -> match -> refine -> assemble."""
    from ageom.architect.catalog import PrimitiveCatalog, seed_builtin_primitives
    from ageom.architect.checkpointer import create_checkpointer
    from ageom.architect.graph import DecompositionAgent
    from ageom.architect.handoff import save_json
    from ageom.config import AgeomConfig, resolve_execution_mode
    from ageom.hunter.graph import HunterAgent
    from ageom.judge.checker import VerificationOracleImpl
    from ageom.orchestrator import run_orchestration
    from ageom.services import ArchitectService, HunterService, SingleAgentPlanner
    from ageom.telemetry import (
        configure_dashboard_output,
        finish_run,
        get_event_log,
        merge_run_metadata,
        start_run,
        telemetry_scope,
        telemetry_stage,
        update_stage,
    )
    from ageom.types import Prover

    config = AgeomConfig()
    mode_settings = resolve_execution_mode(config, getattr(args, "mode", None))
    prover = Prover(args.prover)
    output_dir = Path(args.output) if args.output else Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    _print_mode_summary("run", mode_settings)

    configure_dashboard_output(config.telemetry_runs_dir)
    event_log = get_event_log()
    event_log.configure_live_output(None)
    event_log.clear()
    if getattr(args, "trace", False):
        event_log.configure_live_output(output_dir / "trace.jsonl")
    catalog, catalog_alignment = _load_architect_catalog(args, config)
    retrieval_policy = _resolve_retrieval_policy(
        mode_settings=mode_settings,
        catalog=catalog,
        texts=[args.goal],
        config=config,
    )
    _print_retrieval_policy(retrieval_policy)
    architect_routing = None
    if mode_settings.mode != "rapid":
        architect_routing = _summarize_prompt_routing(
            config,
            "architect",
            [
                "architect_strategy",
                "architect_decompose",
                "architect_critique",
                "orchestrator_refine",
            ],
            mode_settings.mode,
        )
    hunter_routing = _summarize_prompt_routing(
        config,
        "hunter",
        [
            "hunter_score",
            "hunter_reformulate",
            "hunter_analyze_failure",
        ],
        mode_settings.mode,
    )
    telemetry_run_id = start_run(
        "algorithm_creation",
        metadata={
            "command": "run",
            "goal": args.goal,
            "prover": prover.value,
            "max_rounds": int(args.max_rounds),
            "execution_mode": mode_settings.mode,
            "execution_path": (
                "rapid_direct"
                if mode_settings.mode == "rapid"
                else "structured_single_pass"
                if mode_settings.mode == "structured"
                else "single_agent_planner"
                if mode_settings.mode == "single_agent"
                else "verified_orchestration"
            ),
            "mode_features": _mode_feature_summary(mode_settings),
            "retrieval_policy": {
                "catalog_confidence": retrieval_policy.catalog_confidence,
                "confidence_band": retrieval_policy.confidence_band,
                "skill_index": retrieval_policy.skill_index_enabled,
                "graph_retrieval": retrieval_policy.graph_retrieval_enabled,
                "semantic_backend": retrieval_policy.semantic_index_backend_override
                or "default",
                "hunter_mode": retrieval_policy.hunter_mode,
            },
            "rapid_direct_path": mode_settings.mode == "rapid",
            "single_agent_mode": mode_settings.mode == "single_agent",
            "llm_routing": (
                {
                    "hunter": _routing_metadata_summary(hunter_routing),
                }
                if architect_routing is None
                else {
                    "architect": _routing_metadata_summary(architect_routing),
                    "hunter": _routing_metadata_summary(hunter_routing),
                }
            ),
            "catalog_alignment": catalog_alignment,
        },
    )

    architect_shared_metrics = None
    hunter_shared_metrics = None
    result = None
    planner_result = None
    try:
        with telemetry_scope(run_id=telemetry_run_id):
            update_stage(stage="setup", status="running", message="loading dependencies")

            skill_index = None
            llm = None
            if mode_settings.mode != "rapid":
                skill_index = _load_skill_index_or_empty(
                    config,
                    enabled=retrieval_policy.skill_index_enabled,
                )

                try:
                    from ageom.llm_router import (
                        ARCHITECT_CRITIQUE,
                        ARCHITECT_DECOMPOSE,
                        ARCHITECT_STRATEGY,
                        ORCHESTRATOR_REFINE,
                    )

                    architect_prompt_keys = [
                        ARCHITECT_STRATEGY,
                        ARCHITECT_DECOMPOSE,
                        ARCHITECT_CRITIQUE,
                        ORCHESTRATOR_REFINE,
                    ]
                    _print_prompt_routing_summary(
                        config,
                        "architect",
                        architect_prompt_keys,
                        mode_settings.mode,
                    )
                    llm = _create_llm_router(
                        args,
                        config,
                        "architect",
                        architect_prompt_keys,
                    )
                    await _warm_llm_if_supported(llm, "architect")
                except (ValueError, ImportError) as exc:
                    print(f"Error: {exc}", file=sys.stderr)
                    finish_run(telemetry_run_id, status="failed", error=str(exc))
                    sys.exit(1)
            update_stage(stage="setup", status="completed")

            cdg = None
            architect_service = None
            if mode_settings.mode not in {"rapid", "single_agent"}:
                print(f"Decomposing: {args.goal}")

                retriever = None
                graph_store_ctx = None
                if retrieval_policy.graph_retrieval_enabled:
                    from ageom.architect.graph_retrieval import make_retriever
                    from ageom.graph_store import GraphStore as _GraphStore

                    graph_store_ctx = _GraphStore(
                        uri=config.memgraph_uri,
                        user=config.memgraph_user,
                        password=config.memgraph_password,
                    )
                architect_run_id = uuid.uuid4().hex
                architect_shared_context, architect_shared_metrics = await _create_shared_context(
                    config,
                    enabled=mode_settings.architect_shared_context_enabled,
                )

                with telemetry_stage("architect_decompose", message="building initial CDG"):
                    async with create_checkpointer(config.postgres_uri) as checkpointer:
                        if graph_store_ctx is not None:
                            async with graph_store_ctx as gstore:
                                retriever = make_retriever(config, gstore, current_repo="")
                                architect = DecompositionAgent(
                                    catalog=catalog,
                                    skill_index=skill_index,
                                    llm=llm,
                                    checkpointer=checkpointer,
                                    graph_retriever=retriever,
                                    shared_context=architect_shared_context,
                                    shared_context_metrics=architect_shared_metrics,
                                    context_namespace=f"architect/{architect_run_id}",
                                    context_budget_chars=config.architect_shared_context_budget_chars,
                                )
                                architect_service = ArchitectService(architect)
                                cdg = await architect.decompose(args.goal)
                        else:
                            architect = DecompositionAgent(
                                catalog=catalog,
                                skill_index=skill_index,
                                llm=llm,
                                checkpointer=checkpointer,
                                shared_context=architect_shared_context,
                                shared_context_metrics=architect_shared_metrics,
                                context_namespace=f"architect/{architect_run_id}",
                                context_budget_chars=config.architect_shared_context_budget_chars,
                            )
                            architect_service = ArchitectService(architect)
                            cdg = await architect.decompose(args.goal)

                print(f"  Decomposed: {len(cdg.nodes)} nodes, {len(cdg.edges)} edges")
            elif mode_settings.mode == "rapid":
                print(f"Rapid mode: matching goal directly without decomposition: {args.goal}")
            else:
                print("Single-agent mode: planner will attempt direct grounding before decomposition.")

                async def _architect_factory():
                    nonlocal architect_service, architect_shared_metrics
                    if architect_service is not None:
                        return architect_service

                    architect_run_id = uuid.uuid4().hex
                    architect_shared_context, architect_shared_metrics = await _create_shared_context(
                        config,
                        enabled=mode_settings.architect_shared_context_enabled,
                    )
                    class _LazyArchitectService:
                        async def decompose(self, request):
                            with telemetry_stage(
                                "architect_decompose",
                                message="building planner decomposition",
                            ):
                                async with create_checkpointer(config.postgres_uri) as checkpointer:
                                    architect = DecompositionAgent(
                                        catalog=catalog,
                                        skill_index=skill_index,
                                        llm=llm,
                                        checkpointer=checkpointer,
                                        shared_context=architect_shared_context,
                                        shared_context_metrics=architect_shared_metrics,
                                        context_namespace=f"architect/{architect_run_id}",
                                        context_budget_chars=config.architect_shared_context_budget_chars,
                                    )
                                    service = ArchitectService(architect)
                                    result = await service.decompose(request)
                                    built_cdg = result.cdg
                                    print(
                                        f"  Decomposed: {len(built_cdg.nodes)} nodes, {len(built_cdg.edges)} edges"
                                    )
                                    return result

                    architect_service = _LazyArchitectService()
                    return architect_service

            index_dir = config.index_dir
            if not index_dir.exists():
                print(
                    f"Error: index directory {index_dir} not found. Run 'ageom index build' first.",
                    file=sys.stderr,
                )
                finish_run(
                    telemetry_run_id,
                    status="failed",
                    error=f"missing index directory: {index_dir}",
                )
                sys.exit(1)

            env = None
            with telemetry_stage("hunter_setup", message="loading retrieval index"):
                index, index_mode = _load_semantic_index(
                    index_dir,
                    config,
                    backend_override=retrieval_policy.semantic_index_backend_override,
                )
                if index_mode != "faiss":
                    print(
                        "Warning: FAISS unavailable; using lexical fallback index for Hunter.",
                        file=sys.stderr,
                    )

                env = _create_proof_env(prover, config)
                if prover == Prover.LEAN4:
                    oracle = VerificationOracleImpl(lean_env=env)
                elif prover == Prover.PYTHON:
                    oracle = VerificationOracleImpl(python_env=env)
                else:
                    oracle = VerificationOracleImpl(coq_env=env)

                try:
                    from ageom.llm_router import (
                        HUNTER_ANALYZE_FAILURE,
                        HUNTER_REFORMULATE,
                        HUNTER_SCORE,
                    )

                    hunter_prompt_keys = [
                        HUNTER_SCORE,
                        HUNTER_REFORMULATE,
                        HUNTER_ANALYZE_FAILURE,
                    ]
                    _print_prompt_routing_summary(
                        config,
                        "hunter",
                        hunter_prompt_keys,
                        mode_settings.mode,
                    )
                    hunter_llm = _create_llm_router(
                        args,
                        config,
                        "hunter",
                        hunter_prompt_keys,
                    )
                    await _warm_llm_if_supported(hunter_llm, "hunter")
                except (ValueError, ImportError) as exc:
                    print(f"Error setting up hunter LLM: {exc}", file=sys.stderr)
                    finish_run(telemetry_run_id, status="failed", error=str(exc))
                    sys.exit(1)

                run_id = uuid.uuid4().hex
                hunter_shared_context, hunter_shared_metrics = await _create_shared_context(
                    config,
                    enabled=mode_settings.hunter_shared_context_enabled,
                )
                hunter = HunterAgent(
                    index=index,
                    oracle=oracle,
                    llm=hunter_llm,
                    max_iterations=config.hunter_max_iterations,
                    top_k_verify=config.hunter_top_k_verify,
                    search_k=config.hunter_search_k,
                    mode=retrieval_policy.hunter_mode,
                    use_gbnf=mode_settings.hunter_use_gbnf,
                    query_batch_size=config.hunter_query_batch_size,
                    top_k_per_query=config.hunter_top_k_per_query,
                    max_candidates_total=config.hunter_max_candidates_total,
                    shared_context=hunter_shared_context,
                    shared_context_metrics=hunter_shared_metrics,
                    context_namespace="hunter",
                    run_id=run_id,
                    context_budget_chars=config.hunter_shared_context_budget_chars,
                )
                hunter_service = HunterService(hunter)

            try:
                if mode_settings.mode == "rapid":
                    with telemetry_stage(
                        "rapid_direct_match",
                        message="matching goal directly",
                    ):
                        result = await _run_rapid_direct_match(
                            args.goal,
                            prover=prover,
                            hunter=hunter,
                        )
                elif mode_settings.mode == "single_agent":
                    print("Running single-agent planner...")

                    async def _architect_factory():
                        if architect_service is None:
                            raise RuntimeError("Architect service unavailable in single_agent mode")
                        return architect_service

                    planner = SingleAgentPlanner(
                        hunter=hunter_service,
                        architect_factory=_architect_factory,
                        orchestrate=run_orchestration,
                        llm=llm,
                        hunter_agent=hunter,
                        prover=prover,
                        max_rounds=args.max_rounds,
                        hunter_concurrency=config.orchestrator_hunter_concurrency,
                    )
                    with telemetry_stage(
                        "single_agent_planner",
                        message="tool-orchestrated direct->decompose->escalate planner",
                    ):
                        planner_result = await planner.run(args.goal)
                        result = planner_result.result
                    update_stage(
                        stage="single_agent_planner",
                        completed=len(planner_result.steps),
                        total=len(planner_result.steps),
                    )
                elif mode_settings.mode == "structured":
                    print("Running structured single-pass matching...")
                    with telemetry_stage(
                        "structured_match",
                        message="matching decomposed leaves once",
                    ):
                        result = await _run_structured_single_pass(
                            cdg,
                            prover=prover,
                            hunter=hunter,
                        )
                    update_stage(
                        stage="structured_match",
                        completed=len(result.match_results),
                        total=len(result.match_results),
                    )
                else:
                    print(f"Running orchestration (max {args.max_rounds} rounds)...")
                    with telemetry_stage(
                        "orchestration",
                        message="architect->hunter refine loop",
                        total=int(args.max_rounds),
                    ):
                        result = await run_orchestration(
                            cdg,
                            hunter_agent=hunter,
                            llm=llm,
                            prover=prover,
                            max_rounds=args.max_rounds,
                            hunter_concurrency=config.orchestrator_hunter_concurrency,
                        )
                    update_stage(
                        stage="orchestration",
                        completed=result.rounds_used,
                        total=int(args.max_rounds),
                    )
            finally:
                if env is not None:
                    await env.close()
    except Exception as exc:
        finish_run(telemetry_run_id, status="failed", error=str(exc))
        raise
    else:
        pass
    finally:
        event_log.configure_live_output(None)

    # Output
    print("\nRun complete:")
    print(f"  Rounds used: {result.rounds_used}")
    print(
        f"  Matches: {sum(1 for mr in result.match_results if mr.success)}/{len(result.match_results)}"
    )
    if result.ungroundable:
        print(f"  Ungroundable: {result.ungroundable}")

    save_json(result.cdg, output_dir / "cdg.json")

    if result.match_results:

        matches_data = [mr.to_dict() for mr in result.match_results]
        with open(output_dir / "matches.json", "w") as f:
            json.dump(matches_data, f, indent=2)

    print(f"  Output: {output_dir}/")
    _print_shared_context_metrics("architect", architect_shared_metrics)
    _print_shared_context_metrics("hunter", hunter_shared_metrics)
    metrics_path = _write_shared_context_metrics_file(
        output_dir / "shared_context_metrics.json",
        {
            "architect": architect_shared_metrics,
            "hunter": hunter_shared_metrics,
        },
    )
    merge_run_metadata(
        {
            "shared_context": _shared_context_metadata(
                {
                    "architect": architect_shared_metrics,
                    "hunter": hunter_shared_metrics,
                },
                metrics_path=metrics_path,
            )
        },
        run_id=telemetry_run_id,
    )
    if planner_result is not None:
        merge_run_metadata(
            {
                "execution_path": planner_result.execution_path,
                "single_agent": {
                    "steps": [
                        {
                            "action": step.action,
                            "detail": step.detail,
                            "status": step.status,
                        }
                        for step in planner_result.steps
                    ]
                },
            },
            run_id=telemetry_run_id,
        )
    if metrics_path is not None:
        print(f"  Shared context metrics: {metrics_path}")

    if getattr(args, "trace", False):
        if len(event_log) > 0:
            event_log.save(output_dir / "trace.jsonl")
            print(f"  Trace: {output_dir / 'trace.jsonl'} ({len(event_log)} events)")

    finish_run(telemetry_run_id, status="completed")
