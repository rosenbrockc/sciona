"""Commands for NAS/AutoML optimization and profiling."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from ageom.commands._helpers import (
    _create_proof_env,
    _create_llm_router,
    _create_shared_context,
    _load_architect_catalog,
    _load_semantic_index,
    _load_skill_index_or_empty,
    _print_mode_summary,
    _print_prompt_routing_summary,
    _print_retrieval_policy,
    _print_shared_context_metrics,
    _resolve_retrieval_policy,
    _warm_llm_if_supported,
    _write_shared_context_metrics_file,
)


def _parse_dataset_vars(entries: list[str] | None) -> dict[str, str]:
    """Parse repeated ``KEY=VALUE`` CLI args into a dict."""
    result: dict[str, str] = {}
    for entry in entries or []:
        key, sep, value = entry.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(
                f"Invalid dataset var {entry!r}; expected KEY=VALUE."
            )
        result[key] = value
    return result


async def _match_results_for_optimize(
    cdg: Any,
    *,
    mode: str,
    prover: Any,
    hunter: Any,
    llm: Any,
) -> list[Any]:
    from ageom.commands.run_cmds import _is_signal_event_rate_scaffold, _run_structured_single_pass
    from ageom.orchestrator import run_orchestration

    if mode in {"structured", "verified", "rapid"}:
        if mode != "verified" or _is_signal_event_rate_scaffold(cdg):
            result = await _run_structured_single_pass(cdg, prover=prover, hunter=hunter)
        else:
            result = await run_orchestration(
                cdg,
                hunter_agent=hunter,
                llm=llm,
                prover=prover,
                max_rounds=3,
            )
        return list(result.match_results)
    raise ValueError(f"unsupported optimize mode {mode!r}")


async def _synthesize_export_bundle_for_optimize(
    cdg: Any,
    match_results: list[Any],
    *,
    prover: Any,
    config: Any,
    catalog: Any | None,
    output_root: Path,
    trial_index: int,
) -> Any:
    from ageom.services import (
        SynthesizerAssembleRequest,
        SynthesizerCompileRequest,
        SynthesizerService,
    )
    from ageom.synthesizer.extractor import ExportTarget, Extractor
    from ageom.synthesizer.models import SkeletonFile, SynthesisResult

    service = SynthesizerService(prover=prover)
    tunable_params_by_primitive: dict[str, list[str]] = {}
    if catalog is not None:
        for node in cdg.nodes:
            primitive_name = str(getattr(node, "matched_primitive", "") or "").strip()
            if not primitive_name:
                continue
            prim = catalog.get(primitive_name)
            if prim is None or not prim.tunable_params:
                continue
            tunable_params_by_primitive[primitive_name] = [
                spec.name for spec in prim.tunable_params
            ]
    skeleton = service.assemble(
        SynthesizerAssembleRequest(
            cdg=cdg,
            match_results=match_results,
            tunable_params_by_primitive=tunable_params_by_primitive,
        )
    ).skeleton

    env = _create_proof_env(prover, config)
    try:
        compile_result = await service.compile(
            SynthesizerCompileRequest(skeleton=skeleton, env=env)
        )
    finally:
        await env.close()

    if not compile_result.result.compiled_ok:
        raise RuntimeError("optimize synthesis compile failed")

    trial_dir = output_root / f"trial_{trial_index:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    source_path = trial_dir / "verified.py"
    source_path.write_text(compile_result.result.skeleton.source_code)
    export_skeleton = SkeletonFile(
        prover=compile_result.result.skeleton.prover,
        source_code=source_path.read_text(),
    )
    synthesis_result = SynthesisResult(
        skeleton=export_skeleton,
        compiled_ok=True,
        sorry_remaining=compile_result.result.skeleton.sorry_count,
        patches_applied=0,
        iterations_used=0,
    )
    extractor = Extractor(config)
    bundle = await extractor.extract(
        synthesis_result,
        ExportTarget("python-pkg"),
        trial_dir / "export_python_pkg",
    )
    bundle.source_path = source_path
    bundle.output_dir = trial_dir
    if bundle.executable_artifact is None:
        bundle.executable_artifact = trial_dir / "export_python_pkg" / "runner.py"
    return bundle


async def _cmd_optimize(args: argparse.Namespace) -> None:
    """Run the Principal NAS/AutoML optimisation loop."""
    from ageom.architect.checkpointer import create_checkpointer
    from ageom.architect.graph import DecompositionAgent
    from ageom.config import AgeomConfig, resolve_execution_mode
    from ageom.hunter.graph import HunterAgent
    from ageom.judge.checker import VerificationOracleImpl
    from ageom.principal.atom_ledger import AtomLedger
    from ageom.principal.evaluator import ExecutionSandbox
    from ageom.principal.graph import (
        PrincipalDeps,
        build_principal_graph,
    )
    from ageom.principal.hpo import OptunaManager
    from ageom.principal.metric_selection import resolve_optimization_objective
    from ageom.types import Prover

    config = AgeomConfig()
    mode_settings = resolve_execution_mode(config, getattr(args, "mode", None))
    prover = Prover(args.prover)
    _print_mode_summary("optimize", mode_settings)

    catalog, _catalog_alignment = _load_architect_catalog(args, config)
    retrieval_policy = _resolve_retrieval_policy(
        mode_settings=mode_settings,
        catalog=catalog,
        texts=[args.goal],
    )
    _print_retrieval_policy(retrieval_policy)

    skill_index = _load_skill_index_or_empty(
        config,
        enabled=retrieval_policy.skill_index_enabled,
    )

    # LLM
    try:
        from ageom.llm_router import (
            ARCHITECT_CRITIQUE,
            ARCHITECT_DECOMPOSE,
            ARCHITECT_STRATEGY,
        )

        prompt_keys = [
            ARCHITECT_STRATEGY,
            ARCHITECT_DECOMPOSE,
            ARCHITECT_CRITIQUE,
        ]
        _print_prompt_routing_summary(
            config, "architect", prompt_keys, getattr(args, "mode", None)
        )
        llm = _create_llm_router(args, config, "architect", prompt_keys)
        await _warm_llm_if_supported(llm, "architect")
    except (ValueError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        metric, evaluation_spec, objective_label = resolve_optimization_objective(
            args.metric,
            getattr(args, "eval_spec", None),
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    postgres_uri = "" if args.no_persist else config.postgres_uri
    architect_run_id = uuid.uuid4().hex
    architect_shared_context, architect_shared_metrics = await _create_shared_context(
        config,
        enabled=mode_settings.architect_shared_context_enabled,
    )
    output_root = Path("output") / f"principal_optimize_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root.mkdir(parents=True, exist_ok=True)

    print("Principal optimisation loop")
    print(f"  Goal: {args.goal}")
    print(f"  Objective: {objective_label}")
    if objective_label != metric.value:
        print(f"  Execution metric: {metric.value} (reference loss: {objective_label})")
    else:
        print(f"  Execution metric: {metric.value}")
    print(f"  Trials: {args.trials}")
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Output: {output_root}")
    print()

    async with create_checkpointer(postgres_uri) as checkpointer:
        architect = DecompositionAgent(
            catalog=catalog,
            skill_index=skill_index,
            llm=llm,
            checkpointer=checkpointer,
            shared_context=architect_shared_context,
            shared_context_metrics=architect_shared_metrics,
            context_namespace=f"architect/{architect_run_id}",
            context_budget_chars=config.architect_shared_context_budget_chars,
            architect_critique_llm_enabled=config.architect_critique_llm_enabled,
        )

        index, _index_mode = _load_semantic_index(
            config.index_dir,
            config,
            backend_override=retrieval_policy.semantic_index_backend_override,
        )
        env = _create_proof_env(prover, config)
        if prover == Prover.LEAN4:
            oracle = VerificationOracleImpl(lean_env=env)
        elif prover == Prover.PYTHON:
            oracle = VerificationOracleImpl(python_env=env)
        else:
            oracle = VerificationOracleImpl(coq_env=env)

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
            config, "hunter", hunter_prompt_keys, getattr(args, "mode", None)
        )
        embedder = getattr(index, "_embedder", None)
        hunter_llm = _create_llm_router(
            args,
            config,
            "hunter",
            hunter_prompt_keys,
            embedder=embedder,
        )
        await _warm_llm_if_supported(hunter_llm, "hunter")
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
        )

        trial_counter = {"value": 0}

        async def _trial_synthesizer(cdg: Any, match_results: list[Any]) -> Any:
            trial_counter["value"] += 1
            return await _synthesize_export_bundle_for_optimize(
                cdg,
                match_results,
                prover=prover,
                config=config,
                catalog=catalog,
                output_root=output_root,
                trial_index=trial_counter["value"],
            )

        sandbox = ExecutionSandbox(timeout_s=args.timeout)
        atom_ledger = AtomLedger()
        hpo_manager = OptunaManager(study_name="principal")
        deps = PrincipalDeps(
            architect=architect,
            sandbox=sandbox,
            match_results_fn=lambda cdg: _match_results_for_optimize(
                cdg,
                mode=mode_settings.mode,
                prover=prover,
                hunter=hunter,
                llm=llm,
            ),
            synthesize_fn=_trial_synthesizer,
            evaluation_spec=evaluation_spec,
            dataset_varset=_parse_dataset_vars(getattr(args, "dataset_var", None)),
            atom_ledger=atom_ledger,
            catalog=catalog,
            hpo_manager=hpo_manager,
            param_trials_per_structure=2,
        )
        graph = build_principal_graph().compile()

        initial_state = {
            "goal": args.goal,
            "metric": metric,
            "dataset_path": args.benchmark,
            "max_trials": args.trials,
        }

        config_dict = {"configurable": {"deps": deps}}

        try:
            final_state = await graph.ainvoke(initial_state, config=config_dict)
        finally:
            await env.close()

    # Report
    print("\nOptimisation complete:")
    print(f"  Trials run: {final_state.get('current_trial', 0)}")
    print(f"  Best loss: {final_state.get('best_loss', float('inf')):.6f}")
    history = final_state.get("trial_history", [])
    if history:
        print("  Trial history:")
        for entry in history:
            structure = entry.get("structure", {})
            print(
                f"    Trial {entry['trial']}: loss={entry['loss']:.6f} "
                f"nodes={structure.get('node_count', 0)} "
                f"edges={structure.get('edge_count', 0)} "
                f"primitive_sig={structure.get('primitive_signature', '')}"
            )
    (output_root / "trial_history.json").write_text(
        json.dumps(history, indent=2) + "\n"
    )
    _print_shared_context_metrics("architect", architect_shared_metrics)
    metrics_out_dir = Path("output")
    metrics_path = _write_shared_context_metrics_file(
        metrics_out_dir / "optimize_shared_context_metrics.json",
        {"architect": architect_shared_metrics},
    )
    if metrics_path is not None:
        print(f"  Shared context metrics: {metrics_path}")


async def _cmd_profile(args: argparse.Namespace) -> None:
    """Evaluate an existing CDG against a dataset and rank error contributors."""
    from ageom.architect.handoff import load_json
    from ageom.principal.metric_selection import resolve_optimization_objective
    from ageom.principal.profiler import profile_algorithm_error
    from ageom.synthesizer.models import ExportBundle
    from ageom.types import MatchResult

    cdg_path = Path(args.cdg)
    if not cdg_path.exists():
        print(f"Error: CDG file not found at {cdg_path}", file=sys.stderr)
        sys.exit(1)

    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        print(f"Error: Artifact file not found at {artifact_path}", file=sys.stderr)
        sys.exit(1)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset file not found at {dataset_path}", file=sys.stderr)
        sys.exit(1)

    cdg = load_json(cdg_path)
    try:
        dataset_varset = _parse_dataset_vars(getattr(args, "dataset_var", None))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    try:
        metric, evaluation_spec, objective_label = resolve_optimization_objective(
            args.metric,
            getattr(args, "eval_spec", None),
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    runner_candidates = [
        artifact_path.parent / "runner.py",
        artifact_path.parent / "export_python_pkg" / "runner.py",
    ]
    executable_artifact = next(
        (candidate for candidate in runner_candidates if candidate.exists()),
        None,
    )

    matches_path = artifact_path.parent / "matches.json"
    match_results: list[MatchResult] = []
    if matches_path.exists():
        try:
            with matches_path.open() as fh:
                matches_payload = json.load(fh)
            if isinstance(matches_payload, list):
                match_results = [MatchResult.from_dict(item) for item in matches_payload]
        except Exception:
            match_results = []

    bundle = ExportBundle(
        target="python-pkg",
        output_dir=artifact_path.parent,
        source_path=artifact_path,
        compiled_artifact=artifact_path,
        executable_artifact=executable_artifact,
    )

    print(
        f"Profiling {artifact_path.name} against {dataset_path.name} using objective {objective_label}..."
    )

    try:
        gradients = await profile_algorithm_error(
            cdg=cdg,
            bundle=bundle,
            dataset_path=str(dataset_path),
            metric=metric,
            dataset_varset=dataset_varset or None,
            match_results=match_results,
            evaluation_spec=evaluation_spec,
        )

        if not gradients:
            print("No gradients were computed. Ensure trace.jsonl is emitted properly.")
            return

        print("\n=== Profiling Results ===")
        print(f"{'Node ID':<20} | {'Score (%)':<10} | {'Reason'}")
        print("-" * 80)
        for g in gradients:
            print(f"{g.node_id:<20} | {g.gradient_score:<10.2f} | {g.bottleneck_reason}")

    except Exception as exc:
        print(f"Error during profiling: {exc}", file=sys.stderr)
        sys.exit(1)
