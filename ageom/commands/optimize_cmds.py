"""Commands for NAS/AutoML optimization and profiling."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any

from ageom.commands._helpers import (
    _create_llm_router,
    _create_shared_context,
    _load_architect_catalog,
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


async def _cmd_optimize(args: argparse.Namespace) -> None:
    """Run the Principal NAS/AutoML optimisation loop."""
    from ageom.architect.catalog import PrimitiveCatalog, seed_builtin_primitives
    from ageom.architect.checkpointer import create_checkpointer
    from ageom.architect.graph import DecompositionAgent
    from ageom.config import AgeomConfig, resolve_execution_mode
    from ageom.principal.evaluator import ExecutionSandbox
    from ageom.principal.graph import (
        PrincipalDeps,
        build_principal_graph,
    )
    from ageom.principal.models import OptimizationMetric

    config = AgeomConfig()
    mode_settings = resolve_execution_mode(config, getattr(args, "mode", None))
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

    metric = OptimizationMetric(args.metric)
    postgres_uri = "" if args.no_persist else config.postgres_uri
    architect_run_id = uuid.uuid4().hex
    architect_shared_context, architect_shared_metrics = await _create_shared_context(
        config,
        enabled=mode_settings.architect_shared_context_enabled,
    )

    print("Principal optimisation loop")
    print(f"  Goal: {args.goal}")
    print(f"  Metric: {metric.value}")
    print(f"  Trials: {args.trials}")
    print(f"  Benchmark: {args.benchmark}")
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

        sandbox = ExecutionSandbox(timeout_s=args.timeout)
        deps = PrincipalDeps(architect=architect, sandbox=sandbox)

        graph = build_principal_graph().compile()

        initial_state = {
            "goal": args.goal,
            "metric": metric,
            "dataset_path": args.benchmark,
            "max_trials": args.trials,
        }

        config_dict = {"configurable": {"deps": deps}}

        final_state = await graph.ainvoke(initial_state, config=config_dict)

    # Report
    print("\nOptimisation complete:")
    print(f"  Trials run: {final_state.get('current_trial', 0)}")
    print(f"  Best loss: {final_state.get('best_loss', float('inf')):.6f}")
    history = final_state.get("trial_history", [])
    if history:
        print("  Trial history:")
        for entry in history:
            print(f"    Trial {entry['trial']}: loss={entry['loss']:.6f}")
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
    from ageom.principal.models import OptimizationMetric
    from ageom.principal.profiler import profile_algorithm_error
    from ageom.synthesizer.models import ExportBundle

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
    metric = OptimizationMetric(args.metric)
    try:
        dataset_varset = _parse_dataset_vars(getattr(args, "dataset_var", None))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    runner_path = artifact_path.parent / "export_python_pkg" / "runner.py"
    executable_artifact = runner_path if runner_path.exists() else None

    bundle = ExportBundle(
        target="python-pkg",
        output_dir=artifact_path.parent,
        source_path=artifact_path,
        compiled_artifact=artifact_path,
        executable_artifact=executable_artifact,
    )

    print(f"Profiling {artifact_path.name} against {dataset_path.name} using metric {metric.value}...")

    try:
        gradients = await profile_algorithm_error(
            cdg=cdg,
            bundle=bundle,
            dataset_path=str(dataset_path),
            metric=metric,
            dataset_varset=dataset_varset or None,
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
