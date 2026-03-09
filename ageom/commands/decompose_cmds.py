"""Commands for goal decomposition into CDGs."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any

from ageom.commands._helpers import (
    _create_llm_router,
    _load_architect_catalog,
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
    _create_shared_context,
)


async def _run_decompose(
    agent: Any,
    args: argparse.Namespace,
    max_depth: int,
    catalog: Any,
) -> Any:
    """Run decomposition and print summary — shared by retrieval on/off paths."""
    print(f"Decomposing: {args.goal}")
    print(f"  Max depth: {max_depth}, Catalog size: {catalog.size}")

    cdg = await agent.decompose(args.goal, thread_id=args.thread_id)

    thread_id = cdg.metadata.get("thread_id", "")
    print(f"  Thread ID: {thread_id}")

    by_status: dict[str, int] = {}
    for node in cdg.nodes:
        status = node.status.value
        by_status[status] = by_status.get(status, 0) + 1

    print("\nDecomposition complete:")
    print(f"  Nodes: {len(cdg.nodes)}, Edges: {len(cdg.edges)}")
    for status, count in sorted(by_status.items()):
        print(f"    {status}: {count}")
    print(f"  Complete: {cdg.is_complete()}")

    return cdg


async def _cmd_decompose(args: argparse.Namespace) -> None:
    """Decompose a goal into a Conceptual Dependency Graph."""
    from ageom.architect.checkpointer import create_checkpointer
    from ageom.architect.graph import DecompositionAgent
    from ageom.architect.handoff import save_json
    from ageom.config import AgeomConfig, resolve_execution_mode
    from ageom.telemetry import (
        configure_dashboard_output,
        finish_run,
        merge_run_metadata,
        start_run,
        telemetry_scope,
        telemetry_stage,
        update_stage,
    )

    config = AgeomConfig()
    mode_settings = resolve_execution_mode(config, getattr(args, "mode", None))
    max_depth = args.max_depth or config.architect_max_depth
    _print_mode_summary("decompose", mode_settings)

    configure_dashboard_output(config.telemetry_runs_dir)
    catalog, catalog_alignment = _load_architect_catalog(args, config)
    retrieval_policy = _resolve_retrieval_policy(
        mode_settings=mode_settings,
        catalog=catalog,
        texts=[args.goal],
    )
    _print_retrieval_policy(retrieval_policy)
    architect_routing = _summarize_prompt_routing(
        config,
        "architect",
        [
            "architect_strategy",
            "architect_decompose",
            "architect_critique",
        ],
        mode_settings.mode,
    )
    telemetry_run_id = start_run(
        "decompose",
        metadata={
            "command": "decompose",
            "goal": args.goal,
            "max_depth": int(max_depth),
            "execution_mode": mode_settings.mode,
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
            "llm_routing": {
                "architect": _routing_metadata_summary(architect_routing),
            },
            "catalog_alignment": catalog_alignment,
        },
    )

    if catalog.size == 0:
        print(
            "Warning: no catalog loaded. Decomposition will have no atomic stop conditions.",
            file=sys.stderr,
        )

    try:
        with telemetry_scope(run_id=telemetry_run_id):
            update_stage(stage="setup", status="running", message="loading dependencies")

            skill_index = _load_skill_index_or_empty(
                config,
                enabled=retrieval_policy.skill_index_enabled,
            )

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
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                finish_run(telemetry_run_id, status="failed", error=str(exc))
                sys.exit(1)
            except ImportError as exc:
                print(f"Error: missing LLM dependency ({exc})", file=sys.stderr)
                finish_run(telemetry_run_id, status="failed", error=str(exc))
                sys.exit(1)

            update_stage(stage="setup", status="completed")

            postgres_uri = "" if args.no_persist else config.postgres_uri
            architect_run_id = uuid.uuid4().hex
            architect_shared_context, architect_shared_metrics = await _create_shared_context(
                config,
                enabled=mode_settings.architect_shared_context_enabled,
            )
            architect_context_namespace = f"architect/{architect_run_id}"

            retriever = None
            graph_store_ctx = None
            if retrieval_policy.graph_retrieval_enabled:
                from ageom.architect.graph_retrieval import make_retriever
                from ageom.graph_store import GraphStore

                graph_store_ctx = GraphStore(
                    uri=config.memgraph_uri,
                    user=config.memgraph_user,
                    password=config.memgraph_password,
                )

            with telemetry_stage("architect_decompose", message="building CDG"):
                async with create_checkpointer(postgres_uri) as checkpointer:
                    if graph_store_ctx is not None:
                        async with graph_store_ctx as store:
                            retriever = make_retriever(config, store, current_repo="")
                            agent = DecompositionAgent(
                                catalog=catalog,
                                skill_index=skill_index,
                                llm=llm,
                                max_depth=max_depth,
                                checkpointer=checkpointer,
                                graph_retriever=retriever,
                                shared_context=architect_shared_context,
                                shared_context_metrics=architect_shared_metrics,
                                context_namespace=architect_context_namespace,
                                context_budget_chars=config.architect_shared_context_budget_chars,
                            )
                            cdg = await _run_decompose(agent, args, max_depth, catalog)
                    else:
                        agent = DecompositionAgent(
                            catalog=catalog,
                            skill_index=skill_index,
                            llm=llm,
                            max_depth=max_depth,
                            checkpointer=checkpointer,
                            shared_context=architect_shared_context,
                            shared_context_metrics=architect_shared_metrics,
                            context_namespace=architect_context_namespace,
                            context_budget_chars=config.architect_shared_context_budget_chars,
                        )
                        cdg = await _run_decompose(agent, args, max_depth, catalog)

            metrics_path = None
            if args.output:
                save_json(cdg, args.output)
                print(f"  Saved to: {args.output}")
                metrics_path = _write_shared_context_metrics_file(
                    Path(args.output).parent / "shared_context_metrics.json",
                    {"architect": architect_shared_metrics},
                )
                if metrics_path is not None:
                    print(f"  Shared context metrics: {metrics_path}")
            _print_shared_context_metrics("architect", architect_shared_metrics)
            merge_run_metadata(
                {
                    "shared_context": _shared_context_metadata(
                        {"architect": architect_shared_metrics},
                        metrics_path=metrics_path,
                    )
                },
                run_id=telemetry_run_id,
            )
        finish_run(telemetry_run_id, status="completed")
    except Exception as exc:
        finish_run(telemetry_run_id, status="failed", error=str(exc))
        raise


async def _cmd_history(args: argparse.Namespace) -> None:
    """Show checkpoint history for a decomposition thread."""
    from ageom.architect.checkpointer import create_checkpointer
    from ageom.architect.graph import DecompositionAgent
    from ageom.config import AgeomConfig
    from ageom.hunter.llm import LLMClient

    config = AgeomConfig()

    # Minimal no-op deps — we only need the compiled graph for state queries
    class _Stub(LLMClient):
        async def complete(self, system: str, user: str) -> str:
            return ""

        async def complete_with_grammar(
            self, system: str, user: str, grammar: str
        ) -> str:
            return ""

    async with create_checkpointer(config.postgres_uri) as checkpointer:
        agent = DecompositionAgent(
            catalog=None,  # type: ignore[arg-type]
            skill_index=None,  # type: ignore[arg-type]
            llm=_Stub(),
            checkpointer=checkpointer,
        )
        history = await agent.get_state_history(args.thread_id)

    if not history:
        print(f"No checkpoints found for thread {args.thread_id}")
        return

    print(f"Thread {args.thread_id}: {len(history)} checkpoint(s)\n")
    for i, entry in enumerate(history):
        vals = entry["values"]
        node_count = len(vals.get("nodes", []))
        pending = len(vals.get("pending_node_ids", []))
        done = vals.get("done", False)
        cp_id = entry.get("checkpoint_id", "?")
        print(f"  [{i}] checkpoint_id={cp_id}")
        print(f"      nodes={node_count}  pending={pending}  done={done}")
