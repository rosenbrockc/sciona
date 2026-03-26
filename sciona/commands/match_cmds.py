"""Command for matching predicates to library functions."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from sciona.commands.llm_helpers import _create_llm_router, _warm_llm_if_supported
from sciona.commands.routing_helpers import (
    _mode_feature_summary,
    _print_mode_summary,
    _print_prompt_routing_summary,
    _print_retrieval_policy,
    _resolve_retrieval_policy,
    _routing_metadata_summary,
    _summarize_prompt_routing,
)
from sciona.commands.runtime_helpers import (
    _create_proof_env,
    _load_architect_catalog,
    _load_semantic_index,
    _shutdown_telemetry_drain,
)
from sciona.commands.shared_context_helpers import (
    _create_shared_context,
    _print_shared_context_metrics,
    _shared_context_metadata,
)


async def _cmd_match(args: argparse.Namespace) -> None:
    """Match predicates to library functions."""
    from sciona.config import AgeomConfig, resolve_execution_mode
    from sciona.hunter.graph import HunterAgent
    from sciona.judge.checker import VerificationOracleImpl
    from sciona.telemetry import (
        configure_dashboard_output,
        configure_postgres_telemetry,
        finish_run,
        merge_run_metadata,
        start_run,
        telemetry_scope,
        telemetry_stage,
        update_stage,
    )
    from sciona.types import PDGNode, Prover

    config = AgeomConfig()
    mode_settings = resolve_execution_mode(config, getattr(args, "mode", None))
    _print_mode_summary("match", mode_settings)
    configure_dashboard_output(config.telemetry_runs_dir)

    _telem_drain = None
    _telem_store = None
    if config.telemetry_backend != "file" and config.postgres_uri:
        try:
            from sciona.telemetry_store import PostgresTelemetryStore, TelemetryDrain

            _telem_store = PostgresTelemetryStore(config.postgres_uri)
            await _telem_store.setup()
            _telem_drain = TelemetryDrain(_telem_store)
            configure_postgres_telemetry(_telem_store, _telem_drain)
            await _telem_drain.start()
        except Exception:
            _telem_drain = None
            _telem_store = None

    # Build PDG nodes
    nodes: list[PDGNode] = []
    prover = Prover(args.prover)
    if args.statement:
        nodes.append(
            PDGNode(
                predicate_id="cli-0",
                statement=args.statement,
                prover=prover,
            )
        )
    elif args.pdg_file:
        pdg_path = Path(args.pdg_file)
        with open(pdg_path) as f:
            pdg_data = json.load(f)
        for item in pdg_data:
            nodes.append(
                PDGNode(
                    predicate_id=item.get("predicate_id", ""),
                    statement=item["statement"],
                    informal_desc=item.get("informal_desc", ""),
                    prover=prover,
                    context=item.get("context", {}),
                )
            )
    else:
        print("Error: provide --statement or --pdg-file", file=sys.stderr)
        sys.exit(1)

    catalog, catalog_alignment = _load_architect_catalog(args, config)
    retrieval_policy = _resolve_retrieval_policy(
        mode_settings=mode_settings,
        catalog=catalog,
        texts=[f"{node.statement} {node.informal_desc}".strip() for node in nodes],
    )
    _print_retrieval_policy(retrieval_policy)
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
        "match",
        label=getattr(args, "label", ""),
        metadata={
            "command": "match",
            "prover": prover.value,
            "statement_count": len(nodes),
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
                "hunter": _routing_metadata_summary(hunter_routing),
            },
            "catalog_alignment": catalog_alignment,
        },
    )

    # Load index
    index_dir = Path(args.index_dir) if args.index_dir else config.index_dir
    if not index_dir.exists():
        print(
            f"Error: index directory {index_dir} not found. Run 'sciona index build' first.",
            file=sys.stderr,
        )
        finish_run(
            telemetry_run_id,
            status="failed",
            error=f"missing index directory: {index_dir}",
        )
        sys.exit(1)

    try:
        with telemetry_scope(run_id=telemetry_run_id):
            update_stage(stage="setup", status="running", message="loading dependencies")

            index, index_mode = _load_semantic_index(
                index_dir,
                config,
                backend_override=retrieval_policy.semantic_index_backend_override,
            )
            if index_mode == "lexical_fallback":
                print(
                    "Warning: FAISS unavailable; using lexical fallback index for Hunter.",
                    file=sys.stderr,
                )

            env = _create_proof_env(prover, config)
            try:
                if prover == Prover.LEAN4:
                    oracle = VerificationOracleImpl(lean_env=env)
                elif prover == Prover.PYTHON:
                    oracle = VerificationOracleImpl(python_env=env)
                else:
                    oracle = VerificationOracleImpl(coq_env=env)

                try:
                    from sciona.llm_router import (
                        HUNTER_ANALYZE_FAILURE,
                        HUNTER_REFORMULATE,
                        HUNTER_SCORE,
                    )

                    prompt_keys = [
                        HUNTER_SCORE,
                        HUNTER_REFORMULATE,
                        HUNTER_ANALYZE_FAILURE,
                    ]
                    _print_prompt_routing_summary(
                        config, "hunter", prompt_keys, getattr(args, "mode", None)
                    )
                    _embedder = getattr(index, "_embedder", None)
                    llm = _create_llm_router(
                        args, config, "hunter", prompt_keys, embedder=_embedder
                    )
                    await _warm_llm_if_supported(llm, "hunter")
                except ValueError as exc:
                    print(f"Error: {exc}", file=sys.stderr)
                    finish_run(telemetry_run_id, status="failed", error=str(exc))
                    sys.exit(1)
                except ImportError as exc:
                    print(f"Error: missing LLM dependency ({exc})", file=sys.stderr)
                    finish_run(telemetry_run_id, status="failed", error=str(exc))
                    sys.exit(1)

                run_id = uuid.uuid4().hex
                shared_context, shared_context_metrics = await _create_shared_context(
                    config,
                    enabled=mode_settings.hunter_shared_context_enabled,
                )
                agent = HunterAgent(
                    index=index,
                    oracle=oracle,
                    llm=llm,
                    max_iterations=config.hunter_max_iterations,
                    top_k_verify=config.hunter_top_k_verify,
                    search_k=config.hunter_search_k,
                    mode=retrieval_policy.hunter_mode,
                    use_gbnf=mode_settings.hunter_use_gbnf,
                    query_batch_size=config.hunter_query_batch_size,
                    top_k_per_query=config.hunter_top_k_per_query,
                    max_candidates_total=config.hunter_max_candidates_total,
                    shared_context=shared_context,
                    shared_context_metrics=shared_context_metrics,
                    context_namespace="hunter",
                    run_id=run_id,
                    context_budget_chars=config.hunter_shared_context_budget_chars,
                )

                update_stage(stage="setup", status="completed")
                with telemetry_stage(
                    "matching",
                    message="matching predicates",
                    total=len(nodes),
                ):
                    for idx, node in enumerate(nodes, start=1):
                        print(f"\nMatching: {node.statement}")
                        result = await agent.find_match(node)
                        update_stage(
                            stage="matching",
                            completed=idx,
                            total=len(nodes),
                            message=node.statement[:120],
                        )
                        if result.success:
                            assert result.verified_match is not None
                            print(f"  VERIFIED: {result.verified_match.candidate.declaration.name}")
                            print(
                                f"  Type: {result.verified_match.candidate.declaration.type_signature}"
                            )
                        else:
                            print(f"  NO MATCH FOUND ({len(result.all_candidates)} candidates tried)")
                            for vr in result.all_verifications:
                                print(
                                    f"    - {vr.candidate.declaration.name}: {vr.error_message[:100]}"
                                )
                _print_shared_context_metrics("hunter", shared_context_metrics)
                merge_run_metadata(
                    {
                        "shared_context": _shared_context_metadata(
                            {"hunter": shared_context_metrics},
                        )
                    },
                    run_id=telemetry_run_id,
                )
            finally:
                close_env = getattr(env, "close", None)
                if callable(close_env):
                    await close_env()
        finish_run(telemetry_run_id, status="completed")
        await _shutdown_telemetry_drain(_telem_drain, _telem_store)
    except Exception as exc:
        finish_run(telemetry_run_id, status="failed", error=str(exc))
        await _shutdown_telemetry_drain(_telem_drain, _telem_store)
        raise
