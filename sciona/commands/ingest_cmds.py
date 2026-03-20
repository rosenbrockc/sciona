"""Commands for source code ingestion (Round 0)."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from sciona.commands._helpers import (
    _create_llm_router,
    _create_proof_env,
    _create_shared_context,
    _load_semantic_index,
    _print_mode_summary,
    _print_prompt_routing_summary,
    _print_shared_context_metrics,
    _warm_llm_if_supported,
    _write_shared_context_metrics_file,
)


def _cmd_ingest_status(args: argparse.Namespace) -> None:
    """Inspect ingestion monitor status and return meaningful exit codes."""
    from sciona.ingester.monitor import COMPLETED_FILE, FAILED_FILE, IngestMonitor

    output_dir = Path(args.output)
    status = IngestMonitor.read_status(output_dir)
    derived_state = IngestMonitor.classify_state(
        status, stale_seconds=max(5, int(args.stale_seconds))
    )

    completed_path = output_dir / COMPLETED_FILE
    failed_path = output_dir / FAILED_FILE
    if completed_path.exists():
        derived_state = "completed"
    elif failed_path.exists():
        derived_state = "failed"

    payload = {
        "output_dir": str(output_dir),
        "derived_state": derived_state,
        "status": status,
        "has_completed_marker": completed_path.exists(),
        "has_failed_marker": failed_path.exists(),
    }

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        phase = str(status.get("phase", "")) if status else ""
        step = str(status.get("current_step", "")) if status else ""
        last_heartbeat = float(status.get("last_heartbeat_at") or 0.0) if status else 0.0
        heartbeat_age = max(0.0, time.time() - last_heartbeat) if last_heartbeat else 0.0
        print(f"state={derived_state}")
        print(f"output={output_dir}")
        if phase:
            print(f"phase={phase}")
        if step:
            print(f"step={step}")
        if last_heartbeat:
            print(f"heartbeat_age_sec={heartbeat_age:.1f}")
        if status.get("llm_call_inflight"):
            inflight = status["llm_call_inflight"]
            prompt_key = str(inflight.get("prompt_key", ""))
            print(f"llm_inflight={prompt_key}")
        if failed_path.exists():
            try:
                failed_payload = json.loads(failed_path.read_text())
                if isinstance(failed_payload, dict):
                    err = failed_payload.get("error")
                    if err:
                        print(f"error={err}")
            except json.JSONDecodeError:
                pass

    if derived_state in {"failed", "stalled"}:
        sys.exit(2)
    if derived_state in {"missing", "unknown"}:
        sys.exit(1)


async def _cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a source unit into the atom framework."""
    import os

    from sciona.config import AgeomConfig, resolve_execution_mode
    from sciona.ingester import IngesterAgent
    from sciona.ingester.monitor import IngestMonitor
    from sciona.types import Prover

    config = AgeomConfig()
    mode_settings = resolve_execution_mode(config, getattr(args, "mode", None))
    output_dir = Path(args.output) if args.output else Path("output") / args.class_name
    output_dir.mkdir(parents=True, exist_ok=True)
    _print_mode_summary("ingest", mode_settings)

    llm_provider = (
        getattr(args, "llm_provider", None)
        or config.ingester_llm_provider
        or config.llm_provider
    )
    llm_model = (
        getattr(args, "llm_model", None)
        or config.ingester_llm_model
        or config.llm_model
    )
    stale_seconds = max(5, int(getattr(args, "stale_seconds", 120)))
    monitor = IngestMonitor(
        output_dir,
        enable_trace=bool(getattr(args, "trace", False)),
        monitor_stdout=bool(getattr(args, "monitor", False)),
        stale_seconds=stale_seconds,
    )
    monitor.start(
        source_path=str(args.source),
        class_name=args.class_name,
        procedural=bool(getattr(args, "procedural", False)),
        llm_provider=llm_provider,
        llm_model=llm_model,
        max_depth=int(config.ingester_max_depth),
    )

    proof_env = None
    try:
        source_path = Path(args.source)
        if not source_path.exists():
            raise FileNotFoundError(f"source file not found: {source_path}")

        # Set up LLM
        from sciona.llm_router import (
            INGESTER_ABSTRACT,
            INGESTER_CHUNK,
            INGESTER_DECOMPOSE,
            INGESTER_FIX_GHOST,
            INGESTER_FIX_TYPE,
            INGESTER_HOIST_STATE,
            INGESTER_OPAQUE_WITNESS,
        )

        prompt_keys = [
            INGESTER_CHUNK,
            INGESTER_HOIST_STATE,
            INGESTER_ABSTRACT,
            INGESTER_FIX_TYPE,
            INGESTER_FIX_GHOST,
            INGESTER_OPAQUE_WITNESS,
            INGESTER_DECOMPOSE,
        ]
        _print_prompt_routing_summary(
            config, "ingester", prompt_keys, getattr(args, "mode", None)
        )
        llm = _create_llm_router(args, config, "ingester", prompt_keys)
        await _warm_llm_if_supported(llm, "ingester")

        # Set up proof environment (Python/mypy)
        proof_env = _create_proof_env(Prover.PYTHON, config)

        # Optionally load FAISS index
        faiss_index = None
        if config.index_dir.exists():
            try:
                faiss_index, _index_mode = _load_semantic_index(
                    config.index_dir,
                    config,
                    backend_override=mode_settings.semantic_index_backend_override,
                )
            except Exception as exc:
                print(f"Warning: failed to load semantic index: {exc}", file=sys.stderr)

        ingester_run_id = uuid.uuid4().hex
        shared_context, shared_context_metrics = await _create_shared_context(
            config,
            enabled=mode_settings.ingester_shared_context_enabled,
        )
        agent = IngesterAgent(
            llm=llm,
            proof_env=proof_env,
            faiss_index=faiss_index,
            output_dir=str(output_dir),
            max_depth=config.ingester_max_depth,
            line_threshold=config.ingester_decompose_line_threshold,
            monitor=monitor,
            shared_context=shared_context,
            shared_context_metrics=shared_context_metrics,
            context_namespace=f"ingester/{ingester_run_id}",
            context_budget_chars=config.ingester_shared_context_budget_chars,
            parallelism=config.ingester_parallelism,
            enable_cache=config.ingester_cache_enabled,
            cache_dir=str(config.ingester_cache_dir),
        )

        print(f"Ingesting {'class' if not getattr(args, 'procedural', False) else 'procedural'} '{args.class_name}' from {source_path}")
        if getattr(args, "procedural", False):
            bundle = await agent.ingest_procedural(str(source_path), args.class_name)
        else:
            bundle = await agent.ingest(
                str(source_path), args.class_name, raise_on_error=True
            )

        # Stage output files and publish atomically on successful completion.
        if bundle.generated_atoms:
            monitor.stage_file("atoms.py", bundle.generated_atoms)
        if bundle.generated_state_models:
            monitor.stage_file("state_models.py", bundle.generated_state_models)
        if bundle.generated_witnesses:
            monitor.stage_file("witnesses.py", bundle.generated_witnesses)
        monitor.stage_json("cdg.json", bundle.cdg.model_dump())

        if bundle.match_results:
            matches_data = [mr.to_dict() for mr in bundle.match_results]
            monitor.stage_json("matches.json", matches_data)

        published_files = monitor.publish_staged()
        summary = {
            "cdg_nodes": len(bundle.cdg.nodes),
            "cdg_edges": len(bundle.cdg.edges),
            "matches": len(bundle.match_results),
            "mypy_passed": bool(bundle.mypy_passed),
            "ghost_sim_passed": bool(bundle.ghost_sim_passed),
            "published_files": published_files,
        }
        monitor.complete(summary=summary)

        print("\nIngestion complete:")
        print(f"  CDG: {len(bundle.cdg.nodes)} nodes, {len(bundle.cdg.edges)} edges")
        print(f"  Matches: {len(bundle.match_results)}")
        print(f"  mypy passed: {bundle.mypy_passed}")
        print(f"  Ghost sim passed: {bundle.ghost_sim_passed}")
        print(f"  Output: {output_dir}/")
        print(f"  Status: {output_dir / '.ingest_status.json'}")
        print(f"  Marker: {output_dir / 'COMPLETED.json'}")
        _print_shared_context_metrics("ingester", shared_context_metrics)
        metrics_path = _write_shared_context_metrics_file(
            output_dir / "shared_context_metrics.json",
            {"ingester": shared_context_metrics},
        )
        if metrics_path is not None:
            print(f"  Shared context metrics: {metrics_path}")

        if getattr(args, "trace", False):
            print(f"  Trace: {output_dir / 'trace.jsonl'}")
    except Exception as exc:
        monitor.fail(error=str(exc))
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if proof_env is not None:
            await proof_env.close()
