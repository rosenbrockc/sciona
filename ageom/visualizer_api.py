"""FastAPI server for browsing CDGs stored in Memgraph."""

from __future__ import annotations

import asyncio
import json
import queue
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.types import Receive, Scope, Send


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Create Memgraph driver on startup, close on shutdown."""
    from ageom.config import AgeomConfig
    from neo4j import AsyncGraphDatabase

    config = AgeomConfig()
    auth = (config.memgraph_user, config.memgraph_password) if config.memgraph_user else None
    driver = AsyncGraphDatabase.driver(config.memgraph_uri, auth=auth)
    app.state.driver = driver

    # Postgres telemetry drain
    telem_drain = None
    telem_store = None
    if config.telemetry_backend != "file" and config.postgres_uri:
        try:
            from ageom.telemetry import configure_postgres_telemetry
            from ageom.telemetry_store import PostgresTelemetryStore, TelemetryDrain

            telem_store = PostgresTelemetryStore(config.postgres_uri)
            await telem_store.setup()
            telem_drain = TelemetryDrain(telem_store)
            configure_postgres_telemetry(telem_store, telem_drain)
            await telem_drain.start()
        except Exception:
            telem_drain = None
            telem_store = None

    yield

    if telem_drain is not None:
        try:
            await telem_drain.stop()
        except Exception:
            pass
    if telem_store is not None:
        try:
            await telem_store.close()
        except Exception:
            pass
    await driver.close()


app = FastAPI(title="AGEO CDG Visualizer", lifespan=_lifespan)


def _merge_runs(
    persisted: list[dict[str, Any]],
    runtime: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in persisted + runtime:
        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            continue
        current = merged.get(run_id)
        if current is None:
            merged[run_id] = row
            continue
        if float(row.get("last_update_at", 0.0)) >= float(
            current.get("last_update_at", 0.0)
        ):
            merged[run_id] = row
    rows = list(merged.values())
    rows.sort(key=lambda r: float(r.get("last_update_at", 0.0)), reverse=True)
    return rows


def _annotate_hang_signals(
    run: dict[str, Any],
    *,
    stale_seconds: int,
    now: float,
) -> dict[str, Any]:
    stages = run.get("stages", {}) if isinstance(run.get("stages"), dict) else {}
    stale_stages: list[dict[str, Any]] = []
    for stage_name, stage in stages.items():
        if not isinstance(stage, dict):
            continue
        if str(stage.get("status", "")) != "running":
            continue
        hb = float(stage.get("last_heartbeat_at") or 0.0)
        if hb <= 0:
            continue
        age = max(0.0, now - hb)
        if age >= stale_seconds:
            stale_stages.append(
                {
                    "stage": stage_name,
                    "heartbeat_age_sec": age,
                    "message": str(stage.get("message", "")),
                }
            )
    out = dict(run)
    out["stale_stages"] = stale_stages
    out["is_hung"] = len(stale_stages) > 0 and str(run.get("status")) == "running"
    return out


def _transport_for_provider(provider: str) -> str:
    """Classify a provider name into its transport category."""
    lowered = provider.strip().lower()
    if lowered.endswith("_shim"):
        return "persistent_shim"
    if lowered.endswith("_cli"):
        return "legacy_cli"
    if lowered == "llama_cpp":
        return "local_server"
    if lowered in {"anthropic", "codex", "openai"}:
        return "api"
    if not lowered:
        return "--"
    return "other"


def _routing_line(routing: dict[str, Any], name: str) -> dict[str, Any]:
    """Build a single routing-round summary line."""
    section = routing.get(name, {})
    if not isinstance(section, dict):
        section = {}
    active = section.get("active_overrides", [])
    if not isinstance(active, list):
        active = []
    return {
        "round": name,
        "default": (
            f"{section.get('default_provider', '--')}:{section.get('default_model', '--')}"
        ),
        "active_count": len(active),
        "suppressed_count": len(section.get("suppressed_default_overrides", []) or []),
        "custom_nonbenchmark_count": len(
            section.get("custom_nonbenchmark_overrides", []) or []
        ),
    }


def _build_retrieval_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    """Build the retrieval_summary section."""
    retrieval = metadata.get("retrieval_policy", {})
    if not isinstance(retrieval, dict):
        retrieval = {}
    return {
        "confidence_band": retrieval.get("confidence_band", "--"),
        "skill_index": bool(retrieval.get("skill_index", False)),
        "graph_retrieval": bool(retrieval.get("graph_retrieval", False)),
        "semantic_backend": retrieval.get("semantic_backend", "default"),
        "hunter_mode": retrieval.get("hunter_mode", "--"),
    }


def _build_execution_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    """Build the execution_summary section."""
    return {
        "mode": str(metadata.get("execution_mode", "") or "--"),
        "path": str(metadata.get("execution_path", "") or "--"),
        "rapid_direct": bool(metadata.get("rapid_direct_path", False)),
    }


def _build_routing_summary(
    routing: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build routing_summary and provider_complexity sections.

    Returns a (routing_summary, provider_complexity) tuple.
    """
    provider_models: set[str] = set()
    providers: set[str] = set()
    transports: set[str] = set()
    for section in routing.values():
        if not isinstance(section, dict):
            continue
        default_provider = str(section.get("default_provider", "") or "").strip()
        default_model = str(section.get("default_model", "") or "").strip()
        if default_provider:
            providers.add(default_provider)
            transports.add(_transport_for_provider(default_provider))
            provider_models.add(f"{default_provider}:{default_model or '--'}")
        active = section.get("active_overrides", [])
        if not isinstance(active, list):
            continue
        for row in active:
            if not isinstance(row, dict):
                continue
            provider = str(row.get("provider", "") or "").strip()
            model = str(row.get("model", "") or "").strip()
            if not provider:
                continue
            providers.add(provider)
            transports.add(_transport_for_provider(provider))
            provider_models.add(f"{provider}:{model or '--'}")

    routing_summary = {
        "architect": _routing_line(routing, "architect"),
        "hunter": _routing_line(routing, "hunter"),
    }
    provider_complexity = {
        "provider_count": len(providers),
        "provider_model_count": len(provider_models),
        "transport_count": len(transports),
        "providers": sorted(providers),
        "provider_models": sorted(provider_models),
        "transports": sorted(transports),
    }
    return routing_summary, provider_complexity


def _build_catalog_alignment_summary(
    catalog_alignment: dict[str, Any],
) -> dict[str, Any]:
    """Build the catalog_alignment_summary section."""
    summary: dict[str, Any] = {
        "catalog_size": int(catalog_alignment.get("catalog_size", 0) or 0),
        "total_candidates": int(catalog_alignment.get("total_candidates", 0) or 0),
        "added": int(catalog_alignment.get("added", 0) or 0),
        "merged": int(catalog_alignment.get("merged", 0) or 0),
        "structural_skips": int(catalog_alignment.get("structural_skips", 0) or 0),
        "live_registry": int(
            catalog_alignment.get("source_live_registry_candidates", 0) or 0
        ),
        "ast_fallback": int(catalog_alignment.get("source_ast_candidates", 0) or 0),
        "cdg_matched": int(
            catalog_alignment.get("source_cdg_metadata_matches", 0) or 0
        ),
        "witness_doc": int(
            catalog_alignment.get("source_witness_doc_fallbacks", 0) or 0
        ),
        "witness_signature": int(
            catalog_alignment.get("source_witness_signature_fallbacks", 0) or 0
        ),
    }
    merge_details = catalog_alignment.get("merge_details", {})
    if not isinstance(merge_details, list):
        merge_details = []
    merge_rows: list[dict[str, Any]] = []
    for row in merge_details:
        if not isinstance(row, dict):
            continue
        merge_rows.append(
            {
                "candidate": str(row.get("candidate", "") or ""),
                "incumbent": str(row.get("incumbent", "") or ""),
                "similarity": float(row.get("similarity", 0.0) or 0.0),
            }
        )
    merge_rows.sort(
        key=lambda row: (-row["similarity"], row["candidate"], row["incumbent"])
    )
    summary["top_merges"] = merge_rows[:5]
    source_breakdown = catalog_alignment.get("source_breakdown", {})
    if not isinstance(source_breakdown, dict):
        source_breakdown = {}
    source_rows: list[dict[str, Any]] = []
    for source_name, row in source_breakdown.items():
        if not isinstance(row, dict):
            continue
        source_rows.append(
            {
                "source": str(source_name),
                "added": int(row.get("added", 0) or 0),
                "live_registry_candidates": int(
                    row.get("live_registry_candidates", 0) or 0
                ),
                "ast_candidates": int(row.get("ast_candidates", 0) or 0),
            }
        )
    source_rows.sort(
        key=lambda row: (
            -int(row["added"]),
            -(int(row["live_registry_candidates"]) + int(row["ast_candidates"])),
            row["source"],
        )
    )
    summary["source_count"] = len(source_rows)
    summary["top_sources"] = source_rows[:5]
    return summary


def _build_architect_summary(architect_metrics: dict[str, Any]) -> dict[str, Any]:
    """Build the architect_summary section."""
    status_counts = architect_metrics.get("node_status_counts", {})
    if not isinstance(status_counts, dict):
        status_counts = {}
    blocked_names = architect_metrics.get("blocked_node_names", [])
    if not isinstance(blocked_names, list):
        blocked_names = []
    critique_counts = architect_metrics.get("critique_reject_counts_by_category", {})
    if not isinstance(critique_counts, dict):
        critique_counts = {}
    retry_counts = architect_metrics.get("retry_counts_by_node", {})
    if not isinstance(retry_counts, dict):
        retry_counts = {}
    rewrite_actions = architect_metrics.get("rewrite_actions", [])
    if not isinstance(rewrite_actions, list):
        rewrite_actions = []
    top_critique_categories = sorted(
        (
            {
                "category": str(category),
                "count": int(count or 0),
            }
            for category, count in critique_counts.items()
        ),
        key=lambda row: (-row["count"], row["category"]),
    )
    return {
        "unresolved_leaf_count": int(architect_metrics.get("unresolved_leaf_count", 0) or 0),
        "blocked_count": int(status_counts.get("blocked", 0) or 0),
        "blocked_node_names": [str(name) for name in blocked_names if str(name).strip()],
        "blocked_reason": str(architect_metrics.get("blocked_reason", "") or ""),
        "any_port_pct": float(architect_metrics.get("any_port_pct", 0.0) or 0.0),
        "any_edge_pct": float(architect_metrics.get("any_edge_pct", 0.0) or 0.0),
        "rewrite_action_count": len(rewrite_actions),
        "last_node_name": str(architect_metrics.get("last_node_name", "") or ""),
        "critique_reject_total": sum(int(count or 0) for count in critique_counts.values()),
        "critique_reject_categories": top_critique_categories[:5],
        "retry_total": sum(int(count or 0) for count in retry_counts.values()),
        "retry_node_count": len(retry_counts),
    }


def _build_hunter_summary(hunter_metrics: dict[str, Any]) -> dict[str, Any]:
    """Build the hunter_summary section."""
    return {
        "search_iterations": int(hunter_metrics.get("search_iterations", 0) or 0),
        "embedding_results_total": int(
            hunter_metrics.get("embedding_results_total", 0) or 0
        ),
        "type_results_total": int(hunter_metrics.get("type_results_total", 0) or 0),
        "new_candidates_total": int(
            hunter_metrics.get("new_candidates_total", 0) or 0
        ),
        "candidate_pool_size": int(hunter_metrics.get("candidate_pool_size", 0) or 0),
        "rank_calls": int(hunter_metrics.get("rank_calls", 0) or 0),
        "ranked_candidate_count": int(
            hunter_metrics.get("ranked_candidate_count", 0) or 0
        ),
        "verify_batches": int(hunter_metrics.get("verify_batches", 0) or 0),
        "verified_candidates_total": int(
            hunter_metrics.get("verified_candidates_total", 0) or 0
        ),
        "verification_success_total": int(
            hunter_metrics.get("verification_success_total", 0) or 0
        ),
        "verification_failure_total": int(
            hunter_metrics.get("verification_failure_total", 0) or 0
        ),
        "verified_matches": int(hunter_metrics.get("verified_matches", 0) or 0),
        "reformulations": int(hunter_metrics.get("reformulations", 0) or 0),
        "failure_analyses": int(hunter_metrics.get("failure_analyses", 0) or 0),
        "reformulate_fallbacks": int(
            hunter_metrics.get("reformulate_fallbacks", 0) or 0
        ),
        "empty_search_terminations": int(
            hunter_metrics.get("empty_search_terminations", 0) or 0
        ),
        "empty_verify_batches": int(
            hunter_metrics.get("empty_verify_batches", 0) or 0
        ),
        "query_count": int(hunter_metrics.get("query_count", 0) or 0),
        "iteration": int(hunter_metrics.get("iteration", 0) or 0),
        "last_query": str(hunter_metrics.get("last_query", "") or ""),
        "last_verified_candidate": str(
            hunter_metrics.get("last_verified_candidate", "") or ""
        ),
    }


def _build_single_agent_summary(single_agent: dict[str, Any]) -> dict[str, Any]:
    """Build the single_agent_summary section."""
    if not single_agent:
        return {}
    policy = single_agent.get("policy", {})
    if not isinstance(policy, dict):
        policy = {}
    concrete_artifacts = single_agent.get("concrete_artifacts", {})
    if not isinstance(concrete_artifacts, dict):
        concrete_artifacts = {}
    tool_metrics = single_agent.get("tool_metrics", {})
    if not isinstance(tool_metrics, dict):
        tool_metrics = {}
    escalation_events = single_agent.get("escalation_events", {})
    if not isinstance(escalation_events, list):
        escalation_events = []
    artifacts: list[dict[str, Any]] = []
    for name in sorted(concrete_artifacts):
        row = concrete_artifacts.get(name, {})
        if not isinstance(row, dict):
            continue
        artifacts.append(
            {
                "name": name,
                "source": str(row.get("source", "") or ""),
                "path": str(row.get("path", "") or ""),
                "exists": bool(row.get("exists", False)),
                "mutations": int(row.get("mutations", 0) or 0),
            }
        )
    tool_rows: list[dict[str, Any]] = []
    for name in sorted(tool_metrics):
        row = tool_metrics.get(name, {})
        if not isinstance(row, dict):
            continue
        tool_rows.append(
            {
                "name": name,
                "dispatches": int(row.get("dispatches", 0) or 0),
                "latency_ms_total": round(
                    float(row.get("latency_ms_total", 0.0) or 0.0), 4
                ),
                "avg_latency_ms": round(float(row.get("avg_latency_ms", 0.0) or 0.0), 4),
            }
        )
    return {
        "termination_reason": str(single_agent.get("termination_reason", "") or ""),
        "verification_status": str(single_agent.get("verification_status", "") or ""),
        "steps_used": int(single_agent.get("steps_used", 0) or 0),
        "step_budget": int(single_agent.get("step_budget", 0) or 0),
        "open_failures": list(single_agent.get("open_failures", []) or []),
        "attempt_history": list(single_agent.get("attempt_history", []) or []),
        "artifact_manifest_path": str(
            single_agent.get("artifact_manifest_path", "") or ""
        ),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "tool_dispatch_count_total": int(
            single_agent.get("tool_dispatch_count_total", 0) or 0
        ),
        "tool_latency_ms_total": round(
            float(single_agent.get("tool_latency_ms_total", 0.0) or 0.0), 4
        ),
        "tool_metrics": tool_rows,
        "escalation_events": [
            {
                "from": str(row.get("from", "") or ""),
                "to": str(row.get("to", "") or ""),
                "reason": str(row.get("reason", "") or ""),
            }
            for row in escalation_events
            if isinstance(row, dict)
        ],
        "policy": {
            "direct_grounding_enabled": bool(
                policy.get("direct_grounding_enabled", False)
            ),
            "decomposition_mode": str(policy.get("decomposition_mode", "") or ""),
            "retrieval_intensity": str(policy.get("retrieval_intensity", "") or ""),
            "repair_policy": str(policy.get("repair_policy", "") or ""),
        },
    }


def _build_benchmark_summary(
    benchmark: dict[str, Any],
    release_validation: dict[str, Any],
) -> dict[str, Any]:
    """Build the benchmark_summary section (includes release_validation fields)."""
    return {
        "status": str(benchmark.get("status", "") or ""),
        "prompt_cases": int(benchmark.get("prompt_cases", 0) or 0),
        "prompt_results": int(benchmark.get("prompt_results", 0) or 0),
        "flow_cases": int(benchmark.get("flow_cases", 0) or 0),
        "flow_results": int(benchmark.get("flow_results", 0) or 0),
        "prompt_summary": str(benchmark.get("prompt_summary", "") or ""),
        "prompt_stability_summary": str(
            benchmark.get("prompt_stability_summary", "") or ""
        ),
        "flow_summary": str(benchmark.get("flow_summary", "") or ""),
        "flow_stability_summary": str(
            benchmark.get("flow_stability_summary", "") or ""
        ),
        "flow_gate_summary": str(benchmark.get("flow_gate_summary", "") or ""),
        "flow_execution_path_summary": str(
            benchmark.get("flow_execution_path_summary", "") or ""
        ),
        "runtime_override_policy_summary": str(
            benchmark.get("runtime_override_policy_summary", "") or ""
        ),
        "health_summary": str(benchmark.get("health_summary", "") or ""),
        "warning_summary": str(benchmark.get("warning_summary", "") or ""),
        "top_warning_subcheck": str(
            benchmark.get("top_warning_subcheck", "") or ""
        ),
        "top_warning": str(benchmark.get("top_warning", "") or ""),
        "failure_summary": str(benchmark.get("failure_summary", "") or ""),
        "top_failed_subcheck": str(
            benchmark.get("top_failed_subcheck", "") or ""
        ),
        "top_failure": str(benchmark.get("top_failure", "") or ""),
        "flow_required_variants": list(
            benchmark.get("flow_required_variants", []) or []
        )
        if isinstance(benchmark.get("flow_required_variants", []), list)
        else [],
        "flow_comparison_variants": list(
            benchmark.get("flow_comparison_variants", []) or []
        )
        if isinstance(benchmark.get("flow_comparison_variants", []), list)
        else [],
        "flow_execution_paths": dict(
            benchmark.get("flow_execution_paths", {}) or {}
        )
        if isinstance(benchmark.get("flow_execution_paths", {}), dict)
        else {},
        "flow_prompt_volume": dict(
            benchmark.get("flow_prompt_volume", {}) or {}
        )
        if isinstance(benchmark.get("flow_prompt_volume", {}), dict)
        else {},
        "flow_prompt_volume_summary": str(
            benchmark.get("flow_prompt_volume_summary", "") or ""
        ),
        "single_agent_comparison": dict(
            benchmark.get("single_agent_comparison", {}) or {}
        )
        if isinstance(benchmark.get("single_agent_comparison", {}), dict)
        else {},
        "single_agent_comparison_summary": str(
            benchmark.get("single_agent_comparison_summary", "") or ""
        ),
        "flow_avg_prompt_calls": dict(
            benchmark.get("flow_avg_prompt_calls", {}) or {}
        )
        if isinstance(benchmark.get("flow_avg_prompt_calls", {}), dict)
        else {},
        "flow_avg_planner_tool_dispatches": dict(
            benchmark.get("flow_avg_planner_tool_dispatches", {}) or {}
        )
        if isinstance(benchmark.get("flow_avg_planner_tool_dispatches", {}), dict)
        else {},
        "flow_avg_planner_tool_latency_ms": dict(
            benchmark.get("flow_avg_planner_tool_latency_ms", {}) or {}
        )
        if isinstance(benchmark.get("flow_avg_planner_tool_latency_ms", {}), dict)
        else {},
        "flow_avg_planner_escalations": dict(
            benchmark.get("flow_avg_planner_escalations", {}) or {}
        )
        if isinstance(benchmark.get("flow_avg_planner_escalations", {}), dict)
        else {},
        "prompt_avg_latency_ms": dict(
            benchmark.get("prompt_avg_latency_ms", {}) or {}
        )
        if isinstance(benchmark.get("prompt_avg_latency_ms", {}), dict)
        else {},
        "flow_avg_latency_ms": dict(
            benchmark.get("flow_avg_latency_ms", {}) or {}
        )
        if isinstance(benchmark.get("flow_avg_latency_ms", {}), dict)
        else {},
        "summary_report": str(benchmark.get("summary_report", "") or ""),
        "prompt_tuned_failures": int(benchmark.get("prompt_tuned_failures", 0) or 0),
        "prompt_tuned_unstable_groups": int(
            benchmark.get("prompt_tuned_unstable_groups", 0) or 0
        ),
        "flow_mode_failures": int(benchmark.get("flow_mode_failures", 0) or 0),
        "flow_mode_unstable_groups": int(
            benchmark.get("flow_mode_unstable_groups", 0) or 0
        ),
        "flow_comparison_failures": int(
            benchmark.get("flow_comparison_failures", 0) or 0
        ),
        "flow_comparison_unstable_groups": int(
            benchmark.get("flow_comparison_unstable_groups", 0) or 0
        ),
        "runtime_complexity": dict(
            benchmark.get("runtime_complexity", {}) or {}
        )
        if isinstance(benchmark.get("runtime_complexity", {}), dict)
        else {},
        "release_warning_summary": str(
            release_validation.get("warning_summary", "") or ""
        ),
        "release_runtime_warning_count": int(
            release_validation.get("runtime_warning_count", 0) or 0
        ),
        "release_catalog_warning_count": int(
            release_validation.get("catalog_warning_count", 0) or 0
        ),
        "release_benchmark_warning_count": int(
            release_validation.get("benchmark_warning_count", 0) or 0
        ),
        "release_top_runtime_warning": str(
            release_validation.get("top_runtime_warning", "") or ""
        ),
        "release_top_catalog_warning": str(
            release_validation.get("top_catalog_warning", "") or ""
        ),
        "release_top_benchmark_warning": str(
            release_validation.get("top_benchmark_warning", "") or ""
        ),
        "release_top_benchmark_warning_subcheck": str(
            release_validation.get("top_benchmark_warning_subcheck", "") or ""
        ),
        "release_failure_summary": str(
            release_validation.get("failure_summary", "") or ""
        ),
        "release_top_failed_check": str(
            release_validation.get("top_failed_check", "") or ""
        ),
        "release_top_benchmark_subcheck": str(
            release_validation.get("top_benchmark_subcheck", "") or ""
        ),
        "release_top_benchmark_failure": str(
            release_validation.get("top_benchmark_failure", "") or ""
        ),
        "release_top_runtime_failure": str(
            release_validation.get("top_runtime_failure", "") or ""
        ),
        "release_top_catalog_failure": str(
            release_validation.get("top_catalog_failure", "") or ""
        ),
        "manifest": str(release_validation.get("manifest", "") or ""),
        "benchmarks_dir": str(release_validation.get("benchmarks_dir", "") or ""),
        "release_status": str(release_validation.get("status", "") or ""),
    }


def _build_catalog_validation_summary(
    release_validation: dict[str, Any],
) -> dict[str, Any]:
    """Build the catalog_validation_summary section."""
    catalog_validation = release_validation.get("catalog_validation", {})
    if not isinstance(catalog_validation, dict):
        catalog_validation = {}
    summary: dict[str, Any] = {
        "status": str(catalog_validation.get("status", "") or ""),
        "configured_sources": int(catalog_validation.get("configured_sources", 0) or 0),
        "resolved_sources": int(catalog_validation.get("resolved_sources", 0) or 0),
        "source_candidates": int(catalog_validation.get("source_candidates", 0) or 0),
        "source_added": int(catalog_validation.get("source_added", 0) or 0),
        "coverage_summary": str(catalog_validation.get("coverage_summary", "") or ""),
        "alignment_summary": str(catalog_validation.get("alignment_summary", "") or ""),
        "warning_summary": str(catalog_validation.get("warning_summary", "") or ""),
        "high_severity_sources": list(catalog_validation.get("high_severity_sources", []) or [])
        if isinstance(catalog_validation.get("high_severity_sources", []), list)
        else [],
        "medium_severity_sources": list(catalog_validation.get("medium_severity_sources", []) or [])
        if isinstance(catalog_validation.get("medium_severity_sources", []), list)
        else [],
        "missing_sources": list(catalog_validation.get("missing_sources", []) or [])
        if isinstance(catalog_validation.get("missing_sources", []), list)
        else [],
        "zero_candidate_sources": list(
            catalog_validation.get("zero_candidate_sources", []) or []
        )
        if isinstance(catalog_validation.get("zero_candidate_sources", []), list)
        else [],
        "violations": list(catalog_validation.get("violations", []) or [])
        if isinstance(catalog_validation.get("violations", []), list)
        else [],
        "warnings": list(catalog_validation.get("warnings", []) or [])
        if isinstance(catalog_validation.get("warnings", []), list)
        else [],
        "report": str(catalog_validation.get("report", "") or ""),
    }
    alignment = catalog_validation.get("alignment", {})
    if not isinstance(alignment, dict):
        alignment = {}
    summary["alignment"] = {
        "source_count": int(alignment.get("source_count", 0) or 0),
        "matched_total": int(alignment.get("matched_total", 0) or 0),
        "registry_only_total": int(alignment.get("registry_only_total", 0) or 0),
        "ast_only_total": int(alignment.get("ast_only_total", 0) or 0),
        "highest_severity": str(alignment.get("highest_severity", "") or ""),
        "severity_counts": dict(alignment.get("severity_counts", {}) or {})
        if isinstance(alignment.get("severity_counts", {}), dict)
        else {},
        "drift_sources": list(alignment.get("drift_sources", []) or [])
        if isinstance(alignment.get("drift_sources", []), list)
        else [],
        "registry_error_sources": list(alignment.get("registry_error_sources", []) or [])
        if isinstance(alignment.get("registry_error_sources", []), list)
        else [],
    }
    alignment_rows = alignment.get("rows", {})
    if not isinstance(alignment_rows, list):
        alignment_rows = []
    top_drift_rows: list[dict[str, Any]] = []
    for row in alignment_rows:
        if not isinstance(row, dict):
            continue
        top_drift_rows.append(
            {
                "source": str(row.get("source", "") or ""),
                "registry_only_count": int(row.get("registry_only_count", 0) or 0),
                "ast_only_count": int(row.get("ast_only_count", 0) or 0),
                "severity": str(row.get("severity", "") or ""),
                "registry_only_examples": list(row.get("registry_only_examples", []) or [])
                if isinstance(row.get("registry_only_examples", []), list)
                else [],
                "ast_only_examples": list(row.get("ast_only_examples", []) or [])
                if isinstance(row.get("ast_only_examples", []), list)
                else [],
            }
        )
    top_drift_rows.sort(
        key=lambda row: (
            -(int(row["registry_only_count"]) + int(row["ast_only_count"])),
            row["source"],
        )
    )
    summary["top_drift_sources"] = top_drift_rows[:5]
    return summary


def _build_shared_context_summary(shared_context: dict[str, Any]) -> dict[str, Any]:
    """Build the shared_context_summary section."""
    contexts = shared_context.get("contexts", {})
    if not isinstance(contexts, dict):
        contexts = {}
    shared_rows: list[dict[str, Any]] = []
    total_searches = 0
    total_hits = 0
    total_puts = 0
    total_injected_blocks = 0
    total_promotions = 0
    total_template_searches = 0
    total_template_hits = 0
    total_template_puts = 0
    total_template_injected = 0
    total_failure_searches = 0
    total_failure_hits = 0
    total_failure_puts = 0
    total_failure_injected = 0
    backends: set[str] = set()
    active_contexts = 0
    for label, row in contexts.items():
        if not isinstance(row, dict):
            continue
        searches = int(row.get("searches_total", 0) or 0)
        hits = int(row.get("search_hits", 0) or 0)
        puts = int(row.get("puts_total", 0) or 0)
        injected_blocks = int(row.get("injected_blocks", 0) or 0)
        promotions = int(row.get("promotions_total", 0) or 0)
        template_searches = int(row.get("template_searches_total", 0) or 0)
        template_hits = int(row.get("template_search_hits", 0) or 0)
        template_puts = int(row.get("template_puts_total", 0) or 0)
        template_injected = int(row.get("template_injected_blocks", 0) or 0)
        failure_searches = int(row.get("failure_searches_total", 0) or 0)
        failure_hits = int(row.get("failure_search_hits", 0) or 0)
        failure_puts = int(row.get("failure_puts_total", 0) or 0)
        failure_injected = int(row.get("failure_injected_blocks", 0) or 0)
        backend = str(row.get("backend", "") or "").strip()
        if backend:
            backends.add(backend)
        if any(
            value > 0
            for value in (
                searches,
                puts,
                injected_blocks,
                promotions,
                template_searches,
                template_puts,
                template_injected,
                failure_searches,
                failure_puts,
                failure_injected,
            )
        ):
            active_contexts += 1
        total_searches += searches
        total_hits += hits
        total_puts += puts
        total_injected_blocks += injected_blocks
        total_promotions += promotions
        total_template_searches += template_searches
        total_template_hits += template_hits
        total_template_puts += template_puts
        total_template_injected += template_injected
        total_failure_searches += failure_searches
        total_failure_hits += failure_hits
        total_failure_puts += failure_puts
        total_failure_injected += failure_injected
        shared_rows.append(
            {
                "label": label,
                "backend": backend or "--",
                "searches": searches,
                "hits": hits,
                "puts": puts,
                "injected_blocks": injected_blocks,
                "promotions": promotions,
                "template_searches": template_searches,
                "template_hits": template_hits,
                "template_puts": template_puts,
                "template_injected_blocks": template_injected,
                "failure_searches": failure_searches,
                "failure_hits": failure_hits,
                "failure_puts": failure_puts,
                "failure_injected_blocks": failure_injected,
            }
        )
    return {
        "context_count": len(shared_rows),
        "active_context_count": active_contexts,
        "backends": sorted(backends),
        "total_searches": total_searches,
        "total_hits": total_hits,
        "total_puts": total_puts,
        "total_injected_blocks": total_injected_blocks,
        "total_promotions": total_promotions,
        "total_template_searches": total_template_searches,
        "total_template_hits": total_template_hits,
        "total_template_puts": total_template_puts,
        "total_template_injected_blocks": total_template_injected,
        "total_failure_searches": total_failure_searches,
        "total_failure_hits": total_failure_hits,
        "total_failure_puts": total_failure_puts,
        "total_failure_injected_blocks": total_failure_injected,
        "metrics_path": str(shared_context.get("metrics_path", "") or ""),
        "contexts": sorted(shared_rows, key=lambda row: row["label"]),
    }


def _build_optimize_summary(optimize: dict[str, Any]) -> dict[str, Any]:
    """Build the optimize_summary section."""
    trial_rows = optimize.get("trial_rows", [])
    if not isinstance(trial_rows, list):
        trial_rows = []
    rows: list[dict[str, Any]] = []
    for row in trial_rows:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "trial": int(row.get("trial", 0) or 0),
                "loss": float(row.get("loss", 0.0) or 0.0),
                "node_count": int(row.get("node_count", 0) or 0),
                "edge_count": int(row.get("edge_count", 0) or 0),
                "primitive_signature": str(row.get("primitive_signature", "") or ""),
                "has_parameters": bool(row.get("has_parameters")),
                "parameter_node_count": int(row.get("parameter_node_count", 0) or 0),
                "topology_changed": bool(row.get("topology_changed")),
                "primitive_assignment_changed": bool(
                    row.get("primitive_assignment_changed")
                ),
            }
        )
    best_structure = optimize.get("best_structure", {})
    if not isinstance(best_structure, dict):
        best_structure = {}
    best_params = optimize.get("best_parameter_assignments", {})
    if not isinstance(best_params, dict):
        best_params = {}
    return {
        "objective": str(optimize.get("objective", "") or ""),
        "execution_metric": str(optimize.get("execution_metric", "") or ""),
        "benchmark_path": str(optimize.get("benchmark_path", "") or ""),
        "max_trials": int(optimize.get("max_trials", 0) or 0),
        "trials_run": int(optimize.get("trials_run", len(rows)) or 0),
        "best_loss": (
            float(optimize.get("best_loss"))
            if optimize.get("best_loss") is not None
            else None
        ),
        "best_trial": int(optimize.get("best_trial", 0) or 0),
        "parameterized_trials": int(optimize.get("parameterized_trials", 0) or 0),
        "primitive_change_trials": int(optimize.get("primitive_change_trials", 0) or 0),
        "topology_change_trials": int(optimize.get("topology_change_trials", 0) or 0),
        "unique_primitive_signatures": int(
            optimize.get("unique_primitive_signatures", 0) or 0
        ),
        "unique_topologies": int(optimize.get("unique_topologies", 0) or 0),
        "best_structure": {
            "node_count": int(best_structure.get("node_count", 0) or 0),
            "edge_count": int(best_structure.get("edge_count", 0) or 0),
            "topo_hash": str(best_structure.get("topo_hash", "") or ""),
            "primitive_signature": str(
                best_structure.get("primitive_signature", "") or ""
            ),
        },
        "best_parameter_assignments": best_params,
        "trial_history_path": str(optimize.get("trial_history_path", "") or ""),
        "trial_rows": rows,
    }


def _extract_dashboard_summaries(run: dict[str, Any]) -> dict[str, Any]:
    """Derive dashboard-friendly summaries from run metadata."""
    metadata = run.get("metadata", {}) if isinstance(run.get("metadata"), dict) else {}
    routing = metadata.get("llm_routing", {})
    benchmark = metadata.get("benchmark_validation", {})
    release_validation = metadata.get("release_validation", {})
    shared_context = metadata.get("shared_context", {})
    optimize = metadata.get("optimize", {})
    single_agent = metadata.get("single_agent", {})
    catalog_alignment = metadata.get("catalog_alignment", {})
    architect_metrics = metadata.get("architect_metrics", {})
    hunter_metrics = metadata.get("hunter_metrics", {})
    if not isinstance(routing, dict):
        routing = {}
    if not isinstance(benchmark, dict):
        benchmark = {}
    if not isinstance(release_validation, dict):
        release_validation = {}
    if not isinstance(shared_context, dict):
        shared_context = {}
    if not isinstance(optimize, dict):
        optimize = {}
    if not isinstance(single_agent, dict):
        single_agent = {}
    if not isinstance(catalog_alignment, dict):
        catalog_alignment = {}
    if not isinstance(architect_metrics, dict):
        architect_metrics = {}
    if not isinstance(hunter_metrics, dict):
        hunter_metrics = {}

    routing_summary, provider_complexity = _build_routing_summary(routing)

    out = dict(run)
    out["retrieval_summary"] = _build_retrieval_summary(metadata)
    out["execution_summary"] = _build_execution_summary(metadata)
    out["routing_summary"] = routing_summary
    out["provider_complexity"] = provider_complexity
    out["catalog_alignment_summary"] = _build_catalog_alignment_summary(catalog_alignment)
    out["architect_summary"] = _build_architect_summary(architect_metrics)
    out["hunter_summary"] = _build_hunter_summary(hunter_metrics)
    out["single_agent_summary"] = _build_single_agent_summary(single_agent)
    out["benchmark_summary"] = _build_benchmark_summary(benchmark, release_validation)
    out["catalog_validation_summary"] = _build_catalog_validation_summary(release_validation)
    out["shared_context_summary"] = _build_shared_context_summary(shared_context)
    out["optimize_summary"] = _build_optimize_summary(optimize)
    return out


def _decorate_dashboard_run(
    run: dict[str, Any],
    *,
    stale_seconds: int,
    now: float,
) -> dict[str, Any]:
    """Apply all dashboard-facing derived annotations."""
    return _extract_dashboard_summaries(
        _annotate_hang_signals(run, stale_seconds=stale_seconds, now=now)
    )


@app.get("/api/dashboard/runs")
async def list_dashboard_runs(
    limit: int = Query(50, ge=1, le=500),
    state: str = Query(
        "all", description="Filter by state: all|running|completed|failed"
    ),
) -> dict[str, Any]:
    """List telemetry runs with stale/hang annotations."""
    from ageom.config import AgeomConfig
    from ageom.telemetry import list_runtime_runs, load_persisted_runs, load_runs_from_store

    config = AgeomConfig()
    wanted = state.strip().lower()
    status_filter = wanted if wanted != "all" else None

    # Try Postgres first, merge with runtime, fall back to file
    pg_runs = await load_runs_from_store(limit=max(limit * 3, 100), status=status_filter)
    file_runs = load_persisted_runs(config.telemetry_runs_dir, limit=max(limit * 3, 100)) if pg_runs is None else []
    rows = _merge_runs(
        pg_runs or file_runs,
        list_runtime_runs(),
    )
    if wanted != "all" and pg_runs is None:
        rows = [r for r in rows if str(r.get("status", "")).lower() == wanted]

    now = time.time()
    rows = [
        _decorate_dashboard_run(
            r, stale_seconds=max(5, int(config.telemetry_stale_seconds)), now=now
        )
        for r in rows[:limit]
    ]
    return {"runs": rows, "count": len(rows)}


@app.get("/api/dashboard/runs/{run_id}")
async def get_dashboard_run(run_id: str) -> dict[str, Any]:
    """Get one telemetry run snapshot."""
    from ageom.config import AgeomConfig
    from ageom.telemetry import get_persisted_run, get_runtime_run, load_run_from_store

    config = AgeomConfig()
    row = get_runtime_run(run_id)
    if row is None:
        row = await load_run_from_store(run_id)
    if row is None:
        row = get_persisted_run(config.telemetry_runs_dir, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return _decorate_dashboard_run(
        row,
        stale_seconds=max(5, int(config.telemetry_stale_seconds)),
        now=time.time(),
    )


@app.get("/api/dashboard/latest")
async def get_latest_dashboard_run() -> dict[str, Any]:
    """Get the most recently-updated telemetry run snapshot."""
    payload = await list_dashboard_runs(limit=1, state="all")
    runs = payload.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise HTTPException(status_code=404, detail="No telemetry runs found")
    latest = runs[0]
    if not isinstance(latest, dict):
        raise HTTPException(status_code=404, detail="No telemetry runs found")
    return latest


@app.get("/api/dashboard/runs/{run_id}/events")
async def list_run_events(
    run_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=2000),
    phase: str | None = Query(None),
    event_type: str | None = Query(None),
    prompt_key: str | None = Query(None),
    round_name: str | None = Query(None, alias="round"),
    has_error: bool | None = Query(None),
) -> dict[str, Any]:
    """List events for a telemetry run with optional filters."""
    from ageom.telemetry import get_event_log, load_events_from_store

    # Try Postgres first
    pg_result = await load_events_from_store(
        run_id,
        offset=offset,
        limit=limit,
        phase=phase,
        event_type=event_type,
        prompt_key=prompt_key,
        round_name=round_name,
        has_error=has_error,
    )
    if pg_result is not None:
        events, total = pg_result
        return {"run_id": run_id, "total": total, "offset": offset, "limit": limit, "events": events}

    # Fallback to in-memory
    events = get_event_log().events_for_run(run_id)
    filtered: list[dict[str, Any]] = []
    for ev in events:
        if phase and ev.phase != phase:
            continue
        if event_type and ev.event_type != event_type:
            continue
        if prompt_key and ev.prompt_key != prompt_key:
            continue
        if round_name and ev.round != round_name:
            continue
        d = asdict(ev)
        if has_error is True:
            is_error = (
                "ERROR" in ev.event_type
                or "FAIL" in ev.event_type
                or "error" in ev.payload
            )
            if not is_error:
                continue
        if has_error is False:
            is_error = (
                "ERROR" in ev.event_type
                or "FAIL" in ev.event_type
                or "error" in ev.payload
            )
            if is_error:
                continue
        filtered.append(d)

    total = len(filtered)
    page = filtered[offset : offset + limit]
    return {"run_id": run_id, "total": total, "offset": offset, "limit": limit, "events": page}


@app.get("/api/dashboard/runs/{run_id}/stream")
async def stream_run_events(run_id: str) -> EventSourceResponse:
    """SSE endpoint that streams live events for a run."""
    from ageom.telemetry import get_runtime_run, subscribe_events, unsubscribe_events

    sub_id, q = subscribe_events(run_id)

    async def event_generator():
        keepalive_counter = 0
        try:
            while True:
                try:
                    event_dict = await asyncio.to_thread(q.get, True, 1.0)
                    yield {
                        "event": event_dict.get("event_type", "event"),
                        "data": json.dumps(event_dict, default=str),
                    }
                    keepalive_counter = 0
                except queue.Empty:
                    keepalive_counter += 1
                    # Send keepalive every ~15 seconds
                    if keepalive_counter >= 15:
                        yield {"comment": "keepalive"}
                        keepalive_counter = 0
                    # Check if run is done
                    run = get_runtime_run(run_id)
                    if run and run.get("status") in ("completed", "failed"):
                        # Drain remaining events
                        while True:
                            try:
                                event_dict = q.get(block=False)
                                yield {
                                    "event": event_dict.get("event_type", "event"),
                                    "data": json.dumps(event_dict, default=str),
                                }
                            except queue.Empty:
                                break
                        yield {"event": "done", "data": json.dumps({"status": run["status"]})}
                        return
        finally:
            unsubscribe_events(sub_id)

    return EventSourceResponse(event_generator())


def _compute_coverage_from_dicts(
    run_id: str, event_dicts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build coverage response from event dicts (Postgres or in-memory)."""
    keys: dict[str, dict[str, Any]] = {}
    for ev in event_dicts:
        et = ev.get("event_type", "")
        if et not in ("PROMPT_DISPATCH_DONE", "PROMPT_DISPATCH_ERROR"):
            continue
        pk = ev.get("prompt_key") or "(unknown)"
        row = keys.get(pk)
        if row is None:
            row = {
                "prompt_key": pk,
                "total_dispatches": 0,
                "deterministic_count": 0,
                "llm_fallback_count": 0,
                "providers": set(),
                "error_count": 0,
                "latency_total_ms": 0.0,
            }
            keys[pk] = row
        row["total_dispatches"] += 1
        provider = ev.get("provider") or ""
        if provider:
            row["providers"].add(provider)
        payload = ev.get("payload") or {}
        is_deterministic = (
            provider == "deterministic"
            or "_shim" in provider
            or provider.endswith("_cli")
            or payload.get("critique_source") == "deterministic"
            or payload.get("ghost_fix_source") == "deterministic"
            or payload.get("state_hoist_source") == "deterministic"
            or payload.get("tactic_source") == "deterministic"
        )
        if is_deterministic:
            row["deterministic_count"] += 1
        else:
            row["llm_fallback_count"] += 1
        if et == "PROMPT_DISPATCH_ERROR":
            row["error_count"] += 1
        dur = ev.get("duration_ms")
        if dur and dur > 0:
            row["latency_total_ms"] += dur

    prompt_keys = []
    total_dispatches = 0
    total_deterministic = 0
    for row in sorted(keys.values(), key=lambda r: r["total_dispatches"], reverse=True):
        total = row["total_dispatches"]
        det = row["deterministic_count"]
        total_dispatches += total
        total_deterministic += det
        prompt_keys.append({
            "prompt_key": row["prompt_key"],
            "total_dispatches": total,
            "deterministic_count": det,
            "llm_fallback_count": row["llm_fallback_count"],
            "deterministic_pct": round(det / total * 100, 1) if total else 0.0,
            "providers": sorted(row["providers"]),
            "avg_latency_ms": round(row["latency_total_ms"] / total, 1) if total else 0.0,
            "error_count": row["error_count"],
        })

    return {
        "run_id": run_id,
        "prompt_keys": prompt_keys,
        "overall_deterministic_pct": round(
            total_deterministic / total_dispatches * 100, 1
        )
        if total_dispatches
        else 0.0,
        "total_dispatches": total_dispatches,
    }


def _compute_errors_from_dicts(
    run_id: str, event_dicts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build error drilldown response from event dicts (Postgres or in-memory)."""
    dispatch_groups: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, Any]] = []

    for ev in event_dicts:
        et = ev.get("event_type", "")
        if et in ("PROMPT_DISPATCH_DONE", "PROMPT_DISPATCH_ERROR"):
            group_key = f"{ev.get('prompt_key', '')}:{ev.get('node_id', '')}"
            dispatch_groups.setdefault(group_key, []).append(ev)

    for ev in event_dicts:
        et = ev.get("event_type", "")
        payload = ev.get("payload") or {}
        is_error = "ERROR" in et or "FAIL" in et or "error" in payload
        if not is_error:
            continue
        error_msg = payload.get("error", "") or payload.get("message", "") or et
        group_key = f"{ev.get('prompt_key', '')}:{ev.get('node_id', '')}"
        group = dispatch_groups.get(group_key, [])
        retry_history = []
        ts = ev.get("timestamp", 0)
        for attempt_idx, attempt in enumerate(group):
            if attempt.get("timestamp", 0) > ts:
                break
            if attempt.get("event_type") == "PROMPT_DISPATCH_ERROR":
                retry_history.append({
                    "attempt": attempt_idx + 1,
                    "timestamp": attempt["timestamp"],
                    "error": (attempt.get("payload") or {}).get("error", ""),
                })
        errors.append({
            "timestamp": ts,
            "event_type": et,
            "node_id": ev.get("node_id", ""),
            "prompt_key": ev.get("prompt_key", ""),
            "provider": ev.get("provider", ""),
            "model": ev.get("model", ""),
            "stage": ev.get("stage", ""),
            "error_message": str(error_msg),
            "dispatch_id": ev.get("dispatch_id", ""),
            "retry_count": len(retry_history),
            "retry_history": retry_history,
        })

    errors.sort(key=lambda e: e["timestamp"])
    return {"run_id": run_id, "error_count": len(errors), "errors": errors}


@app.get("/api/dashboard/runs/{run_id}/coverage")
async def run_prompt_coverage(run_id: str) -> dict[str, Any]:
    """Deterministic vs LLM fallback coverage per prompt key."""
    from ageom.telemetry import get_event_log
    from ageom.telemetry import _pg_store as _store_ref

    # Try Postgres first for coverage events
    pg_event_dicts: list[dict[str, Any]] | None = None
    if _store_ref is not None:
        try:
            pg_event_dicts = await _store_ref.list_events_for_coverage(run_id)
        except Exception:
            pass

    if pg_event_dicts is not None:
        return _compute_coverage_from_dicts(run_id, pg_event_dicts)

    events = get_event_log().events_for_run(run_id)
    keys: dict[str, dict[str, Any]] = {}
    for ev in events:
        if ev.event_type not in ("PROMPT_DISPATCH_DONE", "PROMPT_DISPATCH_ERROR"):
            continue
        pk = ev.prompt_key or "(unknown)"
        row = keys.get(pk)
        if row is None:
            row = {
                "prompt_key": pk,
                "total_dispatches": 0,
                "deterministic_count": 0,
                "llm_fallback_count": 0,
                "providers": set(),
                "error_count": 0,
                "latency_total_ms": 0.0,
            }
            keys[pk] = row
        row["total_dispatches"] += 1
        provider = ev.provider or ""
        if provider:
            row["providers"].add(provider)
        is_deterministic = (
            provider == "deterministic"
            or "_shim" in provider
            or provider.endswith("_cli")
            or ev.payload.get("critique_source") == "deterministic"
            or ev.payload.get("ghost_fix_source") == "deterministic"
            or ev.payload.get("state_hoist_source") == "deterministic"
            or ev.payload.get("tactic_source") == "deterministic"
        )
        if is_deterministic:
            row["deterministic_count"] += 1
        else:
            row["llm_fallback_count"] += 1
        if ev.event_type == "PROMPT_DISPATCH_ERROR":
            row["error_count"] += 1
        if ev.duration_ms and ev.duration_ms > 0:
            row["latency_total_ms"] += ev.duration_ms

    prompt_keys = []
    total_dispatches = 0
    total_deterministic = 0
    for row in sorted(keys.values(), key=lambda r: r["total_dispatches"], reverse=True):
        total = row["total_dispatches"]
        det = row["deterministic_count"]
        total_dispatches += total
        total_deterministic += det
        prompt_keys.append({
            "prompt_key": row["prompt_key"],
            "total_dispatches": total,
            "deterministic_count": det,
            "llm_fallback_count": row["llm_fallback_count"],
            "deterministic_pct": round(det / total * 100, 1) if total else 0.0,
            "providers": sorted(row["providers"]),
            "avg_latency_ms": round(row["latency_total_ms"] / total, 1) if total else 0.0,
            "error_count": row["error_count"],
        })

    return {
        "run_id": run_id,
        "prompt_keys": prompt_keys,
        "overall_deterministic_pct": round(
            total_deterministic / total_dispatches * 100, 1
        )
        if total_dispatches
        else 0.0,
        "total_dispatches": total_dispatches,
    }


@app.get("/api/dashboard/runs/{run_id}/errors")
async def run_error_drilldown(run_id: str) -> dict[str, Any]:
    """Structured error list with retry grouping."""
    from ageom.telemetry import get_event_log
    from ageom.telemetry import _pg_store as _store_ref

    # Try Postgres first
    if _store_ref is not None:
        try:
            pg_events = await _store_ref.list_events_for_errors(run_id)
            if pg_events is not None:
                return _compute_errors_from_dicts(run_id, pg_events)
        except Exception:
            pass

    events = get_event_log().events_for_run(run_id)

    # Group dispatches by prompt_key+node_id for retry detection
    dispatch_groups: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, Any]] = []

    for ev in events:
        if ev.event_type not in ("PROMPT_DISPATCH_DONE", "PROMPT_DISPATCH_ERROR"):
            continue
        group_key = f"{ev.prompt_key}:{ev.node_id}"
        dispatch_groups.setdefault(group_key, []).append(asdict(ev))

    for ev in events:
        is_error = (
            "ERROR" in ev.event_type
            or "FAIL" in ev.event_type
            or "error" in ev.payload
        )
        if not is_error:
            continue
        error_msg = (
            ev.payload.get("error", "")
            or ev.payload.get("message", "")
            or ev.event_type
        )
        group_key = f"{ev.prompt_key}:{ev.node_id}"
        group = dispatch_groups.get(group_key, [])
        retry_history = []
        for attempt_idx, attempt in enumerate(group):
            if attempt.get("timestamp", 0) > ev.timestamp:
                break
            if attempt.get("event_type") in ("PROMPT_DISPATCH_ERROR",):
                retry_history.append({
                    "attempt": attempt_idx + 1,
                    "timestamp": attempt["timestamp"],
                    "error": attempt.get("payload", {}).get("error", ""),
                })

        errors.append({
            "timestamp": ev.timestamp,
            "event_type": ev.event_type,
            "node_id": ev.node_id,
            "prompt_key": ev.prompt_key,
            "provider": ev.provider,
            "model": ev.model,
            "stage": ev.stage,
            "error_message": str(error_msg),
            "dispatch_id": ev.dispatch_id,
            "retry_count": len(retry_history),
            "retry_history": retry_history,
        })

    errors.sort(key=lambda e: e["timestamp"])
    return {"run_id": run_id, "error_count": len(errors), "errors": errors}


@app.get("/api/cdgs")
async def list_cdgs(
    concept_type: str | None = Query(None, description="Filter by concept type"),
    status: str | None = Query(None, description="Filter by atom status"),
    q: str | None = Query(None, description="Substring search on repo name"),
) -> list[dict[str, Any]]:
    """List available CDGs with summary stats."""
    driver = app.state.driver

    # Build WHERE clauses
    where_clauses: list[str] = []
    params: dict[str, Any] = {}
    if q:
        where_clauses.append("a.repo CONTAINS $q")
        params["q"] = q

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    cypher = f"""
    MATCH (a:Atom)
    {where_sql}
    WITH a.repo AS repo,
         count(a) AS node_count,
         collect(DISTINCT a.concept_type) AS concept_types,
         collect(DISTINCT a.status) AS statuses
    RETURN repo, node_count, concept_types, statuses
    ORDER BY repo
    """

    async with driver.session() as session:
        result = await session.run(cypher, **params)
        records = [r async for r in result]

    rows = []
    for rec in records:
        concept_types = [ct for ct in rec["concept_types"] if ct]
        statuses = [s for s in rec["statuses"] if s]

        # Apply post-query filters
        if concept_type and concept_type not in concept_types:
            continue
        if status and status not in statuses:
            continue

        rows.append({
            "repo": rec["repo"],
            "node_count": rec["node_count"],
            "concept_types": concept_types,
            "statuses": statuses,
        })

    return rows


async def _load_cdg(repo: str) -> dict[str, Any]:
    """Load full CDG JSON (nodes + edges + metadata) for a repo."""
    driver = app.state.driver

    async with driver.session() as session:
        # Fetch nodes — use explicit parameters dict for Memgraph compatibility
        node_result = await session.run(
            """
            MATCH (a:Atom)
            WHERE a.repo = $repo
            OPTIONAL MATCH (a)-[:HAS_INPUT]->(ip:InputPort)
            OPTIONAL MATCH (a)-[:HAS_OUTPUT]->(op:OutputPort)
            OPTIONAL MATCH (a)-[:PARENT_OF]->(child:Atom)
            OPTIONAL MATCH (parent:Atom)-[:PARENT_OF]->(a)
            RETURN a, collect(DISTINCT ip) AS inputs,
                   collect(DISTINCT op) AS outputs,
                   collect(DISTINCT child.node_id) AS children,
                   parent.node_id AS parent_id
            """,
            parameters={"repo": repo},
        )
        node_records = [r async for r in node_result]

        if not node_records:
            raise HTTPException(status_code=404, detail=f"CDG not found: {repo}")

        # Fetch edges
        edge_result = await session.run(
            """
            MATCH (s:Atom)-[r:DATA_FLOW]->(t:Atom)
            WHERE s.repo = $repo AND t.repo = $repo
            RETURN s.node_id AS source_id, t.node_id AS target_id,
                   r.output_name AS output_name, r.input_name AS input_name,
                   r.source_type AS source_type, r.target_type AS target_type,
                   r.requires_glue AS requires_glue
            """,
            parameters={"repo": repo},
        )
        edge_records = [r async for r in edge_result]

    # Build nodes list
    nodes = []
    for rec in node_records:
        atom = dict(rec["a"])
        # Strip internal props, keep domain props
        node: dict[str, Any] = {
            "node_id": atom.get("node_id", ""),
            "name": atom.get("name", ""),
            "description": atom.get("description", ""),
            "concept_type": atom.get("concept_type", ""),
            "status": atom.get("status", "atomic"),
            "depth": atom.get("depth", 0),
            "type_signature": atom.get("type_signature", ""),
            "is_optional": atom.get("is_optional", False),
            "is_opaque": atom.get("is_opaque", False),
            "is_external": atom.get("is_external", False),
            "parallelizable": atom.get("parallelizable", False),
            "conceptual_summary": atom.get("conceptual_summary", ""),
        }

        # Inputs/outputs
        node["inputs"] = [
            {
                "name": dict(ip).get("name", ""),
                "type_desc": dict(ip).get("type_desc", ""),
                "constraints": dict(ip).get("constraints", ""),
            }
            for ip in rec["inputs"]
            if ip is not None
        ]
        node["outputs"] = [
            {
                "name": dict(op).get("name", ""),
                "type_desc": dict(op).get("type_desc", ""),
                "constraints": dict(op).get("constraints", ""),
            }
            for op in rec["outputs"]
            if op is not None
        ]

        # Children / parent
        children = [c for c in rec["children"] if c is not None]
        if children:
            node["children"] = children
        if rec["parent_id"]:
            node["parent_id"] = rec["parent_id"]

        nodes.append(node)

    # Build edges list
    edges = []
    for rec in edge_records:
        edges.append({
            "source_id": rec["source_id"],
            "target_id": rec["target_id"],
            "output_name": rec["output_name"] or "",
            "input_name": rec["input_name"] or "",
            "source_type": rec["source_type"] or "",
            "target_type": rec["target_type"] or "",
            "requires_glue": bool(rec["requires_glue"]),
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {"repo": repo},
    }


@app.get("/api/cdg")
async def get_cdg(repo: str = Query(..., description="Full repo path")) -> dict[str, Any]:
    """Return full CDG JSON (nodes + edges + metadata) for a repo."""
    return await _load_cdg(repo)


@app.get("/api/cdgs/{repo:path}")
async def get_cdg_by_path(repo: str) -> dict[str, Any]:
    """Backward-compatible path form: /api/cdgs/{repo}."""
    return await _load_cdg(repo)


class IsomorphismQuery(BaseModel):
    repo: str
    node_id: str
    radius: int = 0
    min_jaccard: float = 0.3
    max_results: int = 20
    layers: list[int] = [1, 2, 3]


@app.post("/api/isomorphisms")
async def find_isomorphisms(query: IsomorphismQuery) -> dict[str, Any]:
    """Find similar subgraphs using 3-layer retrieval."""
    driver = app.state.driver

    async with driver.session() as session:
        # 1. Resolve the target node
        node_result = await session.run(
            """
            MATCH (a:Atom)
            WHERE a.repo = $repo AND a.node_id = $node_id
            OPTIONAL MATCH (a)-[:PARENT_OF]->(child:Atom)
            RETURN a, collect(DISTINCT child.concept_type) AS child_types,
                   count(DISTINCT child) AS n_children
            """,
            parameters={"repo": query.repo, "node_id": query.node_id},
        )
        node_rec = await node_result.single()
        if not node_rec:
            raise HTTPException(status_code=404, detail=f"Node not found: {query.repo}/{query.node_id}")

        target = dict(node_rec["a"])
        n_children = node_rec["n_children"]

        # If the node is atomic (no children), walk up to nearest decomposed ancestor
        if n_children == 0 or target.get("status") == "atomic":
            parent_result = await session.run(
                """
                MATCH (child:Atom)
                WHERE child.repo = $repo AND child.node_id = $node_id
                MATCH (parent:Atom)-[:PARENT_OF]->(child)
                OPTIONAL MATCH (parent)-[:PARENT_OF]->(sibling:Atom)
                RETURN parent, collect(DISTINCT sibling.concept_type) AS child_types,
                       count(DISTINCT sibling) AS n_children
                """,
                parameters={"repo": query.repo, "node_id": query.node_id},
            )
            parent_rec = await parent_result.single()
            if parent_rec:
                target = dict(parent_rec["parent"])
                n_children = parent_rec["n_children"]

        # If radius > 0 and we have a parent, walk up further
        if query.radius > 0:
            for _ in range(query.radius):
                up_result = await session.run(
                    """
                    MATCH (child:Atom)
                    WHERE child.repo = $repo AND child.node_id = $node_id
                    MATCH (parent:Atom)-[:PARENT_OF]->(child)
                    OPTIONAL MATCH (parent)-[:PARENT_OF]->(sibling:Atom)
                    RETURN parent, collect(DISTINCT sibling.concept_type) AS child_types,
                           count(DISTINCT sibling) AS n_children
                    """,
                    parameters={"repo": target.get("repo", query.repo), "node_id": target.get("node_id", "")},
                )
                up_rec = await up_result.single()
                if up_rec:
                    target = dict(up_rec["parent"])
                    n_children = up_rec["n_children"]
                else:
                    break

        target_fqn = target.get("fqn", f"{target.get('repo', query.repo)}.{target.get('node_id', '')}")
        target_repo = target.get("repo", query.repo)

        query_node = {
            "fqn": target_fqn,
            "name": target.get("name", ""),
            "concept_type": target.get("concept_type", ""),
            "n_children": n_children,
        }

        # Collect results by fqn, keep highest score
        results_by_fqn: dict[str, dict[str, Any]] = {}

        # Layer 1: topo_hash exact match
        topo_hash = target.get("topo_hash")
        if 1 in query.layers and topo_hash:
            topo_result = await session.run(
                """
                MATCH (parent:Atom:Decomposed)
                WHERE parent.topo_hash = $topo_hash AND parent.repo <> $exclude_repo
                MATCH (parent)-[:PARENT_OF]->(child:Atom)
                WITH parent, collect(DISTINCT child.concept_type) AS child_types,
                     count(DISTINCT child) AS n_children
                RETURN parent.fqn AS fqn, parent.repo AS repo, parent.name AS name,
                       parent.concept_type AS concept_type, parent.topo_hash AS topo_hash,
                       n_children, child_types
                LIMIT $limit
                """,
                parameters={"topo_hash": topo_hash, "exclude_repo": target_repo, "limit": query.max_results},
            )
            async for rec in topo_result:
                fqn = rec["fqn"]
                results_by_fqn[fqn] = {
                    "fqn": fqn,
                    "repo": rec["repo"],
                    "name": rec["name"],
                    "concept_type": rec["concept_type"],
                    "topo_hash": rec["topo_hash"],
                    "n_children": rec["n_children"],
                    "score": 1.0,
                    "jaccard_score": None,
                    "layer": 1,
                    "children_summary": rec["child_types"],
                }

        # Layer 2: structural match (concept_type + port arity ±1)
        if 2 in query.layers:
            struct_result = await session.run(
                """
                MATCH (parent:Atom:Decomposed)
                WHERE parent.concept_type = $concept_type
                  AND parent.repo <> $exclude_repo
                  AND abs(parent.n_inputs - $n_inputs) <= 1
                  AND abs(parent.n_outputs - $n_outputs) <= 1
                MATCH (parent)-[:PARENT_OF]->(child:Atom)
                WITH parent, collect(DISTINCT child.concept_type) AS child_types,
                     count(DISTINCT child) AS n_children
                WHERE n_children >= 2
                RETURN parent.fqn AS fqn, parent.repo AS repo, parent.name AS name,
                       parent.concept_type AS concept_type, parent.topo_hash AS topo_hash,
                       n_children, child_types
                ORDER BY n_children DESC
                LIMIT $limit
                """,
                parameters={
                    "concept_type": target.get("concept_type", ""),
                    "n_inputs": target.get("n_inputs", 0) or 0,
                    "n_outputs": target.get("n_outputs", 0) or 0,
                    "exclude_repo": target_repo,
                    "limit": query.max_results,
                },
            )
            async for rec in struct_result:
                fqn = rec["fqn"]
                # Compute IO match score
                n_in_diff = abs((target.get("n_inputs", 0) or 0) - (rec["n_children"] or 0))
                io_match = 1.0 if n_in_diff == 0 else 0.8
                score = 0.7 * io_match
                if fqn not in results_by_fqn or results_by_fqn[fqn]["score"] < score:
                    results_by_fqn[fqn] = {
                        "fqn": fqn,
                        "repo": rec["repo"],
                        "name": rec["name"],
                        "concept_type": rec["concept_type"],
                        "topo_hash": rec["topo_hash"],
                        "n_children": rec["n_children"],
                        "score": score,
                        "jaccard_score": None,
                        "layer": 2,
                        "children_summary": rec["child_types"],
                    }

        # Layer 3: Jaccard neighborhood
        if 3 in query.layers:
            jaccard_result = await session.run(
                """
                MATCH (query:Atom)
                WHERE query.fqn = $fqn
                MATCH (query)-[:PARENT_OF]->(qc:Atom)
                WITH query, collect(qc) AS query_children
                WHERE size(query_children) > 0
                MATCH (candidate:Atom:Decomposed)-[:PARENT_OF]->(cc:Atom)
                WHERE candidate.repo <> $exclude_repo AND candidate.fqn <> $fqn
                WITH query, query_children, candidate, collect(cc) AS cand_children
                WHERE size(cand_children) > 0
                WITH candidate,
                     [qc IN query_children | qc.concept_type] AS q_types,
                     [cc IN cand_children | cc.concept_type] AS c_types
                WITH candidate,
                     toFloat(size([x IN q_types WHERE x IN c_types])) /
                     toFloat(size(q_types + [y IN c_types WHERE NOT y IN q_types])) AS jaccard_score
                WHERE jaccard_score > $min_jaccard
                WITH candidate, jaccard_score
                ORDER BY jaccard_score DESC
                LIMIT $limit
                MATCH (candidate)-[:PARENT_OF]->(child:Atom)
                WITH candidate, jaccard_score,
                     collect(DISTINCT child.concept_type) AS child_types,
                     count(DISTINCT child) AS n_children
                RETURN candidate.fqn AS fqn, candidate.repo AS repo,
                       candidate.name AS name, candidate.concept_type AS concept_type,
                       candidate.topo_hash AS topo_hash,
                       n_children, child_types, jaccard_score
                """,
                parameters={
                    "fqn": target_fqn,
                    "exclude_repo": target_repo,
                    "min_jaccard": query.min_jaccard,
                    "limit": query.max_results,
                },
            )
            async for rec in jaccard_result:
                fqn = rec["fqn"]
                j_score = rec["jaccard_score"]
                score = j_score  # Jaccard score IS the score for layer 3
                existing = results_by_fqn.get(fqn)
                if existing:
                    existing["jaccard_score"] = j_score
                    if score > existing["score"]:
                        existing["score"] = score
                        existing["layer"] = 3
                else:
                    results_by_fqn[fqn] = {
                        "fqn": fqn,
                        "repo": rec["repo"],
                        "name": rec["name"],
                        "concept_type": rec["concept_type"],
                        "topo_hash": rec["topo_hash"],
                        "n_children": rec["n_children"],
                        "score": score,
                        "jaccard_score": j_score,
                        "layer": 3,
                        "children_summary": rec["child_types"],
                    }

    # Sort by score descending, cap at max_results
    results = sorted(results_by_fqn.values(), key=lambda r: r["score"], reverse=True)
    results = results[: query.max_results]

    return {"query_node": query_node, "results": results}


class NoCacheStaticFiles(StaticFiles):
    """StaticFiles that sends no-cache headers so dev reloads always get fresh files."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        original_send = send

        async def send_with_no_cache(message: dict) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"cache-control", b"no-cache, no-store, must-revalidate"))
                message["headers"] = headers
            await original_send(message)

        await super().__call__(scope, receive, send_with_no_cache)


# Mount static files last so API routes take priority
_static_dir = Path(__file__).resolve().parent / "static"
if _static_dir.exists():
    app.mount("/", NoCacheStaticFiles(directory=str(_static_dir), html=True), name="static")
