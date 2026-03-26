"""Coverage and error drilldown helpers for dashboard event streams."""

from __future__ import annotations

from typing import Any

from sciona.visualizer.dashboard_common import _dict, _float, _str


def _compute_coverage_from_dicts(
    run_id: str, event_dicts: list[dict[str, Any]]
) -> dict[str, Any]:
    keys: dict[str, dict[str, Any]] = {}
    for ev in event_dicts:
        if ev.get("event_type") not in ("PROMPT_DISPATCH_DONE", "PROMPT_DISPATCH_ERROR"):
            continue
        prompt_key = ev.get("prompt_key") or "(unknown)"
        row = keys.setdefault(
            prompt_key,
            {
                "prompt_key": prompt_key,
                "total_dispatches": 0,
                "deterministic_count": 0,
                "llm_fallback_count": 0,
                "providers": set(),
                "error_count": 0,
                "latency_total_ms": 0.0,
            },
        )
        row["total_dispatches"] += 1
        provider = ev.get("provider") or ""
        if provider:
            row["providers"].add(provider)
        payload = _dict(ev.get("payload"))
        deterministic = (
            provider == "deterministic"
            or "_shim" in provider
            or provider.endswith("_cli")
            or payload.get("critique_source") == "deterministic"
            or payload.get("ghost_fix_source") == "deterministic"
            or payload.get("state_hoist_source") == "deterministic"
            or payload.get("tactic_source") == "deterministic"
        )
        row["deterministic_count" if deterministic else "llm_fallback_count"] += 1
        if ev.get("event_type") == "PROMPT_DISPATCH_ERROR":
            row["error_count"] += 1
        duration = ev.get("duration_ms")
        if duration and duration > 0:
            row["latency_total_ms"] += _float(duration)
    prompt_keys = []
    total_dispatches = 0
    total_deterministic = 0
    for row in sorted(keys.values(), key=lambda row: row["total_dispatches"], reverse=True):
        total = row["total_dispatches"]
        deterministic = row["deterministic_count"]
        total_dispatches += total
        total_deterministic += deterministic
        prompt_keys.append(
            {
                "prompt_key": row["prompt_key"],
                "total_dispatches": total,
                "deterministic_count": deterministic,
                "llm_fallback_count": row["llm_fallback_count"],
                "deterministic_pct": round(deterministic / total * 100, 1) if total else 0.0,
                "providers": sorted(row["providers"]),
                "avg_latency_ms": round(row["latency_total_ms"] / total, 1) if total else 0.0,
                "error_count": row["error_count"],
            }
        )
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
    dispatch_groups: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, Any]] = []
    for ev in event_dicts:
        if ev.get("event_type") in ("PROMPT_DISPATCH_DONE", "PROMPT_DISPATCH_ERROR"):
            key = f"{ev.get('prompt_key', '')}:{ev.get('node_id', '')}"
            dispatch_groups.setdefault(key, []).append(ev)
    for ev in event_dicts:
        et = _str(ev.get("event_type"))
        payload = _dict(ev.get("payload"))
        if "ERROR" not in et and "FAIL" not in et and "error" not in payload:
            continue
        error_msg = payload.get("error") or payload.get("message") or et
        key = f"{ev.get('prompt_key', '')}:{ev.get('node_id', '')}"
        retry_history: list[dict[str, Any]] = []
        for attempt_idx, attempt in enumerate(dispatch_groups.get(key, [])):
            if _float(attempt.get("timestamp")) > _float(ev.get("timestamp")):
                break
            if attempt.get("event_type") == "PROMPT_DISPATCH_ERROR":
                retry_history.append(
                    {
                        "attempt": attempt_idx + 1,
                        "timestamp": attempt.get("timestamp"),
                        "error": _dict(attempt.get("payload")).get("error", ""),
                    }
                )
        errors.append(
            {
                "timestamp": ev.get("timestamp", 0),
                "event_type": et,
                "node_id": ev.get("node_id", ""),
                "prompt_key": ev.get("prompt_key", ""),
                "provider": ev.get("provider", ""),
                "model": ev.get("model", ""),
                "stage": ev.get("stage", ""),
                "error_message": _str(error_msg),
                "dispatch_id": ev.get("dispatch_id", ""),
                "retry_count": len(retry_history),
                "retry_history": retry_history,
            }
        )
    errors.sort(key=lambda row: row["timestamp"])
    return {"run_id": run_id, "error_count": len(errors), "errors": errors}
