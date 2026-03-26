"""Common coercion and run-shaping helpers for the telemetry dashboard."""

from __future__ import annotations

from typing import Any


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _str(value: Any, default: str = "") -> str:
    return str(value if value is not None else default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value if value is not None else default)
    except Exception:
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except Exception:
        return default


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def _transport_for_provider(provider: str) -> str:
    lowered = provider.strip().lower()
    if lowered.endswith("_shim"):
        return "persistent_shim"
    if lowered.endswith("_cli"):
        return "legacy_cli"
    if lowered == "llama_cpp":
        return "local_server"
    if lowered in {"anthropic", "codex", "openai"}:
        return "api"
    return "--" if not lowered else "other"


def _merge_runs(
    persisted: list[dict[str, Any]],
    runtime: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in persisted + runtime:
        run_id = _str(row.get("run_id", "")).strip()
        if not run_id:
            continue
        current = merged.get(run_id)
        if current is None or _float(row.get("last_update_at")) >= _float(
            current.get("last_update_at")
        ):
            merged[run_id] = row
    rows = list(merged.values())
    rows.sort(key=lambda r: _float(r.get("last_update_at")), reverse=True)
    return rows


def _annotate_hang_signals(
    run: dict[str, Any],
    *,
    stale_seconds: int,
    now: float,
) -> dict[str, Any]:
    stages = _dict(run.get("stages"))
    stale_stages: list[dict[str, Any]] = []
    for stage_name, stage in stages.items():
        if not isinstance(stage, dict) or _str(stage.get("status")) != "running":
            continue
        hb = _float(stage.get("last_heartbeat_at"))
        if hb <= 0:
            continue
        age = max(0.0, now - hb)
        if age >= stale_seconds:
            stale_stages.append(
                {
                    "stage": stage_name,
                    "heartbeat_age_sec": age,
                    "message": _str(stage.get("message")),
                }
            )
    out = dict(run)
    out["stale_stages"] = stale_stages
    out["is_hung"] = bool(stale_stages) and _str(run.get("status")) == "running"
    return out
