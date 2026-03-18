"""CLI commands for inspecting telemetry runs."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone


def _cmd_telemetry_list(args: argparse.Namespace) -> None:
    """List recent telemetry runs with status, label, and timing."""
    import asyncio

    from ageom.config import AgeomConfig
    from ageom.telemetry import list_runtime_runs, load_persisted_runs, load_runs_from_store

    config = AgeomConfig()
    limit = getattr(args, "limit", 20)
    state_filter = getattr(args, "state", "all").strip().lower()

    # Try Postgres first
    pg_runs = None
    try:
        pg_runs = asyncio.run(
            load_runs_from_store(
                limit=max(limit * 3, 100),
                status=state_filter if state_filter != "all" else None,
            )
        )
    except Exception:
        pass

    persisted = load_persisted_runs(config.telemetry_runs_dir, limit=max(limit * 3, 100)) if pg_runs is None else []
    runtime = list_runtime_runs()

    # Merge (runtime wins for same run_id)
    merged: dict[str, dict] = {}
    for row in (pg_runs or persisted) + runtime:
        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            continue
        current = merged.get(run_id)
        if current is None or float(row.get("last_update_at", 0)) >= float(
            current.get("last_update_at", 0)
        ):
            merged[run_id] = row

    rows = sorted(merged.values(), key=lambda r: float(r.get("last_update_at", 0)), reverse=True)

    if state_filter != "all":
        rows = [r for r in rows if str(r.get("status", "")).lower() == state_filter]

    rows = rows[:limit]

    if not rows:
        print("No telemetry runs found.")
        return

    # Format output
    print(f"{'RUN ID':<34} {'STATUS':<11} {'PIPELINE':<22} {'LABEL':<20} {'STARTED':<20} {'DURATION':<10}")
    print("-" * 117)

    for run in rows:
        run_id = run.get("run_id", "")
        status = run.get("status", "running")
        pipeline = run.get("pipeline", "")
        label = run.get("label", "") or ""
        started_at = run.get("started_at")
        ended_at = run.get("ended_at")

        if started_at:
            started_str = datetime.fromtimestamp(started_at, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        else:
            started_str = "--"

        if started_at and ended_at:
            duration = f"{ended_at - started_at:.1f}s"
        elif started_at:
            duration = "running"
        else:
            duration = "--"

        # Truncate label
        if len(label) > 18:
            label = label[:17] + "\u2026"

        print(f"{run_id:<34} {status:<11} {pipeline:<22} {label:<20} {started_str:<20} {duration:<10}")

    print(f"\n{len(rows)} run(s) shown. Use --limit N to see more.")


def _cmd_telemetry_show(args: argparse.Namespace) -> None:
    """Show details for a specific telemetry run."""
    import asyncio
    import json

    from ageom.config import AgeomConfig
    from ageom.telemetry import get_persisted_run, get_runtime_run, load_run_from_store

    config = AgeomConfig()
    run_id = args.run_id.strip()

    run = get_runtime_run(run_id)
    if run is None:
        try:
            run = asyncio.run(load_run_from_store(run_id))
        except Exception:
            pass
    if run is None:
        run = get_persisted_run(config.telemetry_runs_dir, run_id)
    if run is None:
        print(f"Run not found: {run_id}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(run, indent=2, default=str))
