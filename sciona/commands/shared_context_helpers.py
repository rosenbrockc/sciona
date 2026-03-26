"""Shared-context helpers for CLI command handlers."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sciona.config import AgeomConfig
    from sciona.shared_context import SharedContextMetrics, SharedContextStore


async def _create_shared_context(
    config: "AgeomConfig",
    *,
    enabled: bool,
) -> tuple["SharedContextStore | None", "SharedContextMetrics | None"]:
    """Create a shared context store and metrics wrapper for this command."""
    from sciona.shared_context import SharedContextMetrics, create_shared_context_store

    if not enabled:
        return None, None

    os.environ["SCIONA_SHARED_CONTEXT_INCLUDE_PROVENANCE"] = (
        "1" if config.shared_context_include_provenance else "0"
    )
    metrics = SharedContextMetrics()
    store = await create_shared_context_store(
        enabled=True,
        backend=config.shared_context_backend,
        postgres_uri=config.postgres_uri,
        postgres_table=config.shared_context_postgres_table,
        max_records_per_namespace=config.shared_context_max_records_per_namespace,
        ttl_hours=config.shared_context_ttl_hours,
        promotion_enabled=config.shared_context_promotion_enabled,
        promotion_min_confidence=config.shared_context_promotion_min_confidence,
        repo_namespace=config.shared_context_repo_namespace,
        metrics=metrics,
    )
    return store, metrics


def _print_shared_context_metrics(
    label: str,
    metrics: "SharedContextMetrics | None",
) -> None:
    """Print a compact shared-context metrics line."""
    if metrics is None:
        return
    snap = metrics.snapshot()
    print(
        "  Shared context"
        f" ({label}): backend={snap['backend']} "
        f"searches={snap['searches_total']} "
        f"hit_rate={float(snap['search_hit_rate']):.2f} "
        f"avg_search_ms={float(snap['search_latency_ms_avg']):.1f} "
        f"puts={snap['puts_total']} "
        f"dup_supp_rate={float(snap['duplicate_suppression_rate']):.2f} "
        f"match_delta={float(snap['match_success_delta']):+.2f} "
        f"promotions={snap['promotions_total']} "
        f"injected_blocks={snap['injected_blocks']} "
        f"injected_chars={snap['injected_chars']} "
        f"template_hits={snap['template_search_hits']}/{snap['template_searches_total']} "
        f"template_puts={snap['template_puts_total']} "
        f"template_injected={snap['template_injected_blocks']}"
    )


def _snapshot_shared_context_metrics(
    metrics_by_label: dict[str, "SharedContextMetrics | None"],
) -> dict[str, dict[str, float | int | str]]:
    payload: dict[str, dict[str, float | int | str]] = {}
    for label, metrics in metrics_by_label.items():
        if metrics is None:
            continue
        payload[label] = metrics.snapshot()
    return payload


def _shared_context_metadata(
    metrics_by_label: dict[str, "SharedContextMetrics | None"],
    *,
    metrics_path: Path | None = None,
) -> dict[str, object]:
    """Build run-metadata payload for dashboard shared-context summaries."""
    contexts = _snapshot_shared_context_metrics(metrics_by_label)
    payload: dict[str, object] = {"contexts": contexts}
    if metrics_path is not None:
        payload["metrics_path"] = str(metrics_path)
    return payload


def _write_shared_context_metrics_file(
    path: Path,
    metrics_by_label: dict[str, "SharedContextMetrics | None"],
) -> Path | None:
    """Persist shared-context metrics JSON; return path when written."""
    contexts = _snapshot_shared_context_metrics(metrics_by_label)
    if not contexts:
        return None
    payload = {
        "generated_at_unix": time.time(),
        "contexts": contexts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path
