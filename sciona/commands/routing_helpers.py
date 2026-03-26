"""Routing, mode, and retrieval-policy helpers for CLI commands."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sciona.config import AgeomConfig


@dataclass(frozen=True)
class RetrievalPolicy:
    """Effective retrieval gates after combining mode and catalog-confidence signals."""

    catalog_confidence: float
    confidence_band: str
    confidence_source: str
    skill_index_enabled: bool
    graph_retrieval_enabled: bool
    semantic_index_backend_override: str | None
    hunter_mode: str


def _summarize_prompt_routing(
    config: "AgeomConfig",
    round_name: str,
    prompt_keys: list[str],
    execution_mode: str | None = None,
) -> dict[str, Any]:
    """Summarize which prompt-key overrides are active vs suppressed."""
    from sciona.config import (
        BENCHMARK_JUSTIFIED_PROMPT_KEYS,
        effective_round_provider_model,
        prompt_override_matches_code_default,
        should_apply_prompt_override,
    )

    resolved_mode = str(
        execution_mode or getattr(config, "execution_mode", "verified") or "verified"
    ).strip().lower()
    default_provider, default_model = effective_round_provider_model(
        config,
        round_name,
        execution_mode=resolved_mode,
    )
    active_overrides: list[dict[str, str]] = []
    suppressed_defaults: list[str] = []
    custom_nonbenchmark: list[str] = []

    for key in prompt_keys:
        provider = getattr(config, f"{key}_llm_provider", "")
        model = getattr(config, f"{key}_llm_model", "")
        if not provider:
            continue
        if should_apply_prompt_override(config, key, resolved_mode):
            model = model or config.llm_model
            active_overrides.append(
                {"prompt_key": key, "provider": provider, "model": model}
            )
            if key not in BENCHMARK_JUSTIFIED_PROMPT_KEYS:
                custom_nonbenchmark.append(key)
            continue
        if prompt_override_matches_code_default(config, key):
            suppressed_defaults.append(key)

    return {
        "round": round_name,
        "mode": resolved_mode,
        "default_provider": default_provider,
        "default_model": default_model,
        "active_overrides": active_overrides,
        "suppressed_default_overrides": suppressed_defaults,
        "custom_nonbenchmark_overrides": custom_nonbenchmark,
    }


def _print_prompt_routing_summary(
    config: "AgeomConfig",
    round_name: str,
    prompt_keys: list[str],
    execution_mode: str | None = None,
) -> dict[str, Any]:
    """Print a compact routing audit and return the structured summary."""
    summary = _summarize_prompt_routing(
        config,
        round_name,
        prompt_keys,
        execution_mode,
    )
    active = ", ".join(
        f"{row['prompt_key']}={row['provider']}:{row['model']}"
        for row in summary["active_overrides"]
    ) or "none"
    suppressed = ", ".join(summary["suppressed_default_overrides"]) or "none"
    custom = ", ".join(summary["custom_nonbenchmark_overrides"]) or "none"
    print(
        f"LLM routing ({round_name}): "
        f"mode={summary['mode']} "
        f"default={summary['default_provider']}:{summary['default_model']} "
        f"active=[{active}] "
        f"suppressed_defaults=[{suppressed}] "
        f"custom_nonbenchmark=[{custom}]"
    )
    return summary


def _routing_metadata_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Shrink routing audit into dashboard-friendly run metadata."""
    return {
        "round": summary["round"],
        "mode": summary["mode"],
        "default_provider": summary["default_provider"],
        "default_model": summary["default_model"],
        "active_overrides": [
            {
                "prompt_key": row["prompt_key"],
                "provider": row["provider"],
                "model": row["model"],
            }
            for row in summary["active_overrides"]
        ],
        "suppressed_default_overrides": list(summary["suppressed_default_overrides"]),
        "custom_nonbenchmark_overrides": list(summary["custom_nonbenchmark_overrides"]),
    }


def _resolve_retrieval_policy(
    *,
    mode_settings: Any,
    catalog: Any | None,
    texts: list[str] | tuple[str, ...],
    config: Any | None = None,
) -> RetrievalPolicy:
    """Decide whether retrieval should stay enabled for the current task."""
    semantic_override = getattr(mode_settings, "semantic_index_backend_override", None)
    if semantic_override is None and config is not None:
        cfg_backend = str(
            getattr(config, "semantic_index_backend", "auto")
        ).strip().lower()
        if cfg_backend not in {"auto", ""}:
            semantic_override = cfg_backend
    hunter_mode = str(getattr(mode_settings, "hunter_mode", "standard"))
    graph_enabled = bool(getattr(mode_settings, "graph_retrieval_enabled", False))
    skill_enabled = bool(getattr(mode_settings, "skill_index_enabled", False))

    if catalog is None or getattr(catalog, "size", 0) == 0:
        return RetrievalPolicy(
            catalog_confidence=0.0,
            confidence_band="none",
            confidence_source="empty_catalog",
            skill_index_enabled=False,
            graph_retrieval_enabled=False,
            semantic_index_backend_override=semantic_override or "lexical",
            hunter_mode="standard",
        )

    confidences = []
    for text in texts:
        stripped = str(text or "").strip()
        if not stripped:
            continue
        confidence = catalog.estimate_confidence(stripped)
        confidences.append((stripped, confidence))

    if not confidences:
        return RetrievalPolicy(
            catalog_confidence=0.0,
            confidence_band="low",
            confidence_source="no_text",
            skill_index_enabled=False,
            graph_retrieval_enabled=False,
            semantic_index_backend_override=semantic_override or "lexical",
            hunter_mode="standard",
        )

    best_text, best = max(confidences, key=lambda item: item[1].score)
    band = "low"
    if best.score >= 0.70:
        band = "high"
    elif best.score >= 0.40:
        band = "medium"

    if band == "low":
        return RetrievalPolicy(
            catalog_confidence=best.score,
            confidence_band=band,
            confidence_source=best_text,
            skill_index_enabled=False,
            graph_retrieval_enabled=False,
            semantic_index_backend_override=semantic_override or "lexical",
            hunter_mode="standard",
        )

    if band == "medium":
        return RetrievalPolicy(
            catalog_confidence=best.score,
            confidence_band=band,
            confidence_source=best_text,
            skill_index_enabled=skill_enabled,
            graph_retrieval_enabled=False,
            semantic_index_backend_override=semantic_override or "lexical",
            hunter_mode="standard",
        )

    return RetrievalPolicy(
        catalog_confidence=best.score,
        confidence_band=band,
        confidence_source=best_text,
        skill_index_enabled=skill_enabled,
        graph_retrieval_enabled=graph_enabled,
        semantic_index_backend_override=semantic_override,
        hunter_mode=hunter_mode,
    )


def _print_retrieval_policy(policy: RetrievalPolicy) -> None:
    """Print the effective retrieval policy for the current task."""
    print(
        "Retrieval policy: "
        f"catalog_confidence={policy.catalog_confidence:.2f} "
        f"({policy.confidence_band}), "
        f"skill_index={'on' if policy.skill_index_enabled else 'off'}, "
        f"graph_retrieval={'on' if policy.graph_retrieval_enabled else 'off'}, "
        f"semantic_backend={policy.semantic_index_backend_override or 'default'}, "
        f"hunter_mode={policy.hunter_mode}"
    )


def _parse_prompt_benchmark_provider_specs(
    raw_specs: list[str] | None,
) -> list[tuple[str, str]]:
    specs = [spec.strip() for spec in (raw_specs or []) if spec and spec.strip()]
    if not specs:
        raise ValueError("At least one --provider provider:model spec is required")

    parsed: list[tuple[str, str]] = []
    for spec in specs:
        provider, sep, model = spec.partition(":")
        provider = provider.strip()
        model = model.strip()
        if not sep or not provider or not model:
            raise ValueError(
                f"Invalid provider spec '{spec}'. Expected format provider:model"
            )
        parsed.append((provider, model))
    return parsed


def _add_mode_argument(parser: argparse.ArgumentParser) -> None:
    """Add a shared execution-mode argument to a subcommand parser."""
    parser.add_argument(
        "--mode",
        choices=["rapid", "structured", "single_agent", "verified"],
        default=None,
        help="Execution mode override (default: SCIONA_EXECUTION_MODE or verified)",
    )


def _add_label_argument(parser: argparse.ArgumentParser) -> None:
    """Add a --label argument for tagging telemetry runs."""
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Human-readable label for this telemetry run",
    )


def _mode_feature_summary(mode_settings: Any) -> dict[str, str]:
    """Render the resolved execution-mode feature gates for display/telemetry."""
    return {
        "mode": str(mode_settings.mode),
        "skill_index": "on" if mode_settings.skill_index_enabled else "off",
        "graph_retrieval": "on" if mode_settings.graph_retrieval_enabled else "off",
        "architect_context": "on"
        if mode_settings.architect_shared_context_enabled
        else "off",
        "hunter_context": "on" if mode_settings.hunter_shared_context_enabled else "off",
        "synth_context": "on"
        if mode_settings.synthesizer_shared_context_enabled
        else "off",
        "ingester_context": "on"
        if mode_settings.ingester_shared_context_enabled
        else "off",
        "hunter_mode": str(mode_settings.hunter_mode),
        "hunter_gbnf": "on" if mode_settings.hunter_use_gbnf else "off",
        "semantic_backend": (
            str(mode_settings.semantic_index_backend_override)
            if mode_settings.semantic_index_backend_override
            else "default"
        ),
    }


def _print_mode_summary(command_name: str, mode_settings: Any) -> None:
    """Print a compact execution-mode summary for the current command."""
    summary = _mode_feature_summary(mode_settings)
    print(
        f"Execution mode ({command_name}): {summary['mode']} "
        f"[skill_index={summary['skill_index']}, "
        f"graph_retrieval={summary['graph_retrieval']}, "
        f"architect_context={summary['architect_context']}, "
        f"hunter_context={summary['hunter_context']}, "
        f"synth_context={summary['synth_context']}, "
        f"ingester_context={summary['ingester_context']}, "
        f"hunter_mode={summary['hunter_mode']}, "
        f"hunter_gbnf={summary['hunter_gbnf']}, "
        f"semantic_backend={summary['semantic_backend']}]"
    )
