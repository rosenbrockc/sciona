"""Commands for prompt benchmarking, benchmark validation, and release validation."""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

from ageom.commands._helpers import (
    _parse_prompt_benchmark_provider_specs,
)


def _benchmark_validation_metadata(summary: dict[str, object]) -> dict[str, object]:
    """Normalize benchmark-validation fields for telemetry/dashboard metadata."""
    return {
        "status": str(summary.get("status", "") or ""),
        "summary_report": summary["summary_report"],
        "prompt_report": summary["prompt_report"],
        "flow_report": summary["flow_report"],
        "prompt_cases": summary["prompt_cases"],
        "prompt_results": summary["prompt_results"],
        "prompt_summary": summary["prompt_summary"],
        "prompt_stability_summary": summary.get("prompt_stability_summary", ""),
        "flow_cases": summary["flow_cases"],
        "flow_results": summary["flow_results"],
        "flow_summary": summary["flow_summary"],
        "flow_stability_summary": summary.get("flow_stability_summary", ""),
        "flow_gate_summary": str(summary.get("flow_gate_summary", "") or ""),
        "flow_execution_path_summary": str(
            summary.get("flow_execution_path_summary", "") or ""
        ),
        "runtime_override_policy_summary": str(
            summary.get("runtime_override_policy_summary", "") or ""
        ),
        "health_summary": str(summary.get("health_summary", "") or ""),
        "warning_summary": str(summary.get("warning_summary", "") or ""),
        "top_warning_subcheck": str(summary.get("top_warning_subcheck", "") or ""),
        "top_warning": str(summary.get("top_warning", "") or ""),
        "failure_summary": str(summary.get("failure_summary", "") or ""),
        "top_failed_subcheck": str(summary.get("top_failed_subcheck", "") or ""),
        "top_failure": str(summary.get("top_failure", "") or ""),
        "flow_required_variants": list(summary.get("flow_required_variants", []) or []),
        "flow_comparison_variants": list(
            summary.get("flow_comparison_variants", []) or []
        ),
        "flow_execution_paths": dict(summary.get("flow_execution_paths", {}) or {}),
        "flow_prompt_volume": dict(summary.get("flow_prompt_volume", {}) or {}),
        "flow_prompt_volume_summary": str(
            summary.get("flow_prompt_volume_summary", "") or ""
        ),
        "flow_avg_prompt_calls": dict(summary.get("flow_avg_prompt_calls", {}) or {}),
        "prompt_avg_latency_ms": dict(summary.get("prompt_avg_latency_ms", {}) or {}),
        "flow_avg_latency_ms": dict(summary.get("flow_avg_latency_ms", {}) or {}),
        "prompt_tuned_failures": int(summary.get("prompt_tuned_failures", 0) or 0),
        "prompt_tuned_unstable_groups": int(
            summary.get("prompt_tuned_unstable_groups", 0) or 0
        ),
        "flow_mode_failures": int(summary.get("flow_mode_failures", 0) or 0),
        "flow_mode_unstable_groups": int(
            summary.get("flow_mode_unstable_groups", 0) or 0
        ),
        "flow_comparison_failures": int(summary.get("flow_comparison_failures", 0) or 0),
        "flow_comparison_unstable_groups": int(
            summary.get("flow_comparison_unstable_groups", 0) or 0
        ),
        "runtime_complexity": dict(summary.get("runtime_complexity", {}) or {}),
        "coverage_monotonic": bool(summary.get("coverage_monotonic", True)),
        "coverage_by_variant": dict(summary.get("coverage_by_variant", {}) or {}),
    }


async def _cmd_prompt_benchmark(args: argparse.Namespace) -> None:
    """Run prompt-key A/B benchmarks across a small cross-domain suite."""
    from ageom.config import AgeomConfig
    from ageom.hunter.llm import create_llm_client
    from ageom.prompt_benchmark import (
        PromptBenchmarkProvider,
        format_prompt_benchmark_summary,
        run_prompt_benchmark,
        save_prompt_benchmark_report,
        select_prompt_benchmark_cases,
        summarize_prompt_benchmark,
    )

    config = AgeomConfig()
    try:
        provider_specs = _parse_prompt_benchmark_provider_specs(args.provider)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    providers: list[PromptBenchmarkProvider] = []
    try:
        for provider_name, model_name in provider_specs:
            client = create_llm_client(
                provider=provider_name,
                model=model_name,
                max_tokens=args.max_tokens or config.hunter_llm_max_tokens,
                anthropic_api_key=config.anthropic_api_key,
                openai_api_key=config.openai_api_key,
                openai_base_url=config.openai_base_url,
                llama_cpp_base_url=config.llama_cpp_base_url,
                llama_cpp_api_key=config.llama_cpp_api_key,
                use_agent_layer=config.use_agent_layer,
            )
            providers.append(
                PromptBenchmarkProvider(
                    name=f"{provider_name}:{model_name}",
                    client=client,
                )
            )

        cases = select_prompt_benchmark_cases(prompt_keys=args.prompt_key)
        results = await run_prompt_benchmark(
            providers=providers,
            cases=cases,
            repeats=args.repeats,
            compare_direct_baseline=bool(args.compare_direct_baseline),
        )
        aggregates = summarize_prompt_benchmark(results)
        print(format_prompt_benchmark_summary(aggregates))

        if args.output:
            save_prompt_benchmark_report(
                args.output,
                results=results,
                aggregates=aggregates,
            )
            print(f"\nSaved report: {args.output}")
    finally:
        for provider in providers:
            close = getattr(provider.client, "close", None)
            if not callable(close):
                continue
            maybe_result = close()
            if inspect.isawaitable(maybe_result):
                await maybe_result


async def _cmd_benchmark_validate(args: argparse.Namespace) -> None:
    """Run deterministic release-style benchmark validation."""
    from ageom.benchmark_validation import run_benchmark_validation
    from ageom.config import AgeomConfig
    from ageom.telemetry import configure_dashboard_output, finish_run, merge_run_metadata, start_run

    config = AgeomConfig()
    configure_dashboard_output(config.telemetry_runs_dir)
    telemetry_run_id = start_run(
        "benchmark_validation",
        metadata={
            "command": "benchmark-validate",
            "output_dir": str(args.output),
        },
    )

    try:
        summary = await run_benchmark_validation(args.output)
        merge_run_metadata(
            {
                "benchmark_validation": _benchmark_validation_metadata(summary)
            },
            run_id=telemetry_run_id,
        )
        if str(summary.get("status", "failed")) == "passed":
            finish_run(telemetry_run_id, status="completed")
        else:
            finish_run(
                telemetry_run_id,
                status="failed",
                error="benchmark validation failed",
            )
            raise RuntimeError("benchmark validation failed")
    except Exception as exc:
        current = str(summary.get("status", "")) if "summary" in locals() else ""
        if current == "passed":
            finish_run(telemetry_run_id, status="failed", error=str(exc))
        raise
    print("Prompt benchmark summary")
    print(summary["prompt_summary"])
    print()
    print("Flow benchmark summary")
    print(summary["flow_summary"])
    print()
    print(f"Saved validation bundle: {summary['summary_report']}")


async def _cmd_release_validate(args: argparse.Namespace) -> None:
    """Run deterministic release validation and print manifest locations."""
    from ageom.release_validation import run_release_validation
    from ageom.config import AgeomConfig
    from ageom.telemetry import configure_dashboard_output, finish_run, merge_run_metadata, start_run

    config = AgeomConfig()
    configure_dashboard_output(config.telemetry_runs_dir)
    telemetry_run_id = start_run(
        "release_validation",
        metadata={
            "command": "release-validate",
            "output_dir": str(args.output),
        },
    )
    try:
        summary = await run_release_validation(args.output)
        benchmark_summary = summary["benchmark_summary"]
        merge_run_metadata(
            {
                "release_validation": {
                    "status": summary.get("status", "failed"),
                    "manifest": summary["manifest"],
                    "benchmarks_dir": summary["benchmarks_dir"],
                    "runtime_complexity": dict(summary.get("runtime_complexity", {}) or {}),
                    "warning_summary": str(summary.get("warning_summary", "") or ""),
                    "runtime_warning_count": int(summary.get("runtime_warning_count", 0) or 0),
                    "catalog_warning_count": int(summary.get("catalog_warning_count", 0) or 0),
                    "top_runtime_warning": str(summary.get("top_runtime_warning", "") or ""),
                    "top_catalog_warning": str(summary.get("top_catalog_warning", "") or ""),
                    "failure_summary": str(summary.get("failure_summary", "") or ""),
                    "top_failed_check": str(summary.get("top_failed_check", "") or ""),
                    "top_benchmark_subcheck": str(summary.get("top_benchmark_subcheck", "") or ""),
                    "top_benchmark_failure": str(summary.get("top_benchmark_failure", "") or ""),
                    "top_runtime_failure": str(summary.get("top_runtime_failure", "") or ""),
                    "top_catalog_failure": str(summary.get("top_catalog_failure", "") or ""),
                    "catalog_validation": dict(summary.get("catalog_validation", {}) or {}),
                },
                "benchmark_validation": _benchmark_validation_metadata(benchmark_summary),
            },
            run_id=telemetry_run_id,
        )
        finish_run(telemetry_run_id, status="completed")
    except Exception as exc:
        finish_run(telemetry_run_id, status="failed", error=str(exc))
        raise
    print(f"Release validation manifest: {summary['manifest']}")
    print(f"Benchmark bundle: {summary['benchmarks_dir']}")
