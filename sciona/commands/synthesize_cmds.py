"""Commands for assembly, synthesis, and export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sciona.commands._helpers import (
    _create_llm_router,
    _create_proof_env,
    _create_shared_context,
    _print_mode_summary,
    _print_prompt_routing_summary,
    _print_shared_context_metrics,
    _warm_llm_if_supported,
    _write_shared_context_metrics_file,
)


async def _cmd_assemble(args: argparse.Namespace) -> None:
    """Assemble CDG + match results into a compilable skeleton."""
    from sciona.architect.handoff import load_json
    from sciona.services import (
        SynthesizerAssembleRequest,
        SynthesizerCompileRequest,
        SynthesizerService,
    )
    from sciona.synthesizer.assembler import AssemblyError
    from sciona.types import MatchResult, Prover

    # Load CDG
    cdg_path = Path(args.cdg_file)
    if not cdg_path.exists():
        print(f"Error: CDG file not found: {cdg_path}", file=sys.stderr)
        sys.exit(1)
    cdg = load_json(cdg_path)

    # Load match results
    matches_path = Path(args.matches_file)
    if not matches_path.exists():
        print(f"Error: matches file not found: {matches_path}", file=sys.stderr)
        sys.exit(1)
    with open(matches_path) as f:
        matches_data = json.load(f)
    if not isinstance(matches_data, list):
        print("Error: matches file must contain a JSON array", file=sys.stderr)
        sys.exit(1)
    match_results = [MatchResult.from_dict(d) for d in matches_data]

    prover = Prover(args.prover)
    service = SynthesizerService(prover=prover)

    # Assemble
    try:
        skeleton = service.assemble(
            SynthesizerAssembleRequest(cdg=cdg, match_results=match_results)
        ).skeleton
    except AssemblyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if prover == Prover.LEAN4:
        ext = ".lean"
    elif prover == Prover.PYTHON:
        ext = ".py"
    else:
        ext = ".v"
    output = args.output or (cdg_path.stem + "_skeleton" + ext)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(skeleton.source_code)

    print(f"Skeleton written to {output_path}")
    print(f"  Units: {len(skeleton.units)}, Sorry count: {skeleton.sorry_count}")

    # Optional compilation check
    if args.check:
        from sciona.config import AgeomConfig

        config = AgeomConfig()
        env = _create_proof_env(prover, config)

        try:
            result = (
                await service.compile(
                    SynthesizerCompileRequest(skeleton=skeleton, env=env)
                )
            ).result
            if result.compiled_ok:
                print("  Compilation: OK")
            else:
                print("  Compilation: FAILED")
                if result.feedback:
                    for err in result.feedback.errors:
                        print(f"    {err}")
        finally:
            await env.close()


async def _cmd_export(args: argparse.Namespace) -> None:
    """Export verified source to compiled artifacts and FFI bindings."""
    from sciona.config import AgeomConfig
    from sciona.synthesizer.extractor import ExportTarget, Extractor
    from sciona.synthesizer.models import SkeletonFile, SynthesisResult

    config = AgeomConfig()

    source_path = Path(args.source_file)
    if not source_path.exists():
        print(f"Error: source file not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    # Try loading as SynthesisResult JSON, else treat as raw source
    synthesis_result: SynthesisResult
    if source_path.suffix == ".json":
        with open(source_path) as f:
            data = json.load(f)
        synthesis_result = SynthesisResult(**data)
    else:
        source_code = source_path.read_text()
        skeleton = SkeletonFile(
            prover=args.prover,
            source_code=source_code,
        )
        synthesis_result = SynthesisResult(
            skeleton=skeleton,
            compiled_ok=True,
        )

    # Optional optimizer
    if args.optimize or config.optimize_by_default:
        from sciona.synthesizer.optimizer import Optimizer

        optimizer = Optimizer()
        candidates = optimizer.scan(synthesis_result.skeleton)
        if candidates:
            print(f"Optimizer found {len(candidates)} candidate(s) for hot-path swap")
            # Verify guards (without a real env, just apply comment-guards)
            verified = [c for c in candidates if c.rule.guard_check.startswith("--")]
            for c in verified:
                c.guard_verified = True
            if verified:
                synthesis_result.skeleton = optimizer.apply(
                    synthesis_result.skeleton, verified
                )
                print(f"  Applied {len(verified)} optimization(s)")

    target = ExportTarget(args.target)
    output_dir = Path(args.output_dir) if args.output_dir else config.export_output_dir

    extractor = Extractor(config)
    print(f"Exporting to {target.value} in {output_dir}/...")
    bundle = await extractor.extract(synthesis_result, target, output_dir)

    print("\nExport complete:")
    print(f"  Target: {bundle.target}")
    print(f"  Source: {bundle.source_path}")
    if bundle.compiled_artifact:
        print(f"  Artifact: {bundle.compiled_artifact}")
    if bundle.ffi_files:
        print("  FFI files:")
        for f in bundle.ffi_files:
            print(f"    {f}")
    if bundle.certificate:
        print(f"  Certificate: {output_dir / 'certificate.json'}")
        print(f"    Source hash: {bundle.certificate.source_hash[:16]}...")
    if bundle.errors:
        print("  Errors:")
        for err in bundle.errors:
            print(f"    {err}")


async def _cmd_synthesize(args: argparse.Namespace) -> None:
    """Assemble CDG + match results, then repair via the synthesizer agent."""
    from sciona.architect.handoff import load_json
    from sciona.config import AgeomConfig, resolve_execution_mode
    from sciona.services import (
        SynthesizerAssembleRequest,
        SynthesizerCompileRequest,
        SynthesizerRepairRequest,
        SynthesizerService,
    )
    from sciona.synthesizer.agent import SynthesizerAgent
    from sciona.synthesizer.assembler import AssemblyError
    from sciona.types import MatchResult, Prover

    config = AgeomConfig()
    mode_settings = resolve_execution_mode(config, getattr(args, "mode", None))
    _print_mode_summary("synthesize", mode_settings)

    # Load CDG
    cdg_path = Path(args.cdg_file)
    if not cdg_path.exists():
        print(f"Error: CDG file not found: {cdg_path}", file=sys.stderr)
        sys.exit(1)
    cdg = load_json(cdg_path)

    # Load match results
    matches_path = Path(args.matches_file)
    if not matches_path.exists():
        print(f"Error: matches file not found: {matches_path}", file=sys.stderr)
        sys.exit(1)
    with open(matches_path) as f:
        matches_data = json.load(f)
    if not isinstance(matches_data, list):
        print("Error: matches file must contain a JSON array", file=sys.stderr)
        sys.exit(1)
    match_results = [MatchResult.from_dict(d) for d in matches_data]

    prover = Prover(args.prover)
    base_service = SynthesizerService(prover=prover)

    # Phase 1: Assemble
    try:
        skeleton = base_service.assemble(
            SynthesizerAssembleRequest(cdg=cdg, match_results=match_results)
        ).skeleton
    except AssemblyError as exc:
        print(f"Error assembling skeleton: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Assembled skeleton: {len(skeleton.units)} units, {skeleton.sorry_count} sorrys"
    )

    # Set up ProofEnvironment
    env = _create_proof_env(prover, config)

    try:
        compile_result = await base_service.compile(
            SynthesizerCompileRequest(skeleton=skeleton, env=env)
        )
        if compile_result.result.compiled_ok:
            from sciona.synthesizer.models import SynthesisResult

            result = SynthesisResult(
                skeleton=compile_result.result.skeleton,
                compiled_ok=True,
                sorry_remaining=compile_result.result.skeleton.sorry_count,
                patches_applied=0,
                iterations_used=0,
            )
            synth_shared_metrics = None
        else:
            # Set up LLM only when repair is required.
            try:
                from sciona.llm_router import SYNTHESIZER_REPAIR, SYNTHESIZER_TACTIC

                prompt_keys = [
                    SYNTHESIZER_REPAIR,
                    SYNTHESIZER_TACTIC,
                ]
                _print_prompt_routing_summary(
                    config, "synthesizer", prompt_keys, getattr(args, "mode", None)
                )
                llm = _create_llm_router(args, config, "synthesizer", prompt_keys)
                await _warm_llm_if_supported(llm, "synthesizer")
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)
            except ImportError as exc:
                print(f"Error: missing LLM dependency ({exc})", file=sys.stderr)
                sys.exit(1)

            import uuid

            max_iterations = args.max_iterations or config.synthesizer_max_iterations
            synth_run_id = uuid.uuid4().hex
            synth_shared_context, synth_shared_metrics = await _create_shared_context(
                config,
                enabled=mode_settings.synthesizer_shared_context_enabled,
            )

            agent = SynthesizerAgent(
                env=env,
                llm=llm,
                max_iterations=max_iterations,
                shared_context=synth_shared_context,
                shared_context_metrics=synth_shared_metrics,
                context_namespace=f"synthesizer/{synth_run_id}",
                context_budget_chars=config.synthesizer_shared_context_budget_chars,
            )
            service = SynthesizerService(prover=prover, repair_agent=agent)
            print(f"Starting repair loop (max {max_iterations} iterations)...")
            result = (
                await service.repair(SynthesizerRepairRequest(skeleton=skeleton))
            ).result

        # Output
        if prover == Prover.LEAN4:
            ext = ".lean"
        elif prover == Prover.PYTHON:
            ext = ".py"
        else:
            ext = ".v"
        output = args.output or (cdg_path.stem + "_verified" + ext)
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result.skeleton.source_code)

        print(f"\nResult written to {output_path}")
        print(f"  Compiled OK: {result.compiled_ok}")
        print(f"  Iterations used: {result.iterations_used}")
        print(f"  Patches applied: {result.patches_applied}")
        print(f"  Sorry remaining: {result.sorry_remaining}")
        if synth_shared_metrics is not None:
            _print_shared_context_metrics("synthesizer", synth_shared_metrics)
            metrics_path = _write_shared_context_metrics_file(
                output_path.parent / "shared_context_metrics.json",
                {"synthesizer": synth_shared_metrics},
            )
            if metrics_path is not None:
                print(f"  Shared context metrics: {metrics_path}")

        if result.error_history:
            print("  Errors encountered:")
            for it, cat, text in result.error_history:
                print(f"    [{it}] {cat}: {text[:80]}")

        # Write trace if requested
        if getattr(args, "trace", False):
            from sciona.telemetry import get_event_log

            trace_path = output_path.parent / "trace.jsonl"
            event_log = get_event_log()
            if len(event_log) > 0:
                event_log.save(trace_path)
                print(f"  Trace: {trace_path} ({len(event_log)} events)")
    finally:
        await env.close()
