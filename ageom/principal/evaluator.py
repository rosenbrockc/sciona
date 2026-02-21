"""Execution sandbox for benchmarking instrumented synthesiser artifacts."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from ageom.principal.models import BenchmarkResult, NodeTelemetry, OptimizationMetric
from ageom.synthesizer.models import ExportBundle

logger = logging.getLogger(__name__)

# Penalty loss returned when the artifact fails to execute.
_FAILURE_PENALTY: float = 1e12

# Default subprocess timeout in seconds.
_DEFAULT_TIMEOUT_S: float = 120.0


class ExecutionSandbox:
    """Run an instrumented artifact and collect telemetry.

    The sandbox executes the compiled Python artifact as a subprocess,
    feeds it the user-provided benchmark dataset, and parses the
    ``trace.jsonl`` file emitted by the ``@ageom_probe`` instrumentation.
    """

    def __init__(self, *, timeout_s: float = _DEFAULT_TIMEOUT_S) -> None:
        self._timeout_s = timeout_s

    async def evaluate(
        self,
        bundle: ExportBundle,
        dataset_path: str,
        metric: OptimizationMetric,
    ) -> BenchmarkResult:
        """Execute *bundle* against *dataset_path* and return telemetry.

        Args:
            bundle: The export bundle containing the compiled artifact.
            dataset_path: Path to a JSON or CSV benchmark dataset.
            metric: Which optimisation axis to compute the global loss for.

        Returns:
            A ``BenchmarkResult`` populated from the trace file.  On
            subprocess failure or timeout the result carries a large
            penalty loss and an empty telemetry map.
        """
        artifact = bundle.compiled_artifact or bundle.source_path
        if not artifact.exists():
            logger.error("Artifact not found: %s", artifact)
            return BenchmarkResult(global_loss=_FAILURE_PENALTY)

        trace_path = bundle.output_dir / "trace.jsonl"
        # Remove stale trace from a previous run
        if trace_path.exists():
            trace_path.unlink()

        try:
            proc = await asyncio.create_subprocess_exec(
                "python",
                str(artifact),
                str(dataset_path),
                cwd=str(bundle.output_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._timeout_s,
            )
        except asyncio.TimeoutError:
            logger.error("Artifact timed out after %.1fs", self._timeout_s)
            proc.kill()
            await proc.wait()
            return BenchmarkResult(global_loss=_FAILURE_PENALTY)
        except OSError as exc:
            logger.error("Failed to launch artifact: %s", exc)
            return BenchmarkResult(global_loss=_FAILURE_PENALTY)

        if proc.returncode != 0:
            logger.error(
                "Artifact exited with code %d: %s",
                proc.returncode,
                (stderr or b"").decode(errors="replace")[:500],
            )
            return BenchmarkResult(global_loss=_FAILURE_PENALTY)

        # Parse trace.jsonl
        telemetry = _parse_trace(trace_path)

        # Compute global loss from the chosen metric
        global_loss = _compute_loss(telemetry, metric, stdout)

        return BenchmarkResult(
            global_loss=global_loss,
            node_telemetry=telemetry,
        )


def _parse_trace(trace_path: Path) -> dict[str, NodeTelemetry]:
    """Read the JSON-lines trace file into a ``NodeTelemetry`` map."""
    telemetry: dict[str, NodeTelemetry] = {}
    if not trace_path.exists():
        logger.warning("Trace file not found: %s", trace_path)
        return telemetry

    with open(trace_path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Malformed trace line %d: %s", lineno, line[:120])
                continue
            node_id = record.get("node_id", "")
            if not node_id:
                continue
            telemetry[node_id] = NodeTelemetry(
                node_id=node_id,
                execution_time_ms=float(record.get("execution_time_ms", 0.0)),
                peak_memory_bytes=int(record.get("peak_memory_bytes", 0)),
                error_expansion=float(record.get("error_expansion", 0.0)),
            )
    return telemetry


def _compute_loss(
    telemetry: dict[str, NodeTelemetry],
    metric: OptimizationMetric,
    stdout: bytes | None,
) -> float:
    """Derive a scalar loss from telemetry and the requested metric.

    For LATENCY the loss is total execution time across all nodes.
    For MEMORY it is the maximum peak memory across nodes.
    For PRECISION it uses stdout as a carrier for MSE (the artifact is
    expected to print a JSON object with a ``"mse"`` key).
    For FLOP_COUNT we fall back to LATENCY as a proxy until hardware
    counters are integrated.
    """
    if not telemetry:
        return _FAILURE_PENALTY

    if metric == OptimizationMetric.LATENCY:
        return sum(t.execution_time_ms for t in telemetry.values())

    if metric == OptimizationMetric.MEMORY:
        return float(max(t.peak_memory_bytes for t in telemetry.values()))

    if metric == OptimizationMetric.PRECISION:
        return _parse_mse_from_stdout(stdout)

    # FLOP_COUNT — proxy via latency until HW counters land
    return sum(t.execution_time_ms for t in telemetry.values())


def _parse_mse_from_stdout(stdout: bytes | None) -> float:
    """Extract MSE from artifact stdout.

    Expects the last non-empty line to be a JSON object containing
    ``{"mse": <float>}``.  Returns ``_FAILURE_PENALTY`` on parse failure.
    """
    if not stdout:
        return _FAILURE_PENALTY
    lines = stdout.decode(errors="replace").strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict) and "mse" in data:
                return float(data["mse"])
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return _FAILURE_PENALTY
