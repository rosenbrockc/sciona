"""Execution sandbox for benchmarking instrumented synthesiser artifacts."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import sys

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
        self._python_executable = (
            os.environ.get("AGEOM_PYTHON_PATH")
            or sys.executable
            or "python"
        )

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
        artifact = bundle.executable_artifact or bundle.compiled_artifact or bundle.source_path
        output_dir = bundle.output_dir.resolve()
        if not artifact.is_absolute():
            artifact = artifact.resolve()
        if not artifact.exists():
            logger.error("Artifact not found: %s", artifact)
            return BenchmarkResult(global_loss=_FAILURE_PENALTY)

        trace_path = output_dir / "trace.jsonl"
        # Remove stale trace from a previous run
        if trace_path.exists():
            trace_path.unlink()

        try:
            proc = await asyncio.create_subprocess_exec(
                self._python_executable,
                str(artifact),
                str(dataset_path),
                cwd=str(output_dir),
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
            telemetry = _parse_trace(trace_path)
            logger.error(
                "Artifact exited with code %d: %s",
                proc.returncode,
                (stderr or b"").decode(errors="replace")[:500],
            )
            return BenchmarkResult(
                global_loss=_FAILURE_PENALTY,
                node_telemetry=telemetry,
            )

        # Parse trace.jsonl
        telemetry = _parse_trace(trace_path)

        # Compute global loss from the chosen metric
        global_loss = _compute_loss(telemetry, metric, stdout)

        return BenchmarkResult(
            global_loss=global_loss,
            node_telemetry=telemetry,
        )

    async def evaluate_adapter(
        self,
        bundle: ExportBundle,
        adapter_path: str,
        metric: OptimizationMetric,
        *,
        user: str | None = None,
        serial: str | None = None,
        varset: dict | None = None,
    ) -> BenchmarkResult:
        """Execute *bundle* against a templated adapter dataset.

        Loads multi-group sensor data from an ``adapter.yml`` file, writes
        each group to a parquet file in the bundle's output directory, and
        produces a manifest JSON that the artifact can consume.

        Args:
            bundle: The export bundle containing the compiled artifact.
            adapter_path: Path to the ``adapter.yml`` template file.
            metric: Which optimisation axis to compute the global loss for.
            user: Optional user filter for the dataset collection.
            serial: Optional device serial filter.
            varset: Optional variable substitutions for the adapter template.

        Returns:
            A ``BenchmarkResult`` from the underlying :meth:`evaluate` call.
        """
        from ageom.principal.datasets import create_templated_dataset_collection

        adapter = Path(adapter_path).expanduser()
        if not adapter.exists():
            logger.error("Adapter file not found: %s", adapter)
            return BenchmarkResult(global_loss=_FAILURE_PENALTY)

        executable = bundle.executable_artifact or bundle.compiled_artifact
        if executable is not None and executable.suffix == ".py":
            return await self._evaluate_python_adapter_runner(
                bundle,
                adapter,
                metric,
                varset=varset,
                user=user,
                serial=serial,
            )

        try:
            coll_cls = create_templated_dataset_collection(
                str(adapter), varset=varset,
            )
            options = coll_cls.get_filter_options(user, serial, recursive=True)
            coll = coll_cls.from_folder(options=options)
            dfs = coll.to_pandas()
        except Exception as exc:
            logger.error("Failed to load adapter dataset: %s", exc)
            return BenchmarkResult(global_loss=_FAILURE_PENALTY)

        # Write each sensor group as a parquet file.
        manifest: dict[str, str] = {}
        out_dir = bundle.output_dir
        for group_name, df in dfs.items():
            parquet_path = out_dir / f"{group_name}.parquet"
            df.to_parquet(parquet_path, index=False)
            manifest[group_name] = str(parquet_path)

        manifest_path = out_dir / "dataset_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        return await self.evaluate(bundle, str(manifest_path), metric)

    async def _evaluate_python_adapter_runner(
        self,
        bundle: ExportBundle,
        adapter: Path,
        metric: OptimizationMetric,
        *,
        varset: dict | None = None,
        user: str | None = None,
        serial: str | None = None,
    ) -> BenchmarkResult:
        artifact = bundle.executable_artifact or bundle.compiled_artifact or bundle.source_path
        output_dir = bundle.output_dir.resolve()
        if not artifact.is_absolute():
            artifact = artifact.resolve()
        if not artifact.exists():
            logger.error("Artifact not found: %s", artifact)
            return BenchmarkResult(global_loss=_FAILURE_PENALTY)

        trace_path = output_dir / "trace.jsonl"
        if trace_path.exists():
            trace_path.unlink()

        cmd = [
            self._python_executable,
            str(artifact),
            "--dataset-root",
            str(adapter.parent),
            "--trace-path",
            str(trace_path),
        ]
        for key, value in sorted((varset or {}).items()):
            cmd.extend(["--dataset-var", f"{key}={value}"])
        if user is not None:
            cmd.extend(["--user", user])
        if serial is not None:
            cmd.extend(["--serial", serial])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(output_dir),
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
            telemetry = _parse_trace(trace_path)
            logger.error(
                "Artifact exited with code %d: %s",
                proc.returncode,
                (stderr or b"").decode(errors="replace")[:500],
            )
            return BenchmarkResult(
                global_loss=_FAILURE_PENALTY,
                node_telemetry=telemetry,
            )

        telemetry = _parse_trace(trace_path)
        global_loss = _compute_loss(telemetry, metric, stdout)
        return BenchmarkResult(global_loss=global_loss, node_telemetry=telemetry)


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
