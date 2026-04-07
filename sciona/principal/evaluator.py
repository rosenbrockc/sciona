"""Execution sandbox for benchmarking instrumented synthesiser artifacts."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

from sciona.principal.models import BenchmarkResult, NodeTelemetry, OptimizationMetric
from sciona.principal.runtime_context import summarize_runtime_evidence
from sciona.synthesizer.models import ExportBundle

logger = logging.getLogger(__name__)

# Penalty loss returned when the artifact fails to execute.
_FAILURE_PENALTY: float = 1e12

# Default subprocess timeout in seconds.
_DEFAULT_TIMEOUT_S: float = 120.0


class ExecutionSandbox:
    """Run an instrumented artifact and collect telemetry.

    The sandbox executes the compiled Python artifact as a subprocess,
    feeds it the user-provided benchmark dataset, and parses the
    ``trace.jsonl`` file emitted by the ``@sciona_probe`` instrumentation.
    """

    def __init__(self, *, timeout_s: float = _DEFAULT_TIMEOUT_S) -> None:
        self._timeout_s = timeout_s
        self._python_executable = (
            os.environ.get("SCIONA_PYTHON_PATH")
            or sys.executable
            or "python"
        )

    async def evaluate(
        self,
        bundle: ExportBundle,
        dataset_path: str,
        metric: OptimizationMetric,
        *,
        evaluation_spec: dict[str, Any] | str | None = None,
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
            cmd = [self._python_executable, str(artifact), str(dataset_path)]
            eval_spec_arg = _materialize_evaluation_spec_arg(output_dir, evaluation_spec)
            if eval_spec_arg is not None:
                cmd.extend(["--eval-spec", eval_spec_arg])
            if bundle.parameter_assignments:
                params_path = output_dir / "params.json"
                params_path.write_text(json.dumps(bundle.parameter_assignments, indent=2))
                cmd.extend(["--params", str(params_path)])
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

        stdout_payload = _parse_stdout_payload(stdout)
        runtime_artifacts = _build_runtime_artifacts(
            trace_path=trace_path,
            stdout_payload=stdout_payload,
            signal_data=_load_signal_data_from_dataset_path(dataset_path),
        )

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
                runtime_artifacts=runtime_artifacts,
            )

        # Parse trace.jsonl
        telemetry = _parse_trace(trace_path)

        # Compute global loss from the chosen metric
        global_loss = _compute_loss(telemetry, metric, stdout)

        return BenchmarkResult(
            global_loss=global_loss,
            node_telemetry=telemetry,
            runtime_artifacts=runtime_artifacts,
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
        evaluation_spec: dict[str, Any] | str | None = None,
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
        from sciona.principal.datasets import create_templated_dataset_collection

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
                evaluation_spec=evaluation_spec,
            )

        signal_data: dict[str, Any] = {}
        try:
            coll_cls = create_templated_dataset_collection(
                str(adapter), varset=varset,
            )
            options = coll_cls.get_filter_options(user, serial, recursive=True)
            coll = coll_cls.from_folder(options=options)
            dfs = coll.to_pandas()
            signal_data = _collect_signal_data_from_frames(dfs)
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

        result = await self.evaluate(
            bundle,
            str(manifest_path),
            metric,
            evaluation_spec=evaluation_spec,
        )
        artifacts = dict(result.runtime_artifacts)
        if signal_data:
            artifacts["signal_data"] = signal_data
        return result.model_copy(update={"runtime_artifacts": artifacts})

    async def _evaluate_python_adapter_runner(
        self,
        bundle: ExportBundle,
        adapter: Path,
        metric: OptimizationMetric,
        *,
        varset: dict | None = None,
        user: str | None = None,
        serial: str | None = None,
        evaluation_spec: dict[str, Any] | str | None = None,
    ) -> BenchmarkResult:
        artifact = bundle.executable_artifact or bundle.compiled_artifact or bundle.source_path
        output_dir = bundle.output_dir.resolve()
        if not artifact.is_absolute():
            artifact = artifact.resolve()
        if not artifact.exists():
            logger.error("Artifact not found: %s", artifact)
            return BenchmarkResult(global_loss=_FAILURE_PENALTY)

        signal_data: dict[str, Any] = {}
        try:
            from sciona.principal.datasets import create_templated_dataset_collection

            coll_cls = create_templated_dataset_collection(str(adapter), varset=varset)
            options = coll_cls.get_filter_options(user, serial, recursive=True)
            coll = coll_cls.from_folder(options=options)
            signal_data = _collect_signal_data_from_frames(coll.to_pandas())
        except Exception:
            signal_data = {}

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
        eval_spec_arg = _materialize_evaluation_spec_arg(output_dir, evaluation_spec)
        if eval_spec_arg is not None:
            cmd.extend(["--eval-spec", eval_spec_arg])
        if bundle.parameter_assignments:
            params_path = output_dir / "params.json"
            params_path.write_text(json.dumps(bundle.parameter_assignments, indent=2))
            cmd.extend(["--params", str(params_path)])

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

        stdout_payload = _parse_stdout_payload(stdout)
        runtime_artifacts = _build_runtime_artifacts(
            trace_path=trace_path,
            stdout_payload=stdout_payload,
            signal_data=signal_data,
        )

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
                runtime_artifacts=runtime_artifacts,
            )

        telemetry = _parse_trace(trace_path)
        global_loss = _compute_loss(telemetry, metric, stdout)
        return BenchmarkResult(
            global_loss=global_loss,
            node_telemetry=telemetry,
            runtime_artifacts=runtime_artifacts,
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
        return _parse_precision_loss_from_stdout(stdout)

    # FLOP_COUNT — proxy via latency until HW counters land
    return sum(t.execution_time_ms for t in telemetry.values())


def _parse_precision_loss_from_stdout(stdout: bytes | None) -> float:
    """Extract precision loss from artifact stdout.

    Expects the last non-empty line to be a JSON object containing
    a scalar loss field. Prefers ``loss``, then ``rmse``, then ``mse``.
    Returns ``_FAILURE_PENALTY`` on parse failure.
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
            if isinstance(data, dict):
                for key in ("loss", "rmse", "mse"):
                    if key in data:
                        return float(data[key])
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return _FAILURE_PENALTY


def _parse_stdout_payload(stdout: bytes | None) -> dict[str, Any] | None:
    """Return the last JSON object emitted to stdout, when present."""
    if not stdout:
        return None
    lines = stdout.decode(errors="replace").strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    return None


def _build_runtime_artifacts(
    *,
    trace_path: Path,
    stdout_payload: dict[str, Any] | None,
    signal_data: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble best-effort runtime artifacts for downstream expansion."""
    artifacts: dict[str, Any] = {"trace_path": str(trace_path)}
    outputs: dict[str, Any] = {}
    intermediates: dict[str, Any] = {}
    if stdout_payload:
        artifacts["stdout_payload"] = stdout_payload
        stdout_intermediates = stdout_payload.get("intermediates", {})
        if isinstance(stdout_intermediates, dict):
            intermediates = dict(stdout_intermediates)
            artifacts["intermediates"] = dict(stdout_intermediates)
        raw_outputs = stdout_payload.get("outputs")
        if isinstance(raw_outputs, dict):
            outputs = dict(raw_outputs)
        if isinstance(outputs, dict):
            merged = dict(artifacts.get("intermediates", {}))
            for key, value in outputs.items():
                if key not in merged:
                    merged[key] = value
            artifacts["intermediates"] = merged
    if signal_data:
        artifacts["signal_data"] = dict(signal_data)
    evidence = summarize_runtime_evidence(
        dict(signal_data or {}),
        intermediates=intermediates,
        outputs=outputs,
    )
    artifacts.update(evidence)
    _persist_runtime_evidence(trace_path, evidence)
    return artifacts


def _persist_runtime_evidence(trace_path: Path, evidence: dict[str, Any]) -> None:
    """Persist the compact runtime evidence contract beside the trace file."""
    try:
        payload = {
            "trace_path": str(trace_path),
            "runtime_context": evidence.get("runtime_context", {}),
            "canonical_runtime_context": evidence.get("canonical_runtime_context", {}),
            "telemetry_summary": evidence.get("telemetry_summary", {}),
        }
        (trace_path.parent / "runtime_evidence.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True)
        )
    except Exception:
        logger.debug(
            "Failed to persist runtime evidence beside %s",
            trace_path,
            exc_info=True,
        )


def _load_signal_data_from_dataset_path(dataset_path: str) -> dict[str, Any]:
    """Best-effort signal-data loading from a dataset manifest path."""
    candidate = Path(dataset_path).expanduser()
    if not candidate.exists() or candidate.suffix.lower() != ".json":
        return {}
    try:
        payload = json.loads(candidate.read_text())
    except Exception:
        return {}
    if not isinstance(payload, dict) or not payload:
        return {}
    try:
        import pandas as pd
    except Exception:
        return {}
    frames: dict[str, Any] = {}
    for group_name, raw_path in payload.items():
        if not isinstance(group_name, str) or not isinstance(raw_path, str):
            continue
        path = Path(raw_path).expanduser()
        if not path.exists() or path.suffix.lower() != ".parquet":
            continue
        try:
            frames[group_name] = pd.read_parquet(path)
        except Exception:
            continue
    return _collect_signal_data_from_frames(frames)


def _collect_signal_data_from_frames(group_frames: dict[str, Any]) -> dict[str, Any]:
    """Extract signal-oriented arrays and sampling metadata from dataset frames."""
    if not isinstance(group_frames, dict):
        return {}

    signal_data: dict[str, Any] = {}
    for group_name, frame in group_frames.items():
        if not hasattr(frame, "columns"):
            continue
        group = str(group_name)
        group_lower = group.lower()
        group_alias = group.rsplit("_", 1)[-1] if "_" in group else group
        time_column = _pick_time_column(frame)
        sampling_rate = _infer_sampling_rate(frame, time_column=time_column)
        if sampling_rate is not None:
            signal_data.setdefault("sampling_rate", sampling_rate)
            signal_data[f"{group}_sampling_rate"] = sampling_rate
            if group_alias:
                signal_data[f"{group_alias}_sampling_rate"] = sampling_rate

        for column in frame.columns:
            series = frame[column]
            values = series.to_numpy() if hasattr(series, "to_numpy") else list(series)
            key = str(column)
            signal_data[key] = values
            if key.startswith(f"{group}_"):
                alias = key[len(group) + 1 :]
                signal_data.setdefault(alias, values)
                if group_alias and alias != "value":
                    signal_data.setdefault(f"{group_alias}_{alias}", values)
                if group_alias and alias == "value":
                    signal_data.setdefault(group_alias, values)
            if key == "value" and any(
                token in group_lower for token in ("signal", "wave", "waveform", "ecg", "ppg", "eeg", "emg")
            ):
                signal_data.setdefault("signal", values)
            if time_column is not None and key == time_column:
                signal_data.setdefault("time", values)

    return signal_data


def _pick_time_column(frame: Any) -> str | None:
    columns = [str(column) for column in getattr(frame, "columns", [])]
    if "t" in columns:
        return "t"
    for column in columns:
        if column.endswith("_t"):
            return column
    return None


def _infer_sampling_rate(frame: Any, *, time_column: str | None = None) -> float | None:
    """Estimate sampling rate from a frame time column using median delta."""
    time_column = time_column or _pick_time_column(frame)
    if time_column is None:
        return None
    times = frame[time_column]
    values = times.to_numpy() if hasattr(times, "to_numpy") else times
    if values is None or len(values) < 2:
        return None
    diffs: list[float] = []
    prev: float | None = None
    for raw in values:
        try:
            current = float(raw)
        except (TypeError, ValueError):
            prev = None
            continue
        if prev is not None:
            delta = current - prev
            if delta > 0:
                diffs.append(delta)
        prev = current
    if not diffs:
        return None
    diffs.sort()
    median = diffs[len(diffs) // 2]
    if median <= 0:
        return None
    return 1.0 / median


def _materialize_evaluation_spec_arg(
    output_dir: Path,
    evaluation_spec: dict[str, Any] | str | None,
) -> str | None:
    if evaluation_spec is None:
        return None
    if isinstance(evaluation_spec, dict):
        target = output_dir / "evaluation_spec.json"
        target.write_text(json.dumps(evaluation_spec, indent=2) + "\n")
        return str(target)
    candidate = Path(evaluation_spec).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    try:
        payload = json.loads(evaluation_spec)
    except json.JSONDecodeError:
        return str(candidate)
    target = output_dir / "evaluation_spec.json"
    target.write_text(json.dumps(payload, indent=2) + "\n")
    return str(target)
