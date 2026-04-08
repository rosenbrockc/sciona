"""Counterfactual attribution for reference-based loss objectives."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from sciona.architect.handoff import CDGExport
from sciona.architect.models import NodeStatus
from sciona.principal.eval_spec import compute_evaluation_payload, load_evaluation_spec
from sciona.principal.evaluator import (
    _build_runtime_artifacts,
    _collect_runtime_inputs_from_frames,
)
from sciona.principal.models import NodeGradient, OptimizationMetric
from sciona.synthesizer.models import ExportBundle


def is_reference_loss_objective(
    metric: OptimizationMetric,
    evaluation_spec: dict[str, Any] | str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return the normalized evaluation spec when the objective is reference-based."""
    if metric != OptimizationMetric.PRECISION or evaluation_spec is None:
        return None, None
    payload = (
        dict(evaluation_spec)
        if isinstance(evaluation_spec, dict)
        else load_evaluation_spec(evaluation_spec)
    )
    if not isinstance(payload, dict):
        return None, None
    loss_name = str(payload.get("loss", payload.get("metric", ""))).lower()
    if loss_name not in {"rmse", "mse", "mae"}:
        return None, None
    return payload, loss_name


async def compute_reference_loss_gradients(
    cdg: CDGExport,
    bundle: ExportBundle,
    dataset_path: str,
    evaluation_spec: dict[str, Any] | str | None,
    *,
    dataset_varset: dict[str, str] | None = None,
    dataset_slice_start_s: float | None = None,
    dataset_slice_stop_s: float | None = None,
) -> list[NodeGradient]:
    """Estimate per-node attribution by perturbing exported atom outputs."""
    spec, loss_name = is_reference_loss_objective(
        OptimizationMetric.PRECISION,
        evaluation_spec,
    )
    if spec is None or loss_name is None:
        return []

    dataset_file = Path(dataset_path).expanduser().resolve()
    if dataset_file.suffix not in {".yml", ".yaml"}:
        return []

    with _load_exported_modules(bundle) as loaded:
        if loaded is None:
            return []
        package_name, atoms_mod, pipeline_mod = loaded
        del package_name  # not needed after import side effects

        node_to_function = _extract_traced_node_functions(atoms_mod)
        atomic_nodes = {
            node.node_id: node.name
            for node in cdg.nodes
            if node.status == NodeStatus.ATOMIC and node.node_id in node_to_function
        }
        if not atomic_nodes:
            return []

        entrypoint = getattr(pipeline_mod, "DEFAULT_ENTRYPOINT", None)
        group_frames = _load_dataset_with_optional_slice(
            pipeline_mod,
            dataset_file.parent,
            dataset_varset=dataset_varset,
            entrypoint=entrypoint,
            spec=spec,
            dataset_slice_start_s=dataset_slice_start_s,
            dataset_slice_stop_s=dataset_slice_stop_s,
        )
        flat_inputs = pipeline_mod._flatten_inputs(group_frames)
        baseline_result = pipeline_mod.run_pipeline(entrypoint=entrypoint, **group_frames)
        try:
            runtime_inputs = (
                _collect_runtime_inputs_from_frames(group_frames)
                if isinstance(group_frames, dict)
                else {}
            )
            if not runtime_inputs and isinstance(flat_inputs, dict):
                runtime_inputs = _collect_runtime_inputs_from_runtime_inputs(flat_inputs)
        except Exception:
            runtime_inputs = {}
        runtime_artifacts = _build_runtime_artifacts(
            trace_path=bundle.output_dir / "trace.jsonl",
            stdout_payload=baseline_result if isinstance(baseline_result, dict) else None,
            runtime_inputs=runtime_inputs,
        )
        profile_artifacts = {
            key: runtime_artifacts[key]
            for key in (
                "trace_path",
                "runtime_context",
                "canonical_runtime_context",
                "telemetry_summary",
            )
            if key in runtime_artifacts
        }
        if profile_artifacts:
            try:
                (bundle.output_dir / "profile_runtime_artifacts.json").write_text(
                    json.dumps(profile_artifacts, indent=2)
                )
            except Exception:
                pass
        baseline_payload = compute_evaluation_payload(baseline_result, flat_inputs, spec)
        baseline_loss = float(baseline_payload["loss"])

        deltas: dict[str, float] = {}
        reasons: dict[str, str] = {}
        for node_id, node_name in atomic_nodes.items():
            function_name = node_to_function[node_id]
            original = getattr(atoms_mod, function_name, None)
            if original is None:
                continue

            def _patched(*args: Any, __orig: Any = original, **kwargs: Any) -> Any:
                return _perturb_value(__orig(*args, **kwargs))

            setattr(atoms_mod, function_name, _patched)
            try:
                perturbed_result = pipeline_mod.run_pipeline(entrypoint=entrypoint, **group_frames)
                perturbed_payload = compute_evaluation_payload(
                    perturbed_result,
                    flat_inputs,
                    spec,
                )
                perturbed_loss = float(perturbed_payload["loss"])
                delta = abs(perturbed_loss - baseline_loss)
                if delta > 0:
                    deltas[node_id] = delta
                    reasons[node_id] = (
                        f"Node '{node_name}' changed {loss_name} by {delta:.4f} "
                        f"(baseline {baseline_loss:.4f} -> perturbed {perturbed_loss:.4f}) "
                        f"under a small output perturbation"
                    )
            except Exception:
                deltas[node_id] = max(abs(baseline_loss), 1.0) * 10.0
                reasons[node_id] = (
                    f"Node '{node_name}' caused {loss_name} evaluation to fail "
                    f"under a small output perturbation"
                )
            finally:
                setattr(atoms_mod, function_name, original)

        total = sum(deltas.values())
        if total <= 0:
            return []

        gradients = [
            NodeGradient(
                node_id=node_id,
                gradient_score=(delta / total) * 100.0,
                metric_type=OptimizationMetric.PRECISION,
                bottleneck_reason=reasons[node_id],
            )
            for node_id, delta in deltas.items()
        ]
        gradients.sort(key=lambda item: item.gradient_score, reverse=True)
    return gradients


def _load_dataset_with_optional_slice(
    pipeline_mod: Any,
    dataset_root: Path,
    *,
    dataset_varset: dict[str, str] | None,
    entrypoint: str | None,
    spec: dict[str, Any],
    dataset_slice_start_s: float | None,
    dataset_slice_stop_s: float | None,
) -> Any:
    """Call exported load_dataset with slice args only when supported."""
    kwargs: dict[str, Any] = {
        "dataset_vars": dataset_varset,
        "entrypoint": entrypoint,
        "eval_spec": spec,
        "slice_start": dataset_slice_start_s,
        "slice_stop": dataset_slice_stop_s,
    }
    try:
        signature = inspect.signature(pipeline_mod.load_dataset)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and not any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        kwargs = {
            name: value
            for name, value in kwargs.items()
            if name in signature.parameters
        }
    return pipeline_mod.load_dataset(dataset_root, **kwargs)


@contextmanager
def _load_exported_modules(
    bundle: ExportBundle,
) -> Iterator[tuple[str, Any, Any] | None]:
    runner = bundle.executable_artifact
    if runner is None:
        runner = bundle.output_dir / "export_python_pkg" / "runner.py"
    if runner is None or not runner.exists():
        yield None
        return
    src_dir = runner.parent / "src"
    if not src_dir.exists():
        yield None
        return
    package_dir = next(
        (
            child
            for child in src_dir.iterdir()
            if child.is_dir()
            and (child / "pipeline.py").exists()
            and (child / "atoms.py").exists()
        ),
        None,
    )
    if package_dir is None:
        yield None
        return

    package_name = package_dir.name
    module_names = [
        package_name,
        f"{package_name}.atoms",
        f"{package_name}.pipeline",
    ]
    sys.path.insert(0, str(src_dir))
    try:
        importlib.invalidate_caches()
        for name in module_names:
            sys.modules.pop(name, None)
        atoms_mod = importlib.import_module(f"{package_name}.atoms")
        pipeline_mod = importlib.import_module(f"{package_name}.pipeline")
        yield package_name, atoms_mod, pipeline_mod
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        try:
            sys.path.remove(str(src_dir))
        except ValueError:
            pass


def _extract_traced_node_functions(atoms_mod: Any) -> dict[str, str]:
    atoms_path = Path(getattr(atoms_mod, "__file__", ""))
    if not atoms_path.exists():
        return {}
    tree = ast.parse(atoms_path.read_text())
    mapping: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            if not isinstance(inner.func, ast.Name) or inner.func.id != "_sciona_probe":
                continue
            if not inner.args or not isinstance(inner.args[0], ast.Constant):
                continue
            node_id = inner.args[0].value
            if isinstance(node_id, str):
                mapping[node_id] = node.name
                break
    return mapping


def _collect_runtime_inputs_from_runtime_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Collect array-like runtime inputs when no grouped frames are available."""
    runtime_inputs: dict[str, Any] = {}
    for raw_key, raw_value in inputs.items():
        key = str(raw_key)
        lowered = key.lower()
        if "reference" in lowered or lowered.startswith("target"):
            continue
        if isinstance(raw_value, (list, tuple, np.ndarray)):
            runtime_inputs[key] = raw_value
            continue
        if np.isscalar(raw_value):
            runtime_inputs[key] = raw_value
    return runtime_inputs


def _collect_signal_data_from_runtime_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible alias for runtime-input collection."""
    return _collect_runtime_inputs_from_runtime_inputs(inputs)


def _perturb_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_perturb_value(item) for item in value)
    if isinstance(value, list):
        return [_perturb_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _perturb_value(item) for key, item in value.items()}

    array = np.asarray(value)
    if array.shape == ():
        scalar = array.item()
        if isinstance(scalar, (int, np.integer)):
            return type(scalar)(scalar + 1)
        if isinstance(scalar, (float, np.floating)):
            return type(scalar)(scalar + _float_step(array))
        return value

    if array.dtype.kind in {"i", "u"}:
        return np.asarray(array + 1, dtype=array.dtype)
    if array.dtype.kind == "f":
        return np.asarray(array + _float_step(array), dtype=array.dtype)
    return value


def _float_step(array: np.ndarray) -> float:
    finite = np.asarray(array, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1e-3
    scale = max(
        float(np.max(np.abs(finite))),
        float(np.std(finite)),
        1.0,
    )
    return scale * 1e-3
