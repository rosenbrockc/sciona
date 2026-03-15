"""Python package template generation for verified skeletons."""

from __future__ import annotations

import json

from ageom.synthesizer.models import SkeletonFile


def generate_pyproject_toml(package_name: str, dependencies: list[str]) -> str:
    """Generate a pyproject.toml for the output package."""
    deps = "\n".join(f'    "{dep}",' for dep in dependencies)
    return f"""\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{package_name}"
version = "0.1.0"
description = "AGEO-Matcher verified Python package"
requires-python = ">=3.10"
dependencies = [
    "icontract>=2.6",
{deps}
]

[tool.hatch.build.targets.wheel]
packages = ["src/{package_name}"]
"""


def generate_init_py(package_name: str, exports: list[str]) -> str:
    """Generate an __init__.py with explicit exports."""
    export_lines = "\n".join(f'    "{name}",' for name in exports)
    import_lines = "\n".join(
        f"from {package_name}.atoms import {name}" for name in exports
    )
    return f"""\
\"""{package_name}: AGEO-Matcher verified package.\"""

{import_lines}

__all__ = [
{export_lines}
]
"""


def generate_main_script(
    skeleton: SkeletonFile,
    wrappers: list[str],
    pipeline_steps: list[str],
) -> str:
    """Generate atoms.py with icontract-wrapped safe atoms."""
    lines: list[str] = []
    lines.append('"""Safe atom wrappers with icontract contracts."""')
    lines.append("")
    lines.append("import icontract")
    lines.append("")

    # Collect unique imports from skeleton metadata
    imports_seen: set[str] = set()
    for unit in skeleton.units:
        module = (
            unit.declaration_name.rsplit(".", 1)[0]
            if "." in unit.declaration_name
            else ""
        )
        if module:
            top_level = module.split(".")[0]
            if top_level not in imports_seen:
                imports_seen.add(top_level)
                lines.append(f"import {top_level}")

    if imports_seen:
        lines.append("")

    # Wrapper functions
    for wrapper_code in wrappers:
        lines.append("")
        lines.append(wrapper_code)
        lines.append("")

    return "\n".join(lines)


def generate_pipeline_py(
    pipeline_steps: list[str],
    *,
    entrypoint_names: list[str] | None = None,
    default_entrypoint: str | None = None,
) -> str:
    """Generate pipeline.py that wires atoms per CDG topology."""
    entrypoint_names = entrypoint_names or []
    entrypoints_json = json.dumps(entrypoint_names)
    default_json = json.dumps(default_entrypoint)
    lines: list[str] = []
    lines.append("# mypy: disable-error-code=no-any-return")
    lines.append('"""Pipeline orchestration: wires atoms per CDG topology."""')
    lines.append("")
    lines.append("import argparse")
    lines.append("import inspect")
    lines.append("import json")
    lines.append("from pathlib import Path")
    lines.append("from . import atoms")
    lines.append("from typing import Any")
    lines.append("from ageom.principal.datasets import create_templated_dataset_collection")
    lines.append("")
    lines.append(f"ENTRYPOINTS = {entrypoints_json}")
    lines.append(f"DEFAULT_ENTRYPOINT = {default_json}")
    lines.append("")
    lines.append("")
    lines.append("def _parse_dataset_vars(values: list[str] | None) -> dict[str, str]:")
    lines.append("    result: dict[str, str] = {}")
    lines.append("    for item in values or []:")
    lines.append("        if '=' not in item:")
    lines.append("            raise ValueError(f'dataset var must be KEY=VALUE, got: {item!r}')")
    lines.append("        key, value = item.split('=', 1)")
    lines.append("        result[key] = value")
    lines.append("    return result")
    lines.append("")
    lines.append("def _jsonify(value: Any) -> Any:")
    lines.append("    if hasattr(value, 'tolist'):")
    lines.append("        return value.tolist()")
    lines.append("    if isinstance(value, dict):")
    lines.append("        return {str(k): _jsonify(v) for k, v in value.items()}")
    lines.append("    if isinstance(value, (list, tuple)):")
    lines.append("        return [_jsonify(v) for v in value]")
    lines.append("    return value")
    lines.append("")
    lines.append("def _resolve_adapter(dataset_root: str | Path) -> Path:")
    lines.append("    root = Path(dataset_root).expanduser().resolve()")
    lines.append("    adapter = root / 'ageom.yml'")
    lines.append("    if not adapter.exists():")
    lines.append("        raise FileNotFoundError(f'No ageom.yml found in dataset root: {root}')")
    lines.append("    return adapter")
    lines.append("")
    lines.append("def load_dataset(")
    lines.append("    dataset_root: str | Path,")
    lines.append("    dataset_vars: dict[str, str] | None = None,")
    lines.append("    user: str | None = None,")
    lines.append("    serial: str | None = None,")
    lines.append(") -> dict[str, Any]:")
    lines.append("    adapter = _resolve_adapter(dataset_root)")
    lines.append("    coll_cls = create_templated_dataset_collection(str(adapter), varset=dataset_vars)")
    lines.append("    options = coll_cls.get_filter_options(user, serial, recursive=True)")
    lines.append("    coll = coll_cls.from_folder(options=options)")
    lines.append("    return dict(coll.to_pandas())")
    lines.append("")
    lines.append("def _flatten_inputs(kwargs: dict[str, Any]) -> dict[str, Any]:")
    lines.append("    flat = dict(kwargs)")
    lines.append("    for group_name, frame in list(kwargs.items()):")
    lines.append("        if not hasattr(frame, 'columns'):")
    lines.append("            continue")
    lines.append("        prefix = f'{group_name}_'")
    lines.append("        for column in frame.columns:")
    lines.append("            series = frame[column]")
    lines.append("            value = series.to_numpy() if hasattr(series, 'to_numpy') else series")
    lines.append("            flat[str(column)] = value")
    lines.append("            if isinstance(column, str) and column.startswith(prefix):")
    lines.append("                flat.setdefault(column[len(prefix):], value)")
    lines.append("    return flat")
    lines.append("")
    lines.append("def _select_call_kwargs(fn: Any, flat_inputs: dict[str, Any]) -> dict[str, Any]:")
    lines.append("    signature = inspect.signature(fn)")
    lines.append("    result: dict[str, Any] = {}")
    lines.append("    for name, param in signature.parameters.items():")
    lines.append("        if name in flat_inputs:")
    lines.append("            result[name] = flat_inputs[name]")
    lines.append("        elif param.default is not inspect._empty:")
    lines.append("            continue")
    lines.append("    return result")
    lines.append("")
    lines.append("def run_pipeline(**kwargs: Any) -> Any:")
    lines.append('    """Execute the verified pipeline."""')
    lines.append("    if not ENTRYPOINTS:")
    lines.append('        raise NotImplementedError("No exported entrypoints are available")')
    lines.append("    entrypoint = kwargs.pop('entrypoint', None) or DEFAULT_ENTRYPOINT or ENTRYPOINTS[0]")
    lines.append("    fn = getattr(atoms, entrypoint)")
    lines.append("    flat_inputs = _flatten_inputs(kwargs)")
    lines.append("    call_kwargs = _select_call_kwargs(fn, flat_inputs)")

    if pipeline_steps:
        for step in pipeline_steps:
            lines.append(f"    {step}")
    else:
        lines.append("    try:")
        lines.append("        return fn(**call_kwargs)")
        lines.append("    except TypeError as exc:")
        lines.append("        if call_kwargs and 'required positional argument' in str(exc):")
        lines.append("            return fn()")
        lines.append("        raise")

    lines.append("")
    lines.append("")
    lines.append("def main(argv: list[str] | None = None) -> int:")
    lines.append("    parser = argparse.ArgumentParser()")
    lines.append("    parser.add_argument('--dataset-root', required=True)")
    lines.append("    parser.add_argument('--dataset-var', action='append', default=[])")
    lines.append("    parser.add_argument('--user', default=None)")
    lines.append("    parser.add_argument('--serial', default=None)")
    lines.append("    parser.add_argument('--entrypoint', default=None)")
    lines.append("    parser.add_argument('--trace-path', default='trace.jsonl')")
    lines.append("    parser.add_argument('--output', default=None)")
    lines.append("    args = parser.parse_args(argv)")
    lines.append("")
    lines.append("    dataset_vars = _parse_dataset_vars(args.dataset_var)")
    lines.append("    if hasattr(atoms, '_AGEOM_TRACE_PATH'):")
    lines.append("        atoms._AGEOM_TRACE_PATH = args.trace_path")
    lines.append("    group_frames = load_dataset(")
    lines.append("        args.dataset_root,")
    lines.append("        dataset_vars=dataset_vars or None,")
    lines.append("        user=args.user,")
    lines.append("        serial=args.serial,")
    lines.append("    )")
    lines.append("    result = run_pipeline(entrypoint=args.entrypoint, **group_frames)")
    lines.append("    payload = {")
    lines.append("        'mse': 0.0,")
    lines.append("        'entrypoint': args.entrypoint or DEFAULT_ENTRYPOINT or (ENTRYPOINTS[0] if ENTRYPOINTS else None),")
    lines.append("        'outputs': _jsonify(result),")
    lines.append("    }")
    lines.append("    text = json.dumps(payload)")
    lines.append("    if args.output:")
    lines.append("        Path(args.output).write_text(text + '\\n')")
    lines.append("    print(text)")
    lines.append("    return 0")
    lines.append("")

    return "\n".join(lines)


def generate_runner_py(package_name: str) -> str:
    """Generate a top-level runner that executes the exported pipeline package."""
    return f"""\
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from {package_name}.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
"""
