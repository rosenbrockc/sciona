"""Python package template generation for verified skeletons."""

from __future__ import annotations

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


def generate_pipeline_py(pipeline_steps: list[str]) -> str:
    """Generate pipeline.py that wires atoms per CDG topology."""
    lines: list[str] = []
    lines.append('"""Pipeline orchestration: wires atoms per CDG topology."""')
    lines.append("")
    lines.append("from . import atoms")
    lines.append("")
    lines.append("")
    lines.append("def run_pipeline(**kwargs):")
    lines.append('    """Execute the verified pipeline."""')

    if pipeline_steps:
        for step in pipeline_steps:
            lines.append(f"    {step}")
    else:
        lines.append('    raise NotImplementedError("TODO: wire atoms into pipeline")')

    lines.append("")

    return "\n".join(lines)
