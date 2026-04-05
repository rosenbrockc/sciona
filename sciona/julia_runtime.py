"""Shared JuliaCall/JuliaPkg runtime configuration for sciona.

This keeps Julia's writable project/depot state out of shared Python virtual
environments, which avoids repeated lockfile failures in tests and ghost-sim
execution.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_JULIA_PROJECT = Path("/tmp/sciona_juliacall_project")
DEFAULT_JULIA_DEPOT = Path("/tmp/sciona_julia_depot")
PYTHONCALL_UUID = "6099a3de-4c0c-538f-b890-85f7d6d5f1b6"


@dataclass(frozen=True)
class JuliaRuntimeConfig:
    project: Path
    depot: Path
    julia_exe: str


def discover_julia_executable() -> str:
    """Return the preferred Julia executable, if any."""
    explicit = os.environ.get("PYTHON_JULIACALL_EXE")
    if explicit:
        return explicit

    launcher = shutil.which("julia") or ""
    direct_bins = sorted(
        Path.home().joinpath(".julia", "juliaup").glob("julia-*/bin/julia"),
        reverse=True,
    )
    if direct_bins:
        return str(direct_bins[0])
    return launcher


def _project_has_pythoncall(project: Path) -> bool:
    project_toml = project / "Project.toml"
    if not project_toml.exists():
        return False
    text = project_toml.read_text(encoding="utf-8")
    return "PythonCall" in text or PYTHONCALL_UUID in text


def configure_juliacall_env(
    *,
    project: str | Path | None = None,
    depot: str | Path | None = None,
) -> JuliaRuntimeConfig:
    """Point JuliaCall/JuliaPkg at writable runtime locations."""

    project_path = Path(
        project
        or os.environ.get("PYTHON_JULIACALL_PROJECT")
        or os.environ.get("PYTHON_JULIAPKG_PROJECT")
        or DEFAULT_JULIA_PROJECT
    )
    depot_path = Path(
        depot
        or os.environ.get("JULIA_DEPOT_PATH")
        or DEFAULT_JULIA_DEPOT
    )
    project_path.mkdir(parents=True, exist_ok=True)
    depot_path.mkdir(parents=True, exist_ok=True)

    julia_exe = discover_julia_executable()
    os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(project_path))
    os.environ.setdefault("JULIA_DEPOT_PATH", str(depot_path))
    if _project_has_pythoncall(project_path):
        if julia_exe:
            os.environ.setdefault("PYTHON_JULIACALL_EXE", julia_exe)
        os.environ.setdefault("PYTHON_JULIACALL_PROJECT", str(project_path))

    return JuliaRuntimeConfig(
        project=project_path,
        depot=depot_path,
        julia_exe=julia_exe,
    )


def prewarm_juliacall_project(
    *,
    project: str | Path | None = None,
    depot: str | Path | None = None,
) -> JuliaRuntimeConfig:
    """Provision a writable Julia project that contains PythonCall."""

    cfg = configure_juliacall_env(project=project, depot=depot)
    if not cfg.julia_exe:
        raise RuntimeError("Julia executable not found on PATH")

    bootstrap = (
        "import Pkg\n"
        f'Pkg.activate(raw"{cfg.project}")\n'
        "try\n"
        "    import PythonCall\n"
        "catch\n"
        '    Pkg.add(\"PythonCall\")\n'
        "    import PythonCall\n"
        "end\n"
        "Pkg.instantiate()\n"
        "println(Base.active_project())\n"
    )
    env = dict(os.environ)
    env["JULIA_DEPOT_PATH"] = str(cfg.depot)
    env["PYTHON_JULIAPKG_PROJECT"] = str(cfg.project)
    subprocess.run(
        [cfg.julia_exe, "--startup-file=no", "-e", bootstrap],
        check=True,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    os.environ["PYTHON_JULIACALL_PROJECT"] = str(cfg.project)
    return cfg
