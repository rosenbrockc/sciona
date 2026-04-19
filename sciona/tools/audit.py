"""Subprocess wrappers for the contribution and dejargon validation scripts.

These run the shared validation scripts from sciona-atoms as subprocesses,
returning (passed, output) tuples.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _resolve_atoms_repo(atoms_repo: str | None) -> Path:
    """Resolve the sciona-atoms repo path."""
    if atoms_repo:
        return Path(atoms_repo)
    env = os.environ.get("SCIONA_ATOMS_ROOT")
    if env:
        return Path(env)
    # Default: sibling of sciona-matcher
    matcher_root = Path(__file__).resolve().parent.parent.parent
    default = matcher_root.parent / "sciona-atoms"
    if default.is_dir():
        return default
    raise FileNotFoundError(
        "Cannot find sciona-atoms repo. Set SCIONA_ATOMS_ROOT or pass atoms_repo."
    )


def _resolve_venv_python() -> str:
    """Resolve the sciona-matcher venv Python path."""
    matcher_root = Path(__file__).resolve().parent.parent.parent
    venv_python = matcher_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def run_contribution_check(
    repo_root: str,
    *,
    atoms_repo: str | None = None,
) -> tuple[bool, str]:
    """Run verify_contribution_rules.py on a repo.

    Args:
        repo_root: Path to the atom repo to validate.
        atoms_repo: Optional path to sciona-atoms (for locating scripts).

    Returns:
        (passed, output) tuple.
    """
    atoms = _resolve_atoms_repo(atoms_repo)
    script = atoms / "scripts" / "verify_contribution_rules.py"
    if not script.exists():
        return False, f"Script not found: {script}"

    result = subprocess.run(
        [_resolve_venv_python(), str(script), "--repo-root", repo_root],
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output


def run_dejargon_check(
    repo_root: str,
    *,
    atoms_repo: str | None = None,
) -> tuple[bool, str]:
    """Run validate_dejargon.py on a repo.

    Args:
        repo_root: Path to the atom repo to validate.
        atoms_repo: Optional path to sciona-atoms (for locating scripts).

    Returns:
        (passed, output) tuple.
    """
    atoms = _resolve_atoms_repo(atoms_repo)
    script = atoms / "scripts" / "validate_dejargon.py"
    if not script.exists():
        return False, f"Script not found: {script}"

    result = subprocess.run(
        [_resolve_venv_python(), str(script), "--root", repo_root],
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output


__all__ = ["run_contribution_check", "run_dejargon_check"]
