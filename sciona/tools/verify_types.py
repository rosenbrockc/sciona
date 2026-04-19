"""Type verification tools: mypy runner, failure classifier, and deterministic fixer.

The mypy runner writes source files to a temp directory, invokes mypy --strict,
and returns the raw error output. The classifier and fixer are re-exports from
the ingester's deterministic modules.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any


def run_mypy(
    source_files: dict[str, str],
    *,
    strict: bool = True,
    python_executable: str | None = None,
) -> str:
    """Run mypy on a set of source files and return the error output.

    Writes the source files to a temporary directory, runs mypy, and
    returns the combined stdout+stderr. Cleans up the temp directory
    afterwards.

    Args:
        source_files: Mapping of filename -> source code content.
            Example: {"atoms.py": "...", "witnesses.py": "..."}
        strict: Whether to use --strict mode (default True).
        python_executable: Optional path to the Python interpreter for
            mypy's --python-executable flag. Defaults to the current
            interpreter.

    Returns:
        The raw mypy output as a string. Empty string if mypy passes
        with no errors.

    Raises:
        FileNotFoundError: If mypy is not installed.
    """
    import shutil
    import sys

    mypy_path = shutil.which("mypy")
    if mypy_path is None:
        raise FileNotFoundError(
            "mypy not found on PATH. Install with: pip install mypy"
        )

    with tempfile.TemporaryDirectory(prefix="sciona_mypy_") as tmpdir:
        tmp = Path(tmpdir)
        file_paths: list[str] = []
        for filename, content in source_files.items():
            fpath = tmp / filename
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content)
            file_paths.append(str(fpath))

        cmd = [mypy_path]
        if strict:
            cmd.append("--strict")
        cmd.append("--no-error-summary")
        if python_executable:
            cmd.extend(["--python-executable", python_executable])
        else:
            cmd.extend(["--python-executable", sys.executable])
        cmd.extend(file_paths)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = (result.stdout + result.stderr).strip()
        return output


def classify_type_failure(
    mypy_errors: str,
    source_files: dict[str, str],
) -> dict[str, Any]:
    """Classify mypy errors into repairable vs non-repairable categories.

    Categories:
    - mechanical_import, mechanical_annotation: repairable deterministically
    - semantic_signature, semantic_output_binding, etc.: not repairable
    - unknown_or_unclassified: not repairable

    Args:
        mypy_errors: Raw mypy error output.
        source_files: The source files that were checked.

    Returns:
        Classification dict with keys: stage, reason_code, repairable,
        route, summary, issues.
    """
    from sciona.ingester.verification_classifier import (
        classify_type_failure as _classify,
    )

    return _classify(mypy_errors, source_files=source_files)


def build_type_fixes(
    mypy_errors: str,
    source_files: dict[str, str],
) -> list[dict[str, Any]] | None:
    """Attempt deterministic fixes for mypy errors.

    Handles missing imports (inserts import statements) and return type
    wrapping. Returns None if no deterministic fix is available.

    Args:
        mypy_errors: Raw mypy error output.
        source_files: The source files with errors.

    Returns:
        List of patch dicts with keys {file, line_start, line_end,
        replacement}, or None if no fix applies.
    """
    from sciona.ingester.deterministic_type_fixer import (
        build_deterministic_type_fixes,
    )

    return build_deterministic_type_fixes(mypy_errors, source_files)


__all__ = ["run_mypy", "classify_type_failure", "build_type_fixes"]
