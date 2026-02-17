"""Python proof environment using mypy for type checking."""

from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path

from ageom.judge.models import CompilerFeedback


class PythonEnvironment:
    """ProofEnvironment implementation for Python via mypy.

    Uses temporary .py files type-checked through mypy --strict.
    icontract decorators provide contract checking (analogous to proof obligations).
    """

    def __init__(self, mypy_path: str = "mypy", python_path: str = "python") -> None:
        self._mypy_path = mypy_path
        self._python_path = python_path
        self._tmpdir = tempfile.mkdtemp(prefix="ageom_python_")

    @property
    def prover_name(self) -> str:
        return "python"

    async def check_term(self, term: str, expected_type: str) -> tuple[bool, str]:
        """Check if a term has the expected type.

        Writes: `_result: {expected_type} = {term}` and runs mypy.
        """
        code = f"_result: {expected_type} = {term}\n"
        feedback = await self._run(code)
        return feedback.success, feedback.raw_output

    async def check_proof(self, statement: str, proof_body: str) -> tuple[bool, str]:
        """Check a function with icontract decorators.

        Writes the statement (function signature with decorators) and body,
        then runs mypy to validate.
        """
        code = f"{statement}\n{proof_body}\n"
        feedback = await self._run(code)
        return feedback.success, feedback.raw_output

    async def get_type(self, name: str) -> str | None:
        """Get the type signature of a name via inspect.signature."""
        code = (
            f"import inspect\n"
            f"print(inspect.signature({name}))\n"
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                self._python_path, "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                return stdout.decode().strip() or None
            return None
        except FileNotFoundError:
            return None

    async def close(self) -> None:
        """Clean up temp directory."""
        import shutil

        try:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        except Exception:
            pass

    async def _run(self, code: str) -> CompilerFeedback:
        """Write code to a temp .py file and run mypy --strict."""
        tmp_path = Path(self._tmpdir) / "_check.py"
        tmp_path.write_text(code)

        try:
            proc = await asyncio.create_subprocess_exec(
                self._mypy_path, "--strict", str(tmp_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            raw = stdout.decode() + stderr.decode()
            return _parse_mypy_output(raw)
        except FileNotFoundError:
            return CompilerFeedback(
                raw_output=f"mypy not found at '{self._mypy_path}'",
                errors=[f"mypy not found at '{self._mypy_path}'"],
            )


def _parse_mypy_output(raw: str) -> CompilerFeedback:
    """Parse mypy output into structured CompilerFeedback."""
    errors: list[str] = []
    warnings: list[str] = []

    for line in raw.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if re.search(r":\d+: error:", line_stripped):
            errors.append(line_stripped)
        elif re.search(r":\d+: warning:", line_stripped):
            warnings.append(line_stripped)
        elif re.search(r":\d+: note:", line_stripped):
            pass  # notes are informational
        elif "Found" in line_stripped and "error" in line_stripped:
            pass  # summary line like "Found 2 errors in 1 file"

    return CompilerFeedback(
        raw_output=raw,
        errors=errors,
        warnings=warnings,
    )
