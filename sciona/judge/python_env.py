"""Python proof environment using import-based checking (default) or mypy."""

from __future__ import annotations

import asyncio
import re
import shutil
import tempfile
import textwrap
from pathlib import Path

from sciona.judge.models import CompilerFeedback


def _module(func_name: str) -> str:
    """Extract the module path from a dotted function name."""
    parts = func_name.rsplit(".", 1)
    return parts[0] if len(parts) > 1 else func_name


def _leaf(func_name: str) -> str:
    """Extract the leaf name from a dotted function name."""
    return func_name.rsplit(".", 1)[-1]


def _count_params_from_signature(type_sig: str) -> int | None:
    """Extract expected parameter count from a type signature like '(x: T, y: U) -> R'.

    Returns None if the signature cannot be parsed.
    """
    match = re.match(r"\(([^)]*)\)", type_sig)
    if not match:
        return None
    params_str = match.group(1).strip()
    if not params_str:
        return 0
    return len([p.strip() for p in params_str.split(",") if p.strip()])


class PythonEnvironment:
    """ProofEnvironment implementation for Python.

    Default mode ("import") verifies that the function exists, is callable,
    and has compatible arity.  Optional mode ("mypy") uses mypy --strict
    for full type checking.
    """

    def __init__(
        self,
        mypy_path: str = "mypy",
        python_path: str = "python",
        verify_mode: str = "import",
    ) -> None:
        self._mypy_path = mypy_path
        self._python_path = python_path
        self._verify_mode = verify_mode
        self._tmpdir = tempfile.mkdtemp(prefix="sciona_python_")

    @property
    def prover_name(self) -> str:
        return "python"

    async def check_term(self, term: str, expected_type: str) -> tuple[bool, str]:
        """Check if a term has the expected type.

        In import mode: verifies the function is importable, callable, and
        has compatible arity with the expected type signature.
        In mypy mode: writes `_result: {expected_type} = {term}` and runs mypy.
        """
        if self._verify_mode == "mypy":
            code = f"_result: {expected_type} = {term}\n"
            feedback = await self._run_mypy(code)
            return feedback.success, feedback.raw_output

        return await self._check_term_import(term, expected_type)

    async def _check_term_import(
        self, term: str, expected_type: str
    ) -> tuple[bool, str]:
        """Import-based verification: importable, callable, arity-compatible."""
        func_name = term.lstrip("@")
        mod = _module(func_name)
        leaf = _leaf(func_name)
        expected_arity = _count_params_from_signature(expected_type)

        lines = [
            f"from {mod} import {leaf}",
            f'assert callable({leaf}), "{func_name} is not callable"',
        ]
        if expected_arity is not None:
            lines.extend([
                "import inspect as _insp",
                f"_sig = _insp.signature({leaf})",
                "_params = [p for p in _sig.parameters.values()"
                " if p.default is _insp.Parameter.empty"
                " and p.kind not in (_insp.Parameter.VAR_POSITIONAL, _insp.Parameter.VAR_KEYWORD)]",
                "_actual = len(_params)",
                f'assert _actual <= {expected_arity + 2}, f"arity mismatch: expected ~{expected_arity}, got {{_actual}}"',
            ])
        lines.append('print("OK")')
        code = "\n".join(lines) + "\n"

        try:
            proc = await asyncio.create_subprocess_exec(
                self._python_path,
                "-c",
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            raw = stdout.decode() + stderr.decode()
            if proc.returncode == 0:
                return True, raw.strip()
            return False, raw.strip()
        except FileNotFoundError:
            msg = f"python not found at '{self._python_path}'"
            return False, msg

    async def check_proof(self, statement: str, proof_body: str) -> tuple[bool, str]:
        """Check a function with icontract decorators.

        Writes the statement and body to a temp file, then executes the file
        with the configured Python interpreter. For generated artifacts this is
        a better fit than strict mypy because conceptual annotations and
        untyped third-party packages are expected during synthesis.
        """
        code = f"{statement}\n{proof_body}\n"
        feedback = await self._run_file(code)
        return feedback.success, feedback.raw_output

    async def _run(self, code: str) -> CompilerFeedback:
        """Execute raw Python source for whole-file validation."""
        return await self._run_file(code)

    async def get_type(self, name: str) -> str | None:
        """Get the type signature of a name via inspect.signature."""
        code = f"import inspect\n" f"print(inspect.signature({name}))\n"
        try:
            proc = await asyncio.create_subprocess_exec(
                self._python_path,
                "-c",
                code,
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

        try:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        except Exception:
            pass

    async def _run_mypy(self, code: str) -> CompilerFeedback:
        """Write code to a temp .py file and run mypy --strict."""
        tmp_path = Path(self._tmpdir) / "_check.py"
        tmp_path.write_text(code)

        try:
            proc = await asyncio.create_subprocess_exec(
                self._mypy_path,
                "--strict",
                str(tmp_path),
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

    async def _run_file(self, code: str) -> CompilerFeedback:
        """Write code to a temp .py file and execute it."""
        tmp_path = Path(self._tmpdir) / "_check.py"
        tmp_path.write_text(code)

        try:
            proc = await asyncio.create_subprocess_exec(
                self._python_path,
                str(tmp_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            raw = stdout.decode() + stderr.decode()
            return CompilerFeedback(
                raw_output=raw,
                errors=[] if proc.returncode == 0 else [raw.strip() or raw],
            )
        except FileNotFoundError:
            return CompilerFeedback(
                raw_output=f"python not found at '{self._python_path}'",
                errors=[f"python not found at '{self._python_path}'"],
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
