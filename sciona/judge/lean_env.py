"""Lean 4 proof environment using lean-interact."""

from __future__ import annotations

import asyncio
import re

from sciona.judge.models import CompilerFeedback


class LeanEnvironment:
    """ProofEnvironment implementation for Lean 4 via lean-interact.

    Uses lean-interact's LeanServer with a TempRequireProject
    configured to include Mathlib.
    """

    def __init__(self, toolchain: str = "leanprover/lean4:v4.14.0") -> None:
        from lean_interact import LeanREPLConfig, LeanServer, TempRequireProject

        project = TempRequireProject(packages={"mathlib": ""})
        config = LeanREPLConfig(project=project, toolchain=toolchain)
        self._server = LeanServer(config)

    @property
    def prover_name(self) -> str:
        return "lean4"

    async def check_term(self, term: str, expected_type: str) -> tuple[bool, str]:
        """Check if a term has the expected type.

        Compiles: `example : {expected_type} := {term}`
        """
        code = f"example : {expected_type} := {term}"
        feedback = await self._run(code)
        return feedback.success, feedback.raw_output

    async def check_proof(self, statement: str, proof_body: str) -> tuple[bool, str]:
        """Check if a proof body proves the statement.

        Compiles: `theorem _check : {statement} := by {proof_body}`
        """
        code = f"theorem _check : {statement} := by {proof_body}"
        feedback = await self._run(code)
        return feedback.success, feedback.raw_output

    async def get_type(self, name: str) -> str | None:
        """Get the type of a named declaration via `#check @{name}`."""
        code = f"#check @{name}"
        feedback = await self._run(code)
        if feedback.errors:
            return None
        # Parse the type from the info message: "{name} : {type}"
        for line in feedback.raw_output.splitlines():
            if ":" in line:
                return line.split(":", 1)[1].strip()
        return None

    async def close(self) -> None:
        """Shut down the Lean REPL server."""
        if hasattr(self, "_server"):
            try:
                self._server.close()
            except Exception:
                pass

    async def _run(self, code: str) -> CompilerFeedback:
        """Run Lean code and parse the output into CompilerFeedback."""

        def _execute() -> str:
            result = self._server.run(code)
            return str(result)

        raw = await asyncio.to_thread(_execute)
        return _parse_lean_output(raw)


def _parse_lean_output(raw: str) -> CompilerFeedback:
    """Parse lean-interact output into structured CompilerFeedback."""
    errors: list[str] = []
    warnings: list[str] = []
    goals: list[str] = []

    for line in raw.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        # lean-interact returns error/warning/info messages
        if "error" in line_stripped.lower():
            errors.append(line_stripped)
        elif "warning" in line_stripped.lower():
            warnings.append(line_stripped)
        elif (
            re.match(r"^⊢", line_stripped) or "unsolved goals" in line_stripped.lower()
        ):
            goals.append(line_stripped)

    return CompilerFeedback(
        raw_output=raw,
        errors=errors,
        warnings=warnings,
        goals_remaining=goals,
    )
