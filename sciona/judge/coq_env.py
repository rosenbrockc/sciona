"""Coq proof environment using coqpyt."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from sciona.judge.models import CompilerFeedback


class CoqEnvironment:
    """ProofEnvironment implementation for Coq via coqpyt.

    Uses temporary .v files compiled through CoqFile/ProofFile.
    """

    def __init__(self, project_path: str | Path = "") -> None:
        self._project_path = str(project_path) if project_path else None

    @property
    def prover_name(self) -> str:
        return "coq"

    async def check_term(self, term: str, expected_type: str) -> tuple[bool, str]:
        """Check if a term has the expected type.

        Compiles: `Definition _check : {expected_type} := {term}.`
        """
        code = f"Definition _check : {expected_type} := {term}."
        feedback = await self._run(code)
        return feedback.success, feedback.raw_output

    async def check_proof(self, statement: str, proof_body: str) -> tuple[bool, str]:
        """Check if a proof body proves the statement.

        Compiles: `Lemma _check : {statement}. Proof. {proof_body} Qed.`
        """
        code = f"Lemma _check : {statement}. Proof. {proof_body} Qed."
        feedback = await self._run(code)
        return feedback.success, feedback.raw_output

    async def get_type(self, name: str) -> str | None:
        """Get the type of a named term via `Check {name}.`"""
        code = f"Check {name}."
        feedback = await self._run(code)
        if feedback.errors:
            return None
        # Parse "name : type" from output
        for line in feedback.raw_output.splitlines():
            if ":" in line:
                return line.split(":", 1)[1].strip()
        return None

    async def close(self) -> None:
        """No persistent server to close for coqpyt."""
        pass

    async def _run(self, code: str) -> CompilerFeedback:
        """Write code to a temp .v file and compile it."""

        def _execute() -> CompilerFeedback:
            from coqpyt.coq_file import CoqFile

            with tempfile.NamedTemporaryFile(suffix=".v", mode="w", delete=False) as f:
                f.write(code)
                f.flush()
                tmp_path = f.name

            errors: list[str] = []
            warnings: list[str] = []
            goals: list[str] = []
            raw_parts: list[str] = []

            try:
                coq_file = CoqFile(tmp_path)
                try:
                    for step in coq_file.steps:
                        diagnostics = getattr(step, "diagnostics", [])
                        for diag in diagnostics:
                            msg = str(diag)
                            raw_parts.append(msg)
                            severity = getattr(diag, "severity", None)
                            if severity and "error" in str(severity).lower():
                                errors.append(msg)
                            elif severity and "warning" in str(severity).lower():
                                warnings.append(msg)
                        step_goals = getattr(step, "goals", [])
                        for g in step_goals:
                            goals.append(str(g))
                finally:
                    coq_file.close()
            except Exception as e:
                errors.append(str(e))
                raw_parts.append(str(e))
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            return CompilerFeedback(
                raw_output="\n".join(raw_parts),
                errors=errors,
                warnings=warnings,
                goals_remaining=goals,
            )

        return await asyncio.to_thread(_execute)
