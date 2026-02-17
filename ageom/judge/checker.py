"""Verification Oracle: routes verification to the appropriate proof environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ageom.judge.coq_env import CoqEnvironment
from ageom.judge.lean_env import LeanEnvironment

if TYPE_CHECKING:
    from ageom.judge.python_env import PythonEnvironment
from ageom.types import (
    CandidateMatch,
    PDGNode,
    Prover,
    VerificationResult,
)


class VerificationOracleImpl:
    """Concrete implementation of the VerificationOracle protocol.

    Routes verification requests to LeanEnvironment or CoqEnvironment
    based on the PDG node's prover field.
    """

    def __init__(
        self,
        lean_env: LeanEnvironment | None = None,
        coq_env: CoqEnvironment | None = None,
        python_env: "PythonEnvironment | None" = None,
    ) -> None:
        self._lean_env = lean_env
        self._coq_env = coq_env
        self._python_env = python_env

    def _get_env(self, prover: Prover) -> LeanEnvironment | CoqEnvironment | "PythonEnvironment":
        if prover == Prover.LEAN4:
            if self._lean_env is None:
                raise RuntimeError("LeanEnvironment not configured")
            return self._lean_env
        elif prover == Prover.COQ:
            if self._coq_env is None:
                raise RuntimeError("CoqEnvironment not configured")
            return self._coq_env
        elif prover == Prover.PYTHON:
            if self._python_env is None:
                raise RuntimeError("PythonEnvironment not configured")
            return self._python_env
        else:
            raise ValueError(f"Unsupported prover: {prover}")

    async def verify_candidate(
        self, pdg_node: PDGNode, candidate: CandidateMatch
    ) -> VerificationResult:
        """Verify a single candidate against a PDG node's statement.

        Attempts direct type checking: `@{candidate_name}` as term for `pdg_node.statement`.
        """
        env = self._get_env(pdg_node.prover)
        term = f"@{candidate.declaration.name}"

        success, output = await env.check_term(term, pdg_node.statement)

        if success:
            return VerificationResult(
                candidate=candidate,
                verified=True,
                compiler_output=output,
                proof_term=term,
            )
        else:
            return VerificationResult(
                candidate=candidate,
                verified=False,
                compiler_output=output,
                error_message=output,
            )

    async def verify_candidates(
        self, pdg_node: PDGNode, candidates: list[CandidateMatch]
    ) -> list[VerificationResult]:
        """Verify multiple candidates, short-circuiting on first verified match."""
        results: list[VerificationResult] = []
        for candidate in candidates:
            result = await self.verify_candidate(pdg_node, candidate)
            results.append(result)
            if result.verified:
                break
        return results
