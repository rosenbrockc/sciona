"""Verification Oracle: routes verification to the appropriate proof environment."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ageom.judge.coq_env import CoqEnvironment
from ageom.judge.lean_env import LeanEnvironment

if TYPE_CHECKING:
    from ageom.judge.python_env import PythonEnvironment
from ageom.types import (
    CandidateMatch,
    PDGNode,
    Prover,
    VerificationLevel,
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

    def _resolve_verification_level(
        self, prover: Prover, verified: bool
    ) -> VerificationLevel:
        """Map prover + success to the appropriate verification level."""
        if not verified:
            return VerificationLevel.UNVERIFIED
        if prover in (Prover.LEAN4, Prover.COQ):
            return VerificationLevel.KERNEL_PROOF
        if prover == Prover.PYTHON:
            # Python uses mypy type-checking; icontract is CONTRACT_CHECKED
            # but we can't distinguish here, so default to TYPE_CHECKED
            return VerificationLevel.TYPE_CHECKED
        return VerificationLevel.UNVERIFIED

    async def verify_candidate(
        self, pdg_node: PDGNode, candidate: CandidateMatch
    ) -> VerificationResult:
        """Verify a single candidate against a PDG node's statement.

        Attempts direct type checking: `@{candidate_name}` as term for `pdg_node.statement`.
        """
        env = self._get_env(pdg_node.prover)
        term = f"@{candidate.declaration.name}"

        success, output = await env.check_term(term, pdg_node.statement)
        level = self._resolve_verification_level(pdg_node.prover, success)

        if success:
            return VerificationResult(
                candidate=candidate,
                verified=True,
                compiler_output=output,
                proof_term=term,
                verification_level=level,
            )
        else:
            return VerificationResult(
                candidate=candidate,
                verified=False,
                compiler_output=output,
                error_message=output,
                verification_level=level,
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

    async def verify_candidates_parallel(
        self,
        pdg_node: PDGNode,
        candidates: list[CandidateMatch],
        max_concurrent: int = 3,
    ) -> list[VerificationResult]:
        """Verify multiple candidates in parallel with bounded concurrency.

        Runs up to ``max_concurrent`` verification tasks simultaneously.
        Returns all results (does not short-circuit) but marks verified matches.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _verify_one(candidate: CandidateMatch) -> VerificationResult:
            async with semaphore:
                return await self.verify_candidate(pdg_node, candidate)

        tasks = [_verify_one(c) for c in candidates]
        return list(await asyncio.gather(*tasks))
