"""Protocol interfaces for the AGEO-Matcher components.

Defines the contracts between the Semantic Indexer, Verification Oracle,
and Retrieval Agent using Python Protocols for structural subtyping.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sciona.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
    PDGNode,
    VerificationResult,
)


@runtime_checkable
class SemanticIndex(Protocol):
    """Search interface over indexed formal declarations."""

    def search_by_embedding(
        self, query_text: str, k: int = 10
    ) -> list[tuple[Declaration, float]]:
        """Search by embedding similarity. Returns (declaration, score) pairs."""
        ...

    def search_by_type(self, type_signature: str, k: int = 10) -> list[Declaration]:
        """Search by type signature (exact or approximate)."""
        ...

    def get_declaration(self, name: str) -> Declaration | None:
        """Look up a declaration by fully-qualified name."""
        ...


@runtime_checkable
class ProofEnvironment(Protocol):
    """Interface to a proof assistant's REPL for type checking."""

    @property
    def prover_name(self) -> str:
        """Name of the proof assistant (e.g. 'lean4', 'coq')."""
        ...

    async def check_term(self, term: str, expected_type: str) -> tuple[bool, str]:
        """Check if a term has the expected type.

        Returns (success, compiler_output).
        """
        ...

    async def check_proof(self, statement: str, proof_body: str) -> tuple[bool, str]:
        """Check if a proof body proves the statement.

        Returns (success, compiler_output).
        """
        ...

    async def get_type(self, name: str) -> str | None:
        """Get the type of a named declaration, or None if not found."""
        ...

    async def close(self) -> None:
        """Shut down the proof environment."""
        ...


@runtime_checkable
class VerificationOracle(Protocol):
    """Compiler-based verification of candidate matches."""

    async def verify_candidate(
        self, pdg_node: PDGNode, candidate: CandidateMatch
    ) -> VerificationResult:
        """Verify a single candidate against a PDG node's statement."""
        ...

    async def verify_candidates(
        self, pdg_node: PDGNode, candidates: list[CandidateMatch]
    ) -> list[VerificationResult]:
        """Verify multiple candidates. May short-circuit on first success."""
        ...

    async def verify_candidates_parallel(
        self,
        pdg_node: PDGNode,
        candidates: list[CandidateMatch],
        max_concurrent: int = 3,
    ) -> list[VerificationResult]:
        """Verify multiple candidates in parallel with bounded concurrency.

        Default implementation falls back to sequential verification.
        """
        return await self.verify_candidates(pdg_node, candidates)


@runtime_checkable
class RetrievalAgent(Protocol):
    """Agentic retrieval loop that finds verified matches."""

    async def find_match(self, pdg_node: PDGNode) -> MatchResult:
        """Find and verify a library match for the given PDG node."""
        ...
