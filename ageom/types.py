"""Shared domain types for the AGEO-Matcher pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Prover(str, Enum):
    """Supported proof assistants."""

    LEAN4 = "lean4"
    COQ = "coq"


@dataclass(frozen=True)
class Declaration:
    """A formal declaration extracted from a proof library.

    Represents a theorem, lemma, definition, or axiom from Lean 4/Mathlib or Coq.
    """

    name: str
    type_signature: str
    docstring: str = ""
    source_lib: str = ""
    prover: Prover = Prover.LEAN4
    raw_code: str = ""


@dataclass(frozen=True)
class PDGNode:
    """A node from the Predicate Dependency Graph (Round 1 output).

    Represents a high-level predicate that needs to be grounded
    into a verified library function.
    """

    predicate_id: str
    statement: str
    informal_desc: str = ""
    prover: Prover = Prover.LEAN4
    context: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateMatch:
    """A candidate library function that might match a PDG predicate."""

    declaration: Declaration
    score: float
    retrieval_method: str


@dataclass(frozen=True)
class VerificationResult:
    """Result of attempting to verify a candidate match via the compiler."""

    candidate: CandidateMatch
    verified: bool
    compiler_output: str = ""
    proof_term: str = ""
    error_message: str = ""


@dataclass
class MatchResult:
    """Final result of the matching pipeline for a single PDG node."""

    pdg_node: PDGNode
    verified_match: VerificationResult | None = None
    all_candidates: list[CandidateMatch] = field(default_factory=list)
    all_verifications: list[VerificationResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.verified_match is not None and self.verified_match.verified
