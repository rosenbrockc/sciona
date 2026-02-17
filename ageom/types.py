"""Shared domain types for the AGEO-Matcher pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Prover(str, Enum):
    """Supported proof assistants."""

    LEAN4 = "lean4"
    COQ = "coq"
    PYTHON = "python"


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

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""

        def _decl_dict(d: Declaration) -> dict:
            return {
                "name": d.name,
                "type_signature": d.type_signature,
                "docstring": d.docstring,
                "source_lib": d.source_lib,
                "prover": d.prover.value,
                "raw_code": d.raw_code,
            }

        def _candidate_dict(c: CandidateMatch) -> dict:
            return {
                "declaration": _decl_dict(c.declaration),
                "score": c.score,
                "retrieval_method": c.retrieval_method,
            }

        def _vr_dict(vr: VerificationResult) -> dict:
            return {
                "candidate": _candidate_dict(vr.candidate),
                "verified": vr.verified,
                "compiler_output": vr.compiler_output,
                "proof_term": vr.proof_term,
                "error_message": vr.error_message,
            }

        result: dict = {
            "pdg_node": {
                "predicate_id": self.pdg_node.predicate_id,
                "statement": self.pdg_node.statement,
                "informal_desc": self.pdg_node.informal_desc,
                "prover": self.pdg_node.prover.value,
                "context": dict(self.pdg_node.context),
            },
            "verified_match": _vr_dict(self.verified_match) if self.verified_match else None,
            "all_candidates": [_candidate_dict(c) for c in self.all_candidates],
            "all_verifications": [_vr_dict(vr) for vr in self.all_verifications],
        }
        return result

    @staticmethod
    def from_dict(data: dict) -> "MatchResult":
        """Deserialize from a JSON-friendly dict."""

        def _decl(d: dict) -> Declaration:
            return Declaration(
                name=d["name"],
                type_signature=d.get("type_signature", ""),
                docstring=d.get("docstring", ""),
                source_lib=d.get("source_lib", ""),
                prover=Prover(d.get("prover", "lean4")),
                raw_code=d.get("raw_code", ""),
            )

        def _candidate(d: dict) -> CandidateMatch:
            return CandidateMatch(
                declaration=_decl(d["declaration"]),
                score=d.get("score", 0.0),
                retrieval_method=d.get("retrieval_method", ""),
            )

        def _vr(d: dict) -> VerificationResult:
            return VerificationResult(
                candidate=_candidate(d["candidate"]),
                verified=d.get("verified", False),
                compiler_output=d.get("compiler_output", ""),
                proof_term=d.get("proof_term", ""),
                error_message=d.get("error_message", ""),
            )

        pdg_data = data["pdg_node"]
        pdg_node = PDGNode(
            predicate_id=pdg_data["predicate_id"],
            statement=pdg_data.get("statement", ""),
            informal_desc=pdg_data.get("informal_desc", ""),
            prover=Prover(pdg_data.get("prover", "lean4")),
            context=pdg_data.get("context", {}),
        )

        verified_match = _vr(data["verified_match"]) if data.get("verified_match") else None
        all_candidates = [_candidate(c) for c in data.get("all_candidates", [])]
        all_verifications = [_vr(vr) for vr in data.get("all_verifications", [])]

        return MatchResult(
            pdg_node=pdg_node,
            verified_match=verified_match,
            all_candidates=all_candidates,
            all_verifications=all_verifications,
        )
