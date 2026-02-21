"""Tests for shared domain types."""

from ageom.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
    PDGNode,
    Prover,
    VerificationResult,
)


def test_declaration_creation():
    decl = Declaration(
        name="Nat.add_comm",
        type_signature="∀ (n m : ℕ), n + m = m + n",
        docstring="Addition is commutative",
        source_lib="Mathlib",
        prover=Prover.LEAN4,
    )
    assert decl.name == "Nat.add_comm"
    assert decl.prover == Prover.LEAN4


def test_declaration_frozen():
    decl = Declaration(name="test", type_signature="Nat")
    try:
        decl.name = "other"  # type: ignore[misc]
        assert False, "Should be frozen"
    except AttributeError:
        pass


def test_pdg_node_defaults():
    node = PDGNode(predicate_id="p1", statement="1 + 1 = 2")
    assert node.prover == Prover.LEAN4
    assert node.context == {}


def test_match_result_success():
    decl = Declaration(name="test", type_signature="Nat")
    candidate = CandidateMatch(
        declaration=decl, score=0.9, retrieval_method="embedding"
    )
    vr = VerificationResult(candidate=candidate, verified=True, proof_term="@test")
    node = PDGNode(predicate_id="p1", statement="Nat")

    result = MatchResult(pdg_node=node, verified_match=vr)
    assert result.success is True


def test_match_result_failure():
    node = PDGNode(predicate_id="p1", statement="Nat")
    result = MatchResult(pdg_node=node)
    assert result.success is False


def test_prover_enum_values():
    assert Prover.LEAN4.value == "lean4"
    assert Prover.COQ.value == "coq"
    assert Prover("lean4") == Prover.LEAN4
