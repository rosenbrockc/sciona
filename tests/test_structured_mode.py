from __future__ import annotations

import pytest

from ageom.architect.handoff import CDGExport
from ageom.architect.models import AlgorithmicNode, ConceptType, NodeStatus
from ageom.cli import _run_structured_single_pass
from ageom.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
    PDGNode,
    Prover,
    VerificationLevel,
    VerificationResult,
)


def _make_cdg() -> CDGExport:
    return CDGExport(
        nodes=[
            AlgorithmicNode(
                node_id="n1",
                name="Design Filter",
                description="Design stable ECG bandpass coefficients.",
                concept_type=ConceptType.SIGNAL_FILTER,
                status=NodeStatus.ATOMIC,
                type_signature="FilterSpec -> Coefficients",
            ),
            AlgorithmicNode(
                node_id="n2",
                name="Apply Filter",
                description="Apply bandpass coefficients to ECG samples.",
                concept_type=ConceptType.SIGNAL_FILTER,
                status=NodeStatus.ATOMIC,
                type_signature="np.ndarray -> Coefficients -> np.ndarray",
            ),
        ],
        edges=[],
        metadata={"goal": "Detect heart rate from ECG"},
    )


def _verified_result(node: PDGNode) -> MatchResult:
    declaration = Declaration(
        name="algorithms.design_filter",
        type_signature="FilterSpec -> Coefficients",
        prover=Prover.PYTHON,
    )
    candidate = CandidateMatch(
        declaration=declaration,
        score=0.9,
        retrieval_method="lexical",
    )
    verification = VerificationResult(
        candidate=candidate,
        verified=True,
        verification_level=VerificationLevel.TYPE_CHECKED,
    )
    return MatchResult(
        pdg_node=node,
        verified_match=verification,
        all_candidates=[candidate],
        all_verifications=[verification],
    )


def _failed_result(node: PDGNode) -> MatchResult:
    declaration = Declaration(
        name="algorithms.apply_filter",
        type_signature="np.ndarray -> np.ndarray",
        prover=Prover.PYTHON,
    )
    candidate = CandidateMatch(
        declaration=declaration,
        score=0.5,
        retrieval_method="lexical",
    )
    verification = VerificationResult(
        candidate=candidate,
        verified=False,
        error_message="missing coefficients input",
        verification_level=VerificationLevel.UNVERIFIED,
    )
    return MatchResult(
        pdg_node=node,
        verified_match=None,
        all_candidates=[candidate],
        all_verifications=[verification],
    )


@pytest.mark.asyncio
async def test_run_structured_single_pass_matches_each_leaf_once():
    class _FakeHunter:
        def __init__(self) -> None:
            self.nodes: list[PDGNode] = []

        async def find_match(self, node: PDGNode) -> MatchResult:
            self.nodes.append(node)
            if node.predicate_id == "n1":
                return _verified_result(node)
            return _failed_result(node)

    hunter = _FakeHunter()
    result = await _run_structured_single_pass(
        _make_cdg(),
        prover=Prover.PYTHON,
        hunter=hunter,
    )

    assert result.rounds_used == 1
    assert [node.predicate_id for node in hunter.nodes] == ["n1", "n2"]
    assert len(result.match_results) == 2
    assert sum(1 for row in result.match_results if row.success) == 1
    assert len(result.failures) == 1
    assert result.ungroundable == ["n2"]
