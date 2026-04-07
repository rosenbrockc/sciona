from __future__ import annotations

import pytest

from sciona.cli import _build_rapid_direct_cdg, _run_rapid_direct_match
from sciona.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
    PDGNode,
    Prover,
    VerificationLevel,
    VerificationResult,
)


def _verified_match_result(node: PDGNode | None = None) -> MatchResult:
    pdg_node = node or PDGNode(
        predicate_id="goal_0",
        statement="Detect heart rate from ECG",
        informal_desc="rapid direct test",
        prover=Prover.PYTHON,
    )
    declaration = Declaration(
        name="algorithms.detect_heart_rate",
        type_signature="np.ndarray -> float",
        conceptual_summary="detect heart rate from ecg",
        prover=Prover.PYTHON,
    )
    candidate = CandidateMatch(
        declaration=declaration,
        score=0.92,
        retrieval_method="lexical",
    )
    verification = VerificationResult(
        candidate=candidate,
        verified=True,
        compiler_output="ok",
        proof_term="algorithms.detect_heart_rate",
        verification_level=VerificationLevel.TYPE_CHECKED,
    )
    return MatchResult(
        pdg_node=pdg_node,
        verified_match=verification,
        all_candidates=[candidate],
        all_verifications=[verification],
    )


def _failed_match_result(node: PDGNode | None = None) -> MatchResult:
    pdg_node = node or PDGNode(
        predicate_id="goal_0",
        statement="Detect heart rate from ECG",
        informal_desc="rapid direct test",
        prover=Prover.PYTHON,
    )
    declaration = Declaration(
        name="algorithms.filter_signal",
        type_signature="np.ndarray -> np.ndarray",
        conceptual_summary="bandpass filter",
        prover=Prover.PYTHON,
    )
    candidate = CandidateMatch(
        declaration=declaration,
        score=0.41,
        retrieval_method="lexical",
    )
    verification = VerificationResult(
        candidate=candidate,
        verified=False,
        compiler_output="type mismatch",
        error_message="expected scalar heart rate output",
        verification_level=VerificationLevel.UNVERIFIED,
    )
    return MatchResult(
        pdg_node=pdg_node,
        verified_match=None,
        all_candidates=[candidate],
        all_verifications=[verification],
    )


def test_build_rapid_direct_cdg_success_is_atomic():
    cdg = _build_rapid_direct_cdg(
        "Detect heart rate from ECG",
        Prover.PYTHON,
        _verified_match_result(),
    )

    assert cdg.metadata["rapid_direct_path"] is True
    assert cdg.metadata["matched_directly"] is True
    assert len(cdg.nodes) == 1
    assert cdg.nodes[0].status.value == "atomic"
    assert cdg.nodes[0].matched_primitive == "algorithms.detect_heart_rate"
    assert cdg.nodes[0].type_signature == "np.ndarray -> float"


def test_build_rapid_direct_cdg_failure_is_blocked():
    cdg = _build_rapid_direct_cdg(
        "Detect heart rate from ECG",
        Prover.PYTHON,
        _failed_match_result(),
    )

    assert cdg.metadata["rapid_direct_path"] is True
    assert cdg.metadata["matched_directly"] is False
    assert "architect_error" in cdg.metadata
    assert len(cdg.nodes) == 1
    assert cdg.nodes[0].status.value == "blocked"
    assert "expected scalar heart rate output" in cdg.nodes[0].critic_notes


@pytest.mark.asyncio
async def test_run_rapid_direct_match_wraps_failure_as_single_round():
    class _FakeHunter:
        def __init__(self) -> None:
            self.calls = 0
            self.last_node = None

        async def find_match(self, node: PDGNode) -> MatchResult:
            self.calls += 1
            self.last_node = node
            return _failed_match_result(node)

    hunter = _FakeHunter()
    result = await _run_rapid_direct_match(
        "Compute a running average for a numeric stream",
        prover=Prover.PYTHON,
        hunter=hunter,
    )

    assert hunter.calls == 1
    assert hunter.last_node is not None
    assert hunter.last_node.context["rapid_direct_path"] == "true"
    assert result.rounds_used == 1
    assert len(result.match_results) == 1
    assert result.match_results[0].success is False
    assert result.cdg.metadata["rapid_direct_path"] is True
    assert result.ungroundable == ["goal_0"]


@pytest.mark.asyncio
async def test_run_rapid_direct_match_can_disable_curated_signal_event_rate_shortcut():
    class _FakeHunter:
        def __init__(self) -> None:
            self.calls = 0
            self.last_node = None

        async def find_match(self, node: PDGNode) -> MatchResult:
            self.calls += 1
            self.last_node = node
            return _verified_match_result(node)

    hunter = _FakeHunter()
    result = await _run_rapid_direct_match(
        "Detect heart rate from ECG",
        prover=Prover.PYTHON,
        hunter=hunter,
        allow_curated_signal_event_rate_shortcut=False,
    )

    assert hunter.calls == 1
    assert hunter.last_node is not None
    assert result.cdg.metadata["matched_directly"] is True
    assert len(result.cdg.nodes) == 1
    assert result.cdg.nodes[0].matched_primitive == "algorithms.detect_heart_rate"
