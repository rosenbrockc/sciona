"""Tests for the Verification Oracle components."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from sciona.judge.models import CompilerFeedback
from sciona.types import CandidateMatch, Declaration, PDGNode, Prover


class TestCompilerFeedback:
    def test_success_no_errors(self):
        fb = CompilerFeedback(raw_output="ok")
        assert fb.success is True

    def test_failure_with_errors(self):
        fb = CompilerFeedback(raw_output="error", errors=["type mismatch"])
        assert fb.success is False

    def test_failure_with_remaining_goals(self):
        fb = CompilerFeedback(raw_output="goals", goals_remaining=["⊢ 1 + 1 = 2"])
        assert fb.success is False


class TestLeanOutputParsing:
    def test_parse_clean_output(self):
        from sciona.judge.lean_env import _parse_lean_output

        fb = _parse_lean_output("")
        assert fb.success is True

    def test_parse_error_output(self):
        from sciona.judge.lean_env import _parse_lean_output

        raw = "file:1:0: error: type mismatch\n  expected Nat\n  got Bool"
        fb = _parse_lean_output(raw)
        assert fb.success is False
        assert len(fb.errors) >= 1

    def test_parse_unsolved_goals(self):
        from sciona.judge.lean_env import _parse_lean_output

        raw = "unsolved goals\n⊢ 1 + 1 = 2"
        fb = _parse_lean_output(raw)
        assert fb.success is False
        assert len(fb.goals_remaining) >= 1


class TestVerificationOracleImpl:
    @pytest.fixture
    def oracle(self):
        from sciona.judge.checker import VerificationOracleImpl

        mock_lean = AsyncMock()
        mock_lean.check_term = AsyncMock(return_value=(True, "ok"))
        return VerificationOracleImpl(lean_env=mock_lean)

    @pytest.fixture
    def pdg_node(self):
        return PDGNode(
            predicate_id="p1",
            statement="∀ (n m : ℕ), n + m = m + n",
            prover=Prover.LEAN4,
        )

    @pytest.fixture
    def candidate(self):
        decl = Declaration(
            name="Nat.add_comm",
            type_signature="∀ (n m : ℕ), n + m = m + n",
            prover=Prover.LEAN4,
        )
        return CandidateMatch(
            declaration=decl, score=0.95, retrieval_method="embedding"
        )

    @pytest.mark.asyncio
    async def test_verify_candidate_success(self, oracle, pdg_node, candidate):
        result = await oracle.verify_candidate(pdg_node, candidate)
        assert result.verified is True
        assert result.proof_term == "@Nat.add_comm"

    @pytest.mark.asyncio
    async def test_verify_candidate_failure(self, pdg_node, candidate):
        from sciona.judge.checker import VerificationOracleImpl

        mock_lean = AsyncMock()
        mock_lean.check_term = AsyncMock(return_value=(False, "type mismatch"))
        oracle = VerificationOracleImpl(lean_env=mock_lean)

        result = await oracle.verify_candidate(pdg_node, candidate)
        assert result.verified is False
        assert "type mismatch" in result.error_message

    @pytest.mark.asyncio
    async def test_verify_candidates_short_circuits(self, pdg_node):
        from sciona.judge.checker import VerificationOracleImpl

        mock_lean = AsyncMock()
        # First fails, second succeeds
        mock_lean.check_term = AsyncMock(side_effect=[(False, "error"), (True, "ok")])
        oracle = VerificationOracleImpl(lean_env=mock_lean)

        decl1 = Declaration(name="wrong", type_signature="Bool", prover=Prover.LEAN4)
        decl2 = Declaration(name="right", type_signature="correct", prover=Prover.LEAN4)
        candidates = [
            CandidateMatch(declaration=decl1, score=0.8, retrieval_method="embedding"),
            CandidateMatch(declaration=decl2, score=0.9, retrieval_method="embedding"),
        ]

        results = await oracle.verify_candidates(pdg_node, candidates)
        assert len(results) == 2
        assert results[0].verified is False
        assert results[1].verified is True

    @pytest.mark.asyncio
    async def test_no_lean_env_raises(self, pdg_node, candidate):
        from sciona.judge.checker import VerificationOracleImpl

        oracle = VerificationOracleImpl()
        with pytest.raises(RuntimeError, match="LeanEnvironment not configured"):
            await oracle.verify_candidate(pdg_node, candidate)
