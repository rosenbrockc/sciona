"""Tests for the Hunter (Retrieval Agent) graph."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ageom.shared_context import InMemorySharedContextStore
from ageom.hunter.state import HunterState
from ageom.types import (
    Declaration,
    PDGNode,
    Prover,
    VerificationResult,
)


@pytest.fixture
def pdg_node():
    return PDGNode(
        predicate_id="p1",
        statement="∀ (n m : ℕ), n + m = m + n",
        informal_desc="commutativity of addition on natural numbers",
        prover=Prover.LEAN4,
    )


@pytest.fixture
def correct_decl():
    return Declaration(
        name="Nat.add_comm",
        type_signature="∀ (n m : ℕ), n + m = m + n",
        prover=Prover.LEAN4,
    )


@pytest.fixture
def wrong_decl():
    return Declaration(
        name="Nat.mul_comm",
        type_signature="∀ (n m : ℕ), n * m = m * n",
        prover=Prover.LEAN4,
    )


def _make_mock_index(declarations: list[Declaration]):
    """Create a mock SemanticIndex returning the given declarations."""
    index = AsyncMock()
    index.search_by_embedding = lambda query, k=10: [
        (d, 1.0 - i * 0.1) for i, d in enumerate(declarations[:k])
    ]
    index.search_by_type = lambda sig, k=10: declarations[:k]
    index.get_declaration = lambda name: next(
        (d for d in declarations if d.name == name), None
    )
    return index


def _make_mock_oracle(verified_names: set[str]):
    """Create a mock VerificationOracle that verifies only specific names."""

    async def verify_candidate(pdg_node, candidate):
        is_verified = candidate.declaration.name in verified_names
        return VerificationResult(
            candidate=candidate,
            verified=is_verified,
            compiler_output="ok" if is_verified else "type mismatch",
            proof_term=f"@{candidate.declaration.name}" if is_verified else "",
            error_message="" if is_verified else "type mismatch",
        )

    async def verify_candidates(pdg_node, candidates):
        results = []
        for c in candidates:
            r = await verify_candidate(pdg_node, c)
            results.append(r)
            if r.verified:
                break
        return results

    oracle = AsyncMock()
    oracle.verify_candidate = verify_candidate
    oracle.verify_candidates = verify_candidates
    return oracle


def _make_mock_llm(
    rank_response: str = "[0, 1, 2]", queries_response: str = '["query1"]'
):
    """Create a mock LLMClient."""
    llm = AsyncMock()

    async def complete(system: str, user: str) -> str:
        system_lower = system.lower()
        if "json array of integer indices" in system_lower or "rank" in system_lower or "score" in system_lower:
            return rank_response
        elif "json array of strings" in system_lower or "generate search queries" in system_lower:
            return queries_response
        elif "return exactly three lines" in system_lower or "analy" in system_lower:
            return "The types don't match. Try searching for add_comm instead."
        return '["fallback_query"]'

    llm.complete = complete
    return llm


class TestHunterHappyPath:
    """Test: InitialSearch -> RankCandidates -> VerifyTopK -> End (verified)."""

    @pytest.mark.asyncio
    async def test_finds_correct_match_on_first_try(
        self, pdg_node, correct_decl, wrong_decl
    ):
        from ageom.hunter.graph import HunterAgent

        index = _make_mock_index([correct_decl, wrong_decl])
        oracle = _make_mock_oracle({"Nat.add_comm"})
        llm = _make_mock_llm()

        agent = HunterAgent(index=index, oracle=oracle, llm=llm, max_iterations=3)
        result = await agent.find_match(pdg_node)

        assert result.success
        assert result.verified_match is not None
        assert result.verified_match.candidate.declaration.name == "Nat.add_comm"


class TestHunterRefinement:
    """Test: first verify fails -> reformulate -> second search finds match."""

    @pytest.mark.asyncio
    async def test_refines_and_finds_match(self, pdg_node, correct_decl, wrong_decl):
        from ageom.hunter.graph import HunterAgent

        call_count = 0

        def search_by_embedding(query, k=10):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First search returns only wrong declaration
                return [(wrong_decl, 0.9)]
            else:
                # After reformulation, returns correct one
                return [(correct_decl, 0.95), (wrong_decl, 0.8)]

        index = AsyncMock()
        index.search_by_embedding = search_by_embedding
        index.search_by_type = lambda sig, k=10: []

        oracle = _make_mock_oracle({"Nat.add_comm"})
        llm = _make_mock_llm(queries_response='["Nat.add_comm addition commutative"]')

        agent = HunterAgent(index=index, oracle=oracle, llm=llm, max_iterations=5)
        result = await agent.find_match(pdg_node)

        assert result.success
        assert result.verified_match is not None
        assert result.verified_match.candidate.declaration.name == "Nat.add_comm"


class TestHunterBudgetExhaustion:
    """Test: max_iterations reached -> End with no verified match."""

    @pytest.mark.asyncio
    async def test_exhausts_budget(self, pdg_node, wrong_decl):
        from ageom.hunter.graph import HunterAgent

        index = _make_mock_index([wrong_decl])
        oracle = _make_mock_oracle(set())  # Nothing verifies
        llm = _make_mock_llm()

        agent = HunterAgent(
            index=index, oracle=oracle, llm=llm, max_iterations=2, top_k_verify=1
        )
        result = await agent.find_match(pdg_node)

        assert not result.success
        assert result.verified_match is None
        assert len(result.all_verifications) > 0


class TestHunterNoCandidates:
    """Test: no candidates found -> immediate End."""

    @pytest.mark.asyncio
    async def test_no_candidates(self, pdg_node):
        from ageom.hunter.graph import HunterAgent

        index = _make_mock_index([])
        oracle = _make_mock_oracle(set())
        llm = _make_mock_llm()

        agent = HunterAgent(index=index, oracle=oracle, llm=llm)
        result = await agent.find_match(pdg_node)

        assert not result.success
        assert len(result.all_candidates) == 0


class TestHunterState:
    def test_initial_state(self, pdg_node):
        state = HunterState(pdg_node=pdg_node)
        assert state.iteration == 0
        assert state.verified_match is None
        assert state.candidates_found == []


class _GrammarAwareLLM:
    def __init__(self):
        self.complete_calls = 0
        self.grammar_calls = 0

    async def complete(self, system: str, user: str) -> str:
        self.complete_calls += 1
        s = system.lower()
        if "analy" in s:
            return "analysis"
        return "[]"

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        self.grammar_calls += 1
        system_lower = system.lower()
        if "json array of integer indices" in system_lower or "rank" in system_lower or "score" in system_lower:
            return "[0]"
        if "json array of strings" in system_lower or "generate search queries" in system_lower:
            return '["q1", "q2", "q3", "q4"]'
        return "[]"


class TestHunterSpeculativeLocal:
    @pytest.mark.asyncio
    async def test_uses_gbnf_and_query_batching(self, pdg_node, wrong_decl):
        from ageom.hunter.graph import HunterAgent

        search_calls = 0

        def search_by_embedding(query, k=10):
            nonlocal search_calls
            search_calls += 1
            return [(wrong_decl, 0.9)]

        index = AsyncMock()
        index.search_by_embedding = search_by_embedding
        index.search_by_type = lambda sig, k=10: []

        oracle = _make_mock_oracle(set())  # force reformulation path
        llm = _GrammarAwareLLM()

        agent = HunterAgent(
            index=index,
            oracle=oracle,
            llm=llm,
            max_iterations=1,
            top_k_verify=1,
            search_k=1,
            mode="speculative_local",
            use_gbnf=True,
            query_batch_size=4,
            top_k_per_query=1,
            max_candidates_total=100,
        )
        result = await agent.find_match(pdg_node)

        assert not result.success
        assert llm.grammar_calls >= 2  # rank + reformulate
        # First InitialSearch uses one query; second uses the 4-query batch.
        assert search_calls >= 5


class _CapturePromptLLM:
    def __init__(self) -> None:
        self.rank_users: list[str] = []

    async def complete(self, system: str, user: str) -> str:
        system_lower = system.lower()
        if "json array of integer indices" in system_lower or "rank" in system_lower or "score" in system_lower:
            self.rank_users.append(user)
            return "[0]"
        if "json array of strings" in system_lower or "generate search queries" in system_lower:
            return '["query1"]'
        if "return exactly three lines" in system_lower or "analy" in system_lower:
            return "analysis"
        return "[]"

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)


class TestHunterSharedContext:
    @pytest.mark.asyncio
    async def test_writes_verified_match_to_shared_context(
        self, pdg_node, correct_decl, wrong_decl
    ):
        from ageom.hunter.graph import HunterAgent

        store = InMemorySharedContextStore()
        index = _make_mock_index([correct_decl, wrong_decl])
        oracle = _make_mock_oracle({"Nat.add_comm"})
        llm = _make_mock_llm()

        agent = HunterAgent(
            index=index,
            oracle=oracle,
            llm=llm,
            shared_context=store,
            run_id="test-run",
        )
        result = await agent.find_match(pdg_node)

        assert result.success
        records = await store.recent("hunter/test-run/success", limit=3)
        assert records
        assert any("Nat.add_comm" in rec.text for rec in records)

    @pytest.mark.asyncio
    async def test_injects_shared_context_into_rank_prompt(self, pdg_node, wrong_decl):
        from ageom.hunter.graph import HunterAgent

        store = InMemorySharedContextStore()
        await store.put(
            "hunter/run-ctx/success",
            (
                "Predicate: ∀ (n m : ℕ), n + m = m + n\n"
                "Matched: Nat.add_comm\nType: ∀ (n m : ℕ), n + m = m + n"
            ),
        )

        llm = _CapturePromptLLM()
        index = _make_mock_index([wrong_decl])
        oracle = _make_mock_oracle(set())
        agent = HunterAgent(
            index=index,
            oracle=oracle,
            llm=llm,
            max_iterations=0,
            shared_context=store,
            run_id="run-ctx",
        )
        await agent.find_match(pdg_node)

        assert llm.rank_users
        assert "## Shared Context" in llm.rank_users[0]
