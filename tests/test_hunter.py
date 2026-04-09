"""Tests for the Hunter (Retrieval Agent) graph."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from sciona.architect.catalog import PrimitiveCatalog
from sciona.architect.models import AlgorithmicPrimitive, ConceptType, IOSpec
from sciona.shared_context import InMemorySharedContextStore
from sciona.shared_context import SharedContextMetrics
from sciona.hunter.state import HunterState
from sciona.telemetry import get_runtime_run, reset_telemetry_runtime, start_run, telemetry_scope
from sciona.types import (
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
    class _StubLLM:
        async def complete(self, system: str, user: str) -> str:
            system_lower = system.lower()
            if (
                "json array of integer indices" in system_lower
                or "rank" in system_lower
                or "score" in system_lower
            ):
                return rank_response
            if (
                "json array of strings" in system_lower
                or "generate search queries" in system_lower
            ):
                return queries_response
            if "return exactly three lines" in system_lower or "analy" in system_lower:
                return "The types don't match. Try searching for add_comm instead."
            return '["fallback_query"]'

        async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
            return await self.complete(system, user)

    return _StubLLM()


class TestHunterHappyPath:
    """Test: InitialSearch -> RankCandidates -> VerifyTopK -> End (verified)."""

    @pytest.mark.asyncio
    async def test_finds_correct_match_on_first_try(
        self, pdg_node, correct_decl, wrong_decl
    ):
        from sciona.hunter.graph import HunterAgent

        index = _make_mock_index([correct_decl, wrong_decl])
        oracle = _make_mock_oracle({"Nat.add_comm"})
        llm = _make_mock_llm()

        agent = HunterAgent(index=index, oracle=oracle, llm=llm, max_iterations=3)
        result = await agent.find_match(pdg_node)

        assert result.success
        assert result.verified_match is not None
        assert result.verified_match.candidate.declaration.name == "Nat.add_comm"

    @pytest.mark.asyncio
    async def test_records_hunter_metrics_in_telemetry(
        self, pdg_node, correct_decl, wrong_decl
    ):
        from sciona.hunter.graph import HunterAgent

        reset_telemetry_runtime()
        run_id = start_run("match", run_id="hunter-metrics")

        index = _make_mock_index([correct_decl, wrong_decl])
        oracle = _make_mock_oracle({"Nat.add_comm"})
        llm = _make_mock_llm()

        agent = HunterAgent(index=index, oracle=oracle, llm=llm, max_iterations=3)
        with telemetry_scope(run_id=run_id, stage="matching"):
            result = await agent.find_match(pdg_node)

        assert result.success
        snapshot = get_runtime_run(run_id)
        assert snapshot is not None
        metrics = snapshot["metadata"]["hunter_metrics"]
        assert metrics["search_iterations"] == 1
        assert metrics["new_candidates_total"] == 2
        assert metrics["candidate_pool_size"] == 2
        assert metrics["rank_calls"] == 1
        assert metrics["verify_batches"] == 1
        assert metrics["verified_candidates_total"] == 1
        assert metrics["verification_success_total"] == 1
        assert metrics["verification_failure_total"] == 0
        assert metrics["verified_matches"] == 1
        assert metrics["query_count"] == 1
        assert metrics["last_verified_candidate"] == "Nat.add_comm"

    @pytest.mark.asyncio
    async def test_deterministic_candidate_prior_can_override_bad_llm_ranking(self):
        from sciona.hunter.graph import HunterAgent

        pdg_node = PDGNode(
            predicate_id="ecg_detect",
            statement="Detect peaks in ECG signal",
            informal_desc="Find R-peaks in a conditioned ECG waveform",
            prover=Prover.PYTHON,
        )
        pronto = Declaration(
            name="ageoa.pronto.blip_filter.atoms.r_peak_detection",
            type_signature="(filtered: np.ndarray) -> np.ndarray",
            source_lib="ageoa.pronto.blip_filter.atoms",
            docstring="Detect peaks in a filtered signal.",
            raw_code="def r_peak_detection(filtered):\n    fs = 360.0\n    return filtered\n",
            prover=Prover.PYTHON,
        )
        biosppy = Declaration(
            name="ageoa.biosppy.ecg.r_peak_detection",
            type_signature="(filtered: np.ndarray, sampling_rate: float = 1000.0) -> np.ndarray",
            source_lib="ageoa.biosppy.ecg",
            docstring="Detect R-peaks in an ECG waveform.",
            raw_code=(
                "def r_peak_detection(filtered, sampling_rate=1000.0):\n"
                "    return hamilton_segmenter(signal=filtered, sampling_rate=sampling_rate)\n"
            ),
            prover=Prover.PYTHON,
        )

        index = _make_mock_index([pronto, biosppy])
        oracle = _make_mock_oracle({"ageoa.biosppy.ecg.r_peak_detection"})
        llm = _make_mock_llm(rank_response="[0, 1]")

        agent = HunterAgent(index=index, oracle=oracle, llm=llm, max_iterations=1)
        result = await agent.find_match(pdg_node)

        assert result.success
        assert result.verified_match is not None
        assert (
            result.verified_match.candidate.declaration.name
            == "ageoa.biosppy.ecg.r_peak_detection"
        )

    @pytest.mark.asyncio
    async def test_live_catalog_reconciles_stale_ageoa_declarations(self):
        from sciona.hunter.graph import HunterAgent

        pdg_node = PDGNode(
            predicate_id="ecg_detect",
            statement="Detect peaks in ECG signal",
            informal_desc="Find R-peaks in a conditioned ECG waveform",
            prover=Prover.PYTHON,
        )
        stale_dead = Declaration(
            name="ageoa.biosppy.ecg_hamilton.atoms.hamilton_segmentation",
            type_signature="(signal: np.ndarray, sampling_rate: int) -> np.ndarray",
            source_lib="ageoa.biosppy.ecg_hamilton.atoms",
            docstring="Dead stale declaration from an old index snapshot.",
            prover=Prover.PYTHON,
        )
        stale_live = Declaration(
            name="ageoa.biosppy.ecg.r_peak_detection",
            type_signature="(filtered: np.ndarray, state: ECGPipelineState) -> tuple[np.ndarray, ECGPipelineState]",
            source_lib="ageoa.biosppy.ecg",
            docstring="Old stateful ECG declaration.",
            prover=Prover.PYTHON,
        )

        catalog = PrimitiveCatalog()
        catalog.add(
            AlgorithmicPrimitive(
                name="r_peak_detection",
                source="ageo-atoms",
                category=ConceptType.ANALYSIS,
                description="Detect R-peaks from a filtered ECG waveform using the sampling rate.",
                inputs=[
                    IOSpec(name="filtered", type_desc="np.ndarray"),
                    IOSpec(
                        name="sampling_rate",
                        type_desc="float",
                        required=False,
                        default_value_repr="1000.0",
                    ),
                ],
                outputs=[IOSpec(name="rpeaks", type_desc="np.ndarray")],
                type_signature="(filtered: np.ndarray, sampling_rate: float? = 1000.0) -> np.ndarray",
            )
        )

        index = _make_mock_index([stale_dead, stale_live])
        oracle = _make_mock_oracle({"ageoa.biosppy.ecg.r_peak_detection"})
        llm = _make_mock_llm(rank_response="[0]")

        agent = HunterAgent(
            index=index,
            oracle=oracle,
            llm=llm,
            max_iterations=1,
            live_catalog=catalog,
        )
        result = await agent.find_match(pdg_node)

        assert result.success
        assert [c.declaration.name for c in result.all_candidates] == [
            "ageoa.biosppy.ecg.r_peak_detection"
        ]
        assert result.verified_match is not None
        assert "sampling_rate" in result.verified_match.candidate.declaration.type_signature

    @pytest.mark.asyncio
    async def test_matched_primitive_hint_seeds_exact_candidate_from_live_catalog(self):
        from sciona.hunter.graph import HunterAgent

        pdg_node = PDGNode(
            predicate_id="event_cleanup",
            statement="Remove implausible events before downstream rate estimation.",
            informal_desc="Clean event markers that create unstable intervals.",
            prover=Prover.PYTHON,
            context={"matched_primitive": "ageoa.biosppy.ecg.reject_outlier_intervals"},
        )

        catalog = PrimitiveCatalog()
        catalog.add(
            AlgorithmicPrimitive(
                name="reject_outlier_intervals",
                source="ageo-atoms",
                category=ConceptType.SIGNAL_FILTER,
                description="Remove events that induce implausible adjacent intervals.",
                inputs=[
                    IOSpec(name="rpeaks", type_desc="np.ndarray"),
                    IOSpec(name="sampling_rate", type_desc="float"),
                ],
                outputs=[IOSpec(name="rpeaks", type_desc="np.ndarray")],
                type_signature="(rpeaks: np.ndarray, sampling_rate: float) -> np.ndarray",
            )
        )

        index = _make_mock_index([])
        oracle = _make_mock_oracle({"ageoa.biosppy.ecg.reject_outlier_intervals"})
        llm = _make_mock_llm(rank_response="[0]")

        agent = HunterAgent(
            index=index,
            oracle=oracle,
            llm=llm,
            max_iterations=1,
            live_catalog=catalog,
        )
        result = await agent.find_match(pdg_node)

        assert result.success
        assert result.verified_match is not None
        assert (
            result.verified_match.candidate.declaration.name
            == "ageoa.biosppy.ecg.reject_outlier_intervals"
        )

    @pytest.mark.asyncio
    async def test_stage_prior_penalizes_rate_atom_on_detect_leaf(self):
        from sciona.hunter.graph import HunterAgent

        pdg_node = PDGNode(
            predicate_id="ecg_detect",
            statement="Detect salient peaks or events in the conditioned ECG signal.",
            informal_desc="Find R-peaks from a filtered ECG waveform.",
            prover=Prover.PYTHON,
            context={"matched_primitive": "detect_peaks_in_signal"},
        )
        detector = Declaration(
            name="ageoa.biosppy.ecg.r_peak_detection",
            type_signature="(filtered: np.ndarray, sampling_rate: float) -> np.ndarray",
            source_lib="ageoa.biosppy.ecg",
            docstring="Detect R-peaks in a filtered ECG waveform.",
            prover=Prover.PYTHON,
        )
        rate = Declaration(
            name="ageoa.biosppy.ecg.heart_rate_computation",
            type_signature="(rpeaks: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]",
            source_lib="ageoa.biosppy.ecg",
            docstring="Compute heart rate from R-peak intervals.",
            prover=Prover.PYTHON,
        )

        index = _make_mock_index([rate, detector])
        oracle = _make_mock_oracle({"ageoa.biosppy.ecg.r_peak_detection"})
        llm = _make_mock_llm(rank_response="[0, 1]")

        agent = HunterAgent(index=index, oracle=oracle, llm=llm, max_iterations=1)
        result = await agent.find_match(pdg_node)

        assert result.success
        assert result.verified_match is not None
        assert result.verified_match.candidate.declaration.name == detector.name

    @pytest.mark.asyncio
    async def test_stage_prior_penalizes_filter_atom_on_detect_leaf(self):
        from sciona.hunter.graph import HunterAgent

        pdg_node = PDGNode(
            predicate_id="ecg_detect",
            statement="Detect salient peaks or events in the conditioned ECG signal.",
            informal_desc="Find R-peaks from a filtered ECG waveform.",
            prover=Prover.PYTHON,
            context={"matched_primitive": "detect_peaks_in_signal"},
        )
        detector = Declaration(
            name="ageoa.pronto.blip_filter.atoms.r_peak_detection",
            type_signature="(filtered: np.ndarray) -> np.ndarray",
            source_lib="ageoa.pronto.blip_filter.atoms",
            docstring="Detect R-peaks in a filtered ECG waveform.",
            prover=Prover.PYTHON,
        )
        filt = Declaration(
            name="ageoa.pronto.blip_filter.atoms.bandpass_filter",
            type_signature="(signal: np.ndarray) -> np.ndarray",
            source_lib="ageoa.pronto.blip_filter.atoms",
            docstring="Apply bandpass filtering to an ECG signal.",
            prover=Prover.PYTHON,
        )

        index = _make_mock_index([filt, detector])
        oracle = _make_mock_oracle({"ageoa.pronto.blip_filter.atoms.r_peak_detection"})
        llm = _make_mock_llm(rank_response="[0, 1]")

        agent = HunterAgent(index=index, oracle=oracle, llm=llm, max_iterations=1)
        result = await agent.find_match(pdg_node)

        assert result.success
        assert result.verified_match is not None
        assert result.verified_match.candidate.declaration.name == detector.name

    def test_stage_inference_treats_compute_event_rate_as_rate_leaf(self):
        from sciona.hunter.nodes import _infer_stage_tokens

        pdg_node = PDGNode(
            predicate_id="ecg_rate",
            statement="Compute a target rate or cadence from inter-event intervals.",
            informal_desc="Estimate heart rate from detected events.",
            prover=Prover.PYTHON,
            context={"matched_primitive": "compute_event_rate"},
        )

        stage, _ = _infer_stage_tokens(pdg_node)
        assert stage == "rate"

    def test_stage_inference_treats_filter_signal_for_detection_as_filter_leaf(self):
        from sciona.hunter.nodes import _infer_stage_tokens

        pdg_node = PDGNode(
            predicate_id="ecg_filter",
            statement="Filter or denoise the raw signal into a conditioned waveform for downstream event extraction.",
            informal_desc="Prepare the ECG waveform for downstream peak detection.",
            prover=Prover.PYTHON,
            context={"matched_primitive": "filter_signal_for_detection"},
        )

        stage, _ = _infer_stage_tokens(pdg_node)
        assert stage == "filter"

    @pytest.mark.asyncio
    async def test_stage_prior_prefers_rate_atom_over_template_atom_on_rate_leaf(self):
        from sciona.hunter.graph import HunterAgent

        pdg_node = PDGNode(
            predicate_id="ecg_rate",
            statement="Compute a target rate or cadence from inter-event intervals.",
            informal_desc="Estimate heart rate from detected events.",
            prover=Prover.PYTHON,
            context={"matched_primitive": "compute_event_rate"},
        )
        template = Declaration(
            name="ageoa.biosppy.ecg.template_extraction",
            type_signature="(filtered: np.ndarray, rpeaks: np.ndarray) -> tuple[np.ndarray, np.ndarray]",
            source_lib="ageoa.biosppy.ecg",
            docstring="Extract individual heartbeat waveform templates around each R-peak.",
            prover=Prover.PYTHON,
        )
        rate = Declaration(
            name="ageoa.biosppy.ecg.heart_rate_computation",
            type_signature="(rpeaks: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]",
            source_lib="ageoa.biosppy.ecg",
            docstring="Compute heart rate from R-peak intervals.",
            prover=Prover.PYTHON,
        )

        index = _make_mock_index([template, rate])
        oracle = _make_mock_oracle({"ageoa.biosppy.ecg.heart_rate_computation"})
        llm = _make_mock_llm(rank_response="[0, 1]")

        agent = HunterAgent(index=index, oracle=oracle, llm=llm, max_iterations=1)
        result = await agent.find_match(pdg_node)

        assert result.success
        assert result.verified_match is not None
        assert result.verified_match.candidate.declaration.name == rate.name

    @pytest.mark.asyncio
    async def test_stage_prior_prefers_filter_atom_over_template_atom_on_filter_leaf(self):
        from sciona.hunter.graph import HunterAgent

        pdg_node = PDGNode(
            predicate_id="ecg_filter",
            statement="Filter or denoise the raw signal into a conditioned waveform for downstream event extraction.",
            informal_desc="Prepare the ECG waveform for downstream peak detection.",
            prover=Prover.PYTHON,
            context={"matched_primitive": "filter_signal_for_detection"},
        )
        template = Declaration(
            name="ageoa.biosppy.ecg.template_extraction",
            type_signature="(filtered: np.ndarray, rpeaks: np.ndarray) -> tuple[np.ndarray, np.ndarray]",
            source_lib="ageoa.biosppy.ecg",
            docstring="Extract individual heartbeat waveform templates around each R-peak.",
            prover=Prover.PYTHON,
        )
        filt = Declaration(
            name="ageoa.biosppy.ecg.bandpass_filter",
            type_signature="(signal: np.ndarray) -> np.ndarray",
            source_lib="ageoa.biosppy.ecg",
            docstring="Apply FIR bandpass filtering to an ECG waveform.",
            prover=Prover.PYTHON,
        )

        index = _make_mock_index([template, filt])
        oracle = _make_mock_oracle({"ageoa.biosppy.ecg.bandpass_filter"})
        llm = _make_mock_llm(rank_response="[0, 1]")

        agent = HunterAgent(index=index, oracle=oracle, llm=llm, max_iterations=1)
        result = await agent.find_match(pdg_node)

        assert result.success
        assert result.verified_match is not None
        assert result.verified_match.candidate.declaration.name == filt.name


class TestHunterRefinement:
    """Test: first verify fails -> reformulate -> second search finds match."""

    @pytest.mark.asyncio
    async def test_refines_and_finds_match(self, pdg_node, correct_decl, wrong_decl):
        from sciona.hunter.graph import HunterAgent

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

    @pytest.mark.asyncio
    async def test_failure_context_metrics_track_search_and_write(
        self, pdg_node, correct_decl, wrong_decl
    ):
        from sciona.hunter.graph import HunterAgent

        call_count = 0

        def search_by_embedding(query, k=10):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [(wrong_decl, 0.9)]
            return [(correct_decl, 0.95)]

        index = AsyncMock()
        index.search_by_embedding = search_by_embedding
        index.search_by_type = lambda sig, k=10: []

        oracle = _make_mock_oracle({"Nat.add_comm"})
        llm = _make_mock_llm(queries_response='["Nat.add_comm addition commutative"]')
        store = InMemorySharedContextStore()
        metrics = SharedContextMetrics()
        await store.put(
            "hunter/failure",
            "Predicate: forall n m, n + m = m + n\nRejected: Nat.mul_comm\nSignal: type mismatch",
        )

        agent = HunterAgent(
            index=index,
            oracle=oracle,
            llm=llm,
            max_iterations=5,
            shared_context=store,
            shared_context_metrics=metrics,
        )
        result = await agent.find_match(pdg_node)

        assert result.success
        snap = metrics.snapshot()
        assert snap["failure_searches_total"] >= 1
        assert snap["failure_search_hits"] >= 1
        assert snap["failure_puts_total"] >= 1
        assert snap["failure_injected_blocks"] >= 1


class TestHunterBudgetExhaustion:
    """Test: max_iterations reached -> End with no verified match."""

    @pytest.mark.asyncio
    async def test_exhausts_budget(self, pdg_node, wrong_decl):
        from sciona.hunter.graph import HunterAgent

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
        from sciona.hunter.graph import HunterAgent

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
        from sciona.hunter.graph import HunterAgent

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
        from sciona.hunter.graph import HunterAgent

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
        from sciona.hunter.graph import HunterAgent

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
