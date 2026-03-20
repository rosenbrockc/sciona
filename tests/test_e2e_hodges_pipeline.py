"""Full-pipeline E2E test: Hodges EMG onset detection (Rounds 1-3).

Exercises the complete pipeline:
  Natural language prompt → Architect decomposition → Hunter matching →
  Synthesizer assembly → runnable Python code → validated against a
  reference Hodges implementation on synthetic EMG data.

No live LLM, no Memgraph, no external services required.
"""

from __future__ import annotations

import ast
import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("langgraph") is None:
    pytest.skip("requires langgraph", allow_module_level=True)

from sciona.architect.handoff import CDGExport, to_pdg_nodes
from sciona.architect.models import (
    AlgorithmicNode,
    DependencyEdge,
    NodeStatus,
)
from sciona.architect.nodes import decompose_node
from sciona.synthesizer.assembler import Assembler
from sciona.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
    PDGNode,
    Prover,
    VerificationResult,
)

# Re-use fixtures from the existing Hodges E2E test
from tests.test_retrieval_e2e_hodges import (
    HodgesArchitectLLM,
    _AcceptAllCatalog,
    _build_hodges_state,
    _hodges_node,
    _make_config,
)


# ---------------------------------------------------------------------------
# Hodges algorithm implementation (helpers for exec'd code)
# ---------------------------------------------------------------------------


class _HodgesImpl:
    """Namespace of static helpers implementing the Hodges algorithm stages.

    Each method handles the tuple-passing behaviour of the assembler:
    edges from the same source all pass ``source_name_result`` (the full
    return value), so multi-output stages return tuples and downstream
    consumers destructure internally.
    """

    @staticmethod
    def estimate_rest_baseline(rest_signal, sampling_rate):
        """Stage 1: compute rest-segment mean and std."""
        return float(np.mean(rest_signal)), float(np.std(rest_signal))

    @staticmethod
    def remove_baseline(signal, baseline_stats):
        """Stage 2: subtract rest mean.  *baseline_stats* is (mean, std)."""
        rest_mean = baseline_stats[0]
        return signal - rest_mean

    @staticmethod
    def compute_test_statistic(centered, baseline_stats):
        """Stage 3: h(n) = |centered| / std.  *baseline_stats* is (mean, std)."""
        rest_std = baseline_stats[1]
        return np.abs(centered) / rest_std

    @staticmethod
    def smooth_statistic(test_stat, sampling_rate):
        """Stage 4: 50 ms moving-average smoothing."""
        window = max(int(0.05 * sampling_rate), 3)
        kernel = np.ones(window) / window
        return np.convolve(test_stat, kernel, mode="same")

    @staticmethod
    def threshold_crossing(smoothed, threshold):
        """Stage 5: simple threshold crossing → (onsets, offsets)."""
        above = smoothed > threshold
        diff = np.diff(above.astype(int))
        onsets = np.where(diff == 1)[0] + 1
        offsets = np.where(diff == -1)[0] + 1
        return onsets, offsets

    @staticmethod
    def merge_adjacent(crossing_result_onsets, crossing_result_offsets):
        """Stage 6: pass-through merge (identity for single-burst data).

        Both args arrive as the *full* tuple from threshold_crossing, so
        destructure the first one.
        """
        if isinstance(crossing_result_onsets, tuple):
            onsets = crossing_result_onsets[0]
            offsets = crossing_result_onsets[1]
        else:
            onsets = crossing_result_onsets
            offsets = crossing_result_offsets
        return onsets, offsets


# ---------------------------------------------------------------------------
# Reference implementation (independent of pipeline)
# ---------------------------------------------------------------------------


def _reference_hodges(
    signal: np.ndarray,
    rest_signal: np.ndarray,
    fs: float,
    threshold: float = 3.0,
) -> np.ndarray:
    """Standalone Hodges onset detector for cross-validation."""
    rest_mean = np.mean(rest_signal)
    rest_std = np.std(rest_signal)
    centered = signal - rest_mean
    h = np.abs(centered) / rest_std
    window = max(int(0.05 * fs), 3)
    smoothed = np.convolve(h, np.ones(window) / window, mode="same")
    above = smoothed > threshold
    onsets = np.where(np.diff(above.astype(int)) == 1)[0] + 1
    return onsets


# ---------------------------------------------------------------------------
# Synthetic EMG generator
# ---------------------------------------------------------------------------


def _make_synthetic_emg(
    fs: int = 1000,
    rest_dur: float = 1.0,
    active_dur: float = 0.5,
    seed: int = 42,
):
    """Generate a simple rest→burst EMG signal.

    Returns (signal, rest_signal, fs, true_onset_sample).
    """
    rng = np.random.default_rng(seed)
    n_rest = int(rest_dur * fs)
    n_active = int(active_dur * fs)

    rest = 0.01 * rng.standard_normal(n_rest)
    active = 0.5 * rng.standard_normal(n_active)
    signal = np.concatenate([rest, active])
    return signal, rest.copy(), float(fs), n_rest


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

# Declaration-name mapping for each Hodges stage
_STAGE_DECLS: dict[str, str] = {
    "Estimate Rest Baseline Statistics": "_hodges_impl.estimate_rest_baseline",
    "Remove Baseline Offset": "_hodges_impl.remove_baseline",
    "Compute Normalised Test Statistic": "_hodges_impl.compute_test_statistic",
    "Smooth Test Statistic": "_hodges_impl.smooth_statistic",
    "Threshold Crossing State Machine": "_hodges_impl.threshold_crossing",
    "Merge Adjacent Events": "_hodges_impl.merge_adjacent",
}


def _make_match_result(node: AlgorithmicNode) -> MatchResult:
    """Build a MatchResult mapping *node* to its _hodges_impl function."""
    decl_name = _STAGE_DECLS[node.name]
    type_sig = node.type_signature or "Any -> Any"
    decl = Declaration(
        name=decl_name,
        type_signature=type_sig,
        prover=Prover.PYTHON,
    )
    candidate = CandidateMatch(
        declaration=decl, score=1.0, retrieval_method="test_mock"
    )
    vr = VerificationResult(
        candidate=candidate, verified=True, proof_term=decl_name
    )
    return MatchResult(
        pdg_node=PDGNode(
            predicate_id=node.node_id,
            statement=type_sig,
            prover=Prover.PYTHON,
        ),
        verified_match=vr,
        all_candidates=[candidate],
        all_verifications=[vr],
    )


async def _run_phase_a():
    """Phase A: Architect decomposition → (sub-nodes, edges, parent node)."""
    node = _hodges_node()
    state = _build_hodges_state(node)
    config = _make_config(_AcceptAllCatalog(), HodgesArchitectLLM())
    result = await decompose_node(state, config)
    sub_nodes = result["nodes"]
    edges = result["edges"]
    return sub_nodes, edges, node


def _build_cdg(sub_nodes, edges, parent_node):
    """Phase B: Build a CDGExport from decomposition output."""
    # Parent becomes DECOMPOSED with children list and type_signature
    root = parent_node.model_copy(
        update={
            "status": NodeStatus.DECOMPOSED,
            "children": [n.node_id for n in sub_nodes],
            "type_signature": (
                "(signal: ndarray, rest_signal: ndarray, "
                "sampling_rate: float, threshold: float, "
                "active_state_duration: float) -> ndarray"
            ),
        }
    )
    return CDGExport(
        nodes=[root] + list(sub_nodes),
        edges=list(edges),
        metadata={"goal": "Hodges EMG onset detection"},
    )


def _build_match_results(cdg: CDGExport) -> list[MatchResult]:
    """Phase C (partial): Build MatchResults for every atomic leaf."""
    return [_make_match_result(n) for n in cdg.leaf_nodes()]


def _assemble(cdg: CDGExport, match_results: list[MatchResult]):
    """Phase C: Assemble CDG + matches into a SkeletonFile."""
    assembler = Assembler(Prover.PYTHON)
    return assembler.assemble(cdg, match_results)


def _exec_skeleton(skeleton):
    """Phase D: exec() generated code and return the composition function.

    The assembler emits ``import _hodges_impl`` (because declaration names
    use dotted paths like ``_hodges_impl.estimate_rest_baseline``).  We
    temporarily register a fake module so the import succeeds, then remove
    it afterwards.
    """
    import sys
    import types

    # Create a temporary module containing our implementation
    mod = types.ModuleType("_hodges_impl")
    impl = _HodgesImpl()
    for attr in dir(impl):
        if not attr.startswith("_"):
            setattr(mod, attr, getattr(impl, attr))
    sys.modules["_hodges_impl"] = mod

    try:
        namespace: dict = {}
        exec(skeleton.source_code, namespace)  # noqa: S102
    finally:
        sys.modules.pop("_hodges_impl", None)

    # The composition function name comes from sanitize_name(root.name)
    fn_name = "hodges_time_domain_emg_onset_detection_composition"
    assert fn_name in namespace, (
        f"{fn_name} not found in generated code. "
        f"Available: {[k for k in namespace if not k.startswith('_')]}"
    )
    return namespace[fn_name]


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHodgesFullPipeline:
    """Full-pipeline E2E: prompt → decompose → match → assemble → execute."""

    # -- Phase A -----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_decompose_produces_6_atomic_nodes(self):
        """Phase A: decompose_node returns 6 ATOMIC sub-nodes."""
        sub_nodes, edges, _parent = await _run_phase_a()
        assert len(sub_nodes) == 6, f"Expected 6 sub-nodes, got {len(sub_nodes)}"
        atomic = [n for n in sub_nodes if n.status == NodeStatus.ATOMIC]
        assert len(atomic) == 6, (
            f"Expected all 6 nodes ATOMIC, got {len(atomic)} ATOMIC "
            f"({[n.name for n in sub_nodes if n.status != NodeStatus.ATOMIC]})"
        )

    # -- Phase B -----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_cdg_handoff_produces_6_pdg_nodes(self):
        """Phase B: to_pdg_nodes returns 6 PDGNodes from the CDG."""
        sub_nodes, edges, parent = await _run_phase_a()
        cdg = _build_cdg(sub_nodes, edges, parent)
        pdg_nodes = to_pdg_nodes(cdg, prover=Prover.PYTHON)
        assert len(pdg_nodes) == 6, f"Expected 6 PDGNodes, got {len(pdg_nodes)}"

    # -- Phase C -----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_assembler_produces_valid_python(self):
        """Phase C: SkeletonFile source is parseable Python."""
        sub_nodes, edges, parent = await _run_phase_a()
        cdg = _build_cdg(sub_nodes, edges, parent)
        match_results = _build_match_results(cdg)
        skeleton = _assemble(cdg, match_results)

        # Must parse without SyntaxError
        try:
            ast.parse(skeleton.source_code)
        except SyntaxError as exc:
            pytest.fail(
                f"Generated code has syntax error: {exc}\n\n"
                f"--- source ---\n{skeleton.source_code}"
            )

    @pytest.mark.asyncio
    async def test_composition_function_exists(self):
        """Phase C: generated code defines hodges_onset_detection_composition."""
        sub_nodes, edges, parent = await _run_phase_a()
        cdg = _build_cdg(sub_nodes, edges, parent)
        match_results = _build_match_results(cdg)
        skeleton = _assemble(cdg, match_results)

        assert "hodges_time_domain_emg_onset_detection_composition" in skeleton.source_code

    # -- Phase D -----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_generated_code_detects_onset(self):
        """Phase D: composition detects onset within ±50 samples of truth."""
        sub_nodes, edges, parent = await _run_phase_a()
        cdg = _build_cdg(sub_nodes, edges, parent)
        match_results = _build_match_results(cdg)
        skeleton = _assemble(cdg, match_results)
        composition_fn = _exec_skeleton(skeleton)

        signal, rest_signal, fs, true_onset = _make_synthetic_emg()
        result = composition_fn(signal, rest_signal, fs, 3.0, 0.05)

        # Result is (onsets, offsets) tuple from merge_adjacent
        if isinstance(result, tuple):
            onsets = result[0]
        else:
            onsets = np.asarray(result)

        assert len(onsets) >= 1, "Expected at least one detected onset"
        closest = min(abs(int(o) - true_onset) for o in onsets)
        assert closest <= 50, (
            f"Closest onset {closest} samples from truth ({true_onset}), "
            f"expected ≤50. Onsets: {onsets}"
        )

    @pytest.mark.asyncio
    async def test_matches_reference_implementation(self):
        """Phase D: pipeline onsets match reference implementation exactly."""
        sub_nodes, edges, parent = await _run_phase_a()
        cdg = _build_cdg(sub_nodes, edges, parent)
        match_results = _build_match_results(cdg)
        skeleton = _assemble(cdg, match_results)
        composition_fn = _exec_skeleton(skeleton)

        signal, rest_signal, fs, _true_onset = _make_synthetic_emg()
        result = composition_fn(signal, rest_signal, fs, 3.0, 0.05)
        if isinstance(result, tuple):
            pipeline_onsets = np.asarray(result[0])
        else:
            pipeline_onsets = np.asarray(result)

        ref_onsets = _reference_hodges(signal, rest_signal, fs, threshold=3.0)

        np.testing.assert_array_equal(
            pipeline_onsets,
            ref_onsets,
            err_msg=(
                f"Pipeline onsets {pipeline_onsets} != "
                f"reference onsets {ref_onsets}"
            ),
        )

    @pytest.mark.asyncio
    async def test_no_onset_on_pure_noise(self):
        """Robustness: pure low-amplitude noise → 0 onsets."""
        sub_nodes, edges, parent = await _run_phase_a()
        cdg = _build_cdg(sub_nodes, edges, parent)
        match_results = _build_match_results(cdg)
        skeleton = _assemble(cdg, match_results)
        composition_fn = _exec_skeleton(skeleton)

        rng = np.random.default_rng(99)
        noise = 0.01 * rng.standard_normal(2000)
        rest = noise[:1000]

        result = composition_fn(noise, rest, 1000.0, 3.0, 0.05)
        if isinstance(result, tuple):
            onsets = result[0]
        else:
            onsets = np.asarray(result)

        assert len(onsets) == 0, (
            f"Expected 0 onsets on pure noise, got {len(onsets)}: {onsets}"
        )

    # -- Combined ----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_full_pipeline_end_to_end(self):
        """Combined: all phases chained, validates correct onset detection."""
        # Phase A
        sub_nodes, edges, parent = await _run_phase_a()
        assert len(sub_nodes) == 6

        # Phase B
        cdg = _build_cdg(sub_nodes, edges, parent)
        pdg_nodes = to_pdg_nodes(cdg, prover=Prover.PYTHON)
        assert len(pdg_nodes) == 6

        # Phase C
        match_results = _build_match_results(cdg)
        skeleton = _assemble(cdg, match_results)
        ast.parse(skeleton.source_code)  # must not raise
        assert "hodges_time_domain_emg_onset_detection_composition" in skeleton.source_code

        # Phase D
        composition_fn = _exec_skeleton(skeleton)
        signal, rest_signal, fs, true_onset = _make_synthetic_emg()
        result = composition_fn(signal, rest_signal, fs, 3.0, 0.05)
        if isinstance(result, tuple):
            onsets = result[0]
        else:
            onsets = np.asarray(result)

        assert len(onsets) >= 1, "No onsets detected"
        closest = min(abs(int(o) - true_onset) for o in onsets)
        assert closest <= 50, f"Onset off by {closest} samples (max 50)"

        # Cross-validate with reference
        ref_onsets = _reference_hodges(signal, rest_signal, fs, threshold=3.0)
        np.testing.assert_array_equal(onsets, ref_onsets)
