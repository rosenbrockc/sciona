"""Tests for the ghost witness simulation pass in the synthesizer pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ageom.architect.handoff import CDGExport
from ageom.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from ageom.synthesizer.ghost_sim import (
    GhostSimReport,
    _extract_atom_name,
    _build_abstract_value,
    _GHOST_AVAILABLE,
    run_ghost_simulation,
)

_requires_ageoa = pytest.mark.skipif(
    not _GHOST_AVAILABLE, reason="ageoa package not installed"
)
from ageom.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
    PDGNode,
    Prover,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_match(node_id: str, decl_name: str, type_sig: str = "") -> MatchResult:
    decl = Declaration(name=decl_name, type_signature=type_sig, prover=Prover.PYTHON)
    candidate = CandidateMatch(declaration=decl, score=0.9, retrieval_method="test")
    vr = VerificationResult(candidate=candidate, verified=True)
    return MatchResult(
        pdg_node=PDGNode(predicate_id=node_id, statement=type_sig),
        verified_match=vr,
        all_candidates=[candidate],
        all_verifications=[vr],
    )


# ---------------------------------------------------------------------------
# TestExtractAtomName
# ---------------------------------------------------------------------------


class TestExtractAtomName:
    def test_simple_name(self):
        assert _extract_atom_name("fft") == "fft"

    def test_qualified_name(self):
        assert _extract_atom_name("ageoa.numpy.fft.fft") == "fft"

    def test_two_part_name(self):
        assert _extract_atom_name("signal.butter") == "butter"

    def test_dotted_module(self):
        assert _extract_atom_name("scipy.sparse.csgraph.graph_laplacian") == "graph_laplacian"


# ---------------------------------------------------------------------------
# TestBuildAbstractValue
# ---------------------------------------------------------------------------


@_requires_ageoa
class TestBuildAbstractValue:
    def test_signal_default(self):
        val = _build_abstract_value("np.ndarray", "time domain")
        assert val.domain == "time"
        assert val.dtype == "float64"

    def test_freq_domain_from_constraints(self):
        val = _build_abstract_value("np.ndarray", "freq domain")
        assert val.domain == "freq"

    def test_complex_dtype(self):
        val = _build_abstract_value("complex array", "")
        assert val.dtype == "complex128"

    def test_graph_type(self):
        val = _build_abstract_value("Graph Laplacian", "symmetric")
        assert hasattr(val, "n_nodes")
        assert val.is_symmetric is True

    def test_filter_coefficients(self):
        val = _build_abstract_value("Filter coefficients", "")
        assert hasattr(val, "order")
        assert val.is_stable is True


# ---------------------------------------------------------------------------
# TestGhostSimReport
# ---------------------------------------------------------------------------


class TestGhostSimReport:
    def test_default_report(self):
        report = GhostSimReport()
        assert report.ran is False
        assert report.passed is False
        assert report.node_count == 0
        assert report.error == ""


# ---------------------------------------------------------------------------
# TestRunGhostSimulation — FFT pipeline
# ---------------------------------------------------------------------------


@_requires_ageoa
class TestRunGhostSimulationFFT:
    """Test ghost simulation with a simple FFT -> IFFT pipeline."""

    @pytest.fixture
    def fft_cdg(self) -> CDGExport:
        """CDG: root -> fft_node -> ifft_node (frequency domain round-trip)."""
        nodes = [
            AlgorithmicNode(
                node_id="root",
                name="FFT Round Trip",
                description="Forward FFT then inverse FFT",
                concept_type=ConceptType.SIGNAL_TRANSFORM,
                status=NodeStatus.DECOMPOSED,
                children=["fft_node", "ifft_node"],
                depth=0,
            ),
            AlgorithmicNode(
                node_id="fft_node",
                name="Forward FFT",
                description="Compute FFT of time-domain signal",
                concept_type=ConceptType.SIGNAL_TRANSFORM,
                status=NodeStatus.ATOMIC,
                matched_primitive="fft",
                inputs=[IOSpec(name="sig", type_desc="np.ndarray", constraints="time domain")],
                outputs=[IOSpec(name="spectrum", type_desc="np.ndarray", constraints="freq domain")],
                depth=1,
            ),
            AlgorithmicNode(
                node_id="ifft_node",
                name="Inverse FFT",
                description="Compute IFFT to reconstruct signal",
                concept_type=ConceptType.SIGNAL_TRANSFORM,
                status=NodeStatus.ATOMIC,
                matched_primitive="ifft",
                inputs=[IOSpec(name="sig", type_desc="np.ndarray", constraints="freq domain")],
                outputs=[IOSpec(name="reconstructed", type_desc="np.ndarray", constraints="time domain")],
                depth=1,
            ),
        ]
        edges = [
            DependencyEdge(
                source_id="fft_node",
                target_id="ifft_node",
                output_name="spectrum",
                input_name="sig",
                source_type="np.ndarray",
                target_type="np.ndarray",
            ),
        ]
        return CDGExport(nodes=nodes, edges=edges, metadata={"goal": "FFT round trip"})

    @pytest.fixture
    def fft_matches(self) -> list[MatchResult]:
        return [
            _make_match("fft_node", "ageoa.numpy.fft.fft"),
            _make_match("ifft_node", "ageoa.numpy.fft.ifft"),
        ]

    def test_fft_roundtrip_passes(self, fft_cdg, fft_matches):
        report = run_ghost_simulation(fft_cdg, fft_matches)
        assert report.ran is True
        assert report.passed is True
        assert report.node_count == 2
        assert len(report.trace) == 2
        assert report.error == ""

    def test_fft_roundtrip_trace_order(self, fft_cdg, fft_matches):
        report = run_ghost_simulation(fft_cdg, fft_matches)
        assert report.trace == ["Forward FFT", "Inverse FFT"]


@_requires_ageoa
class TestRunGhostSimulationDomainMismatch:
    """Test that domain mismatches are caught."""

    @pytest.fixture
    def bad_cdg(self) -> CDGExport:
        """CDG where FFT output (freq) is fed into another FFT (expects time)."""
        nodes = [
            AlgorithmicNode(
                node_id="root",
                name="Double FFT (invalid)",
                description="Two FFTs chained — invalid!",
                concept_type=ConceptType.SIGNAL_TRANSFORM,
                status=NodeStatus.DECOMPOSED,
                children=["fft1", "fft2"],
                depth=0,
            ),
            AlgorithmicNode(
                node_id="fft1",
                name="First FFT",
                description="First forward FFT",
                concept_type=ConceptType.SIGNAL_TRANSFORM,
                status=NodeStatus.ATOMIC,
                matched_primitive="fft",
                inputs=[IOSpec(name="sig", type_desc="np.ndarray", constraints="time domain")],
                outputs=[IOSpec(name="spectrum", type_desc="np.ndarray", constraints="freq domain")],
                depth=1,
            ),
            AlgorithmicNode(
                node_id="fft2",
                name="Second FFT",
                description="Second forward FFT (should fail — freq input)",
                concept_type=ConceptType.SIGNAL_TRANSFORM,
                status=NodeStatus.ATOMIC,
                matched_primitive="fft",
                inputs=[IOSpec(name="sig", type_desc="np.ndarray", constraints="freq domain")],
                outputs=[IOSpec(name="spectrum2", type_desc="np.ndarray", constraints="freq domain")],
                depth=1,
            ),
        ]
        edges = [
            DependencyEdge(
                source_id="fft1",
                target_id="fft2",
                output_name="spectrum",
                input_name="sig",
                source_type="np.ndarray",
                target_type="np.ndarray",
            ),
        ]
        return CDGExport(nodes=nodes, edges=edges, metadata={"goal": "bad"})

    @pytest.fixture
    def bad_matches(self) -> list[MatchResult]:
        return [
            _make_match("fft1", "fft"),
            _make_match("fft2", "fft"),
        ]

    def test_domain_mismatch_caught(self, bad_cdg, bad_matches):
        report = run_ghost_simulation(bad_cdg, bad_matches)
        assert report.ran is True
        assert report.passed is False
        assert "time" in report.error.lower() or "domain" in report.error.lower()
        assert report.error_node == "Second FFT"
        assert report.error_function == "fft"


@_requires_ageoa
class TestRunGhostSimulationNoWitness:
    """Test behaviour when atoms have no registered witness."""

    @pytest.fixture
    def non_dsp_cdg(self) -> CDGExport:
        """CDG with non-DSP atoms (no witnesses)."""
        nodes = [
            AlgorithmicNode(
                node_id="custom_node",
                name="Custom Op",
                description="A custom operation with no witness",
                concept_type=ConceptType.CUSTOM,
                status=NodeStatus.ATOMIC,
                inputs=[IOSpec(name="data", type_desc="list[int]")],
                outputs=[IOSpec(name="result", type_desc="list[int]")],
                depth=0,
            ),
        ]
        return CDGExport(nodes=nodes, edges=[], metadata={})

    @pytest.fixture
    def non_dsp_matches(self) -> list[MatchResult]:
        return [_make_match("custom_node", "my_custom_unregistered_op")]

    def test_no_simulable_nodes(self, non_dsp_cdg, non_dsp_matches):
        report = run_ghost_simulation(non_dsp_cdg, non_dsp_matches)
        # Should not crash; just reports no simulation ran
        assert report.ran is False
        assert report.skipped_nodes == ["Custom Op"]


@_requires_ageoa
class TestRunGhostSimulationFilter:
    """Test ghost simulation with a filter design -> apply pipeline."""

    @pytest.fixture
    def filter_cdg(self) -> CDGExport:
        """CDG: root -> butter_node -> lfilter_node."""
        nodes = [
            AlgorithmicNode(
                node_id="root",
                name="Lowpass Filter Pipeline",
                description="Design and apply a Butterworth filter",
                concept_type=ConceptType.SIGNAL_FILTER,
                status=NodeStatus.DECOMPOSED,
                children=["butter_node", "lfilter_node"],
                depth=0,
            ),
            AlgorithmicNode(
                node_id="butter_node",
                name="Butterworth Design",
                description="Design a Butterworth lowpass filter",
                concept_type=ConceptType.SIGNAL_FILTER,
                status=NodeStatus.ATOMIC,
                matched_primitive="butter",
                inputs=[
                    IOSpec(name="order", type_desc="int", constraints=""),
                    IOSpec(name="wn", type_desc="float", constraints=""),
                    IOSpec(name="fs", type_desc="float", constraints=""),
                ],
                outputs=[IOSpec(name="coefficients", type_desc="Filter coefficients")],
                depth=1,
            ),
            AlgorithmicNode(
                node_id="lfilter_node",
                name="Apply Filter",
                description="Apply the designed filter to a signal",
                concept_type=ConceptType.SIGNAL_FILTER,
                status=NodeStatus.ATOMIC,
                matched_primitive="lfilter",
                inputs=[
                    IOSpec(name="coefficients", type_desc="Filter coefficients"),
                    IOSpec(name="sig", type_desc="np.ndarray", constraints="time domain"),
                ],
                outputs=[IOSpec(name="filtered", type_desc="np.ndarray", constraints="time domain")],
                depth=1,
            ),
        ]
        edges = [
            DependencyEdge(
                source_id="butter_node",
                target_id="lfilter_node",
                output_name="coefficients",
                input_name="coefficients",
                source_type="Filter coefficients",
                target_type="Filter coefficients",
            ),
        ]
        return CDGExport(nodes=nodes, edges=edges, metadata={"goal": "Filter pipeline"})

    @pytest.fixture
    def filter_matches(self) -> list[MatchResult]:
        return [
            _make_match("butter_node", "scipy.signal.butter"),
            _make_match("lfilter_node", "scipy.signal.lfilter"),
        ]

    def test_filter_pipeline_passes(self, filter_cdg, filter_matches):
        report = run_ghost_simulation(filter_cdg, filter_matches)
        assert report.ran is True
        assert report.passed is True
        assert report.node_count == 2


@_requires_ageoa
class TestRunGhostSimulationMixed:
    """Test CDG with a mix of DSP and non-DSP nodes."""

    @pytest.fixture
    def mixed_cdg(self) -> CDGExport:
        """CDG with one DSP node and one non-DSP node (unregistered), independent."""
        nodes = [
            AlgorithmicNode(
                node_id="fft_node",
                name="FFT",
                description="Forward FFT",
                concept_type=ConceptType.SIGNAL_TRANSFORM,
                status=NodeStatus.ATOMIC,
                matched_primitive="fft",
                inputs=[IOSpec(name="sig", type_desc="np.ndarray", constraints="time domain")],
                outputs=[IOSpec(name="spectrum", type_desc="np.ndarray")],
                depth=0,
            ),
            AlgorithmicNode(
                node_id="custom_node",
                name="Custom Op",
                description="A custom unregistered operation",
                concept_type=ConceptType.CUSTOM,
                status=NodeStatus.ATOMIC,
                inputs=[IOSpec(name="data", type_desc="list[int]")],
                outputs=[IOSpec(name="result", type_desc="list[int]")],
                depth=0,
            ),
        ]
        return CDGExport(nodes=nodes, edges=[], metadata={})

    @pytest.fixture
    def mixed_matches(self) -> list[MatchResult]:
        return [
            _make_match("fft_node", "fft"),
            _make_match("custom_node", "my_custom_unregistered_op"),
        ]

    def test_mixed_simulates_dsp_only(self, mixed_cdg, mixed_matches):
        report = run_ghost_simulation(mixed_cdg, mixed_matches)
        assert report.ran is True
        assert report.passed is True
        assert report.node_count == 1
        assert "Custom Op" in report.skipped_nodes
        assert "FFT" in report.trace
