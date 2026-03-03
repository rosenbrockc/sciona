"""Integration test: ingest BioSPPy ECG pipeline into ageo-atoms.

Creates a stateful wrapper class around biosppy.signals.ecg.ecg(),
runs the Smart Ingester with deterministic (mocked) LLM responses
that encode the *actual* BioSPPy data flow, validates the resulting
CDGExport and generated atoms, and writes both artefacts to
ageo-atoms/ageoa/biosppy/.

Pipeline under test (from biosppy.signals.ecg.ecg):

    signal ─► Bandpass Filter ─► R-Peak Detection ─► Peak Correction
                  │                                       │
                  ├───────────► Template Extraction ◄─────┤
                  │                                       │
                  └──────────────────────────────────────► Heart Rate
"""

from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ageom.architect.handoff import save_json
from ageom.architect.models import ConceptType, NodeStatus
from ageom.ingester.extractor import extract_data_flow
from ageom.ingester.graph import IngesterAgent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

AGEO_ATOMS_DIR = Path(__file__).resolve().parent.parent.parent / "ageo-atoms"
BIOSPPY_OUT = AGEO_ATOMS_DIR / "ageoa" / "biosppy"


# ---------------------------------------------------------------------------
# ECG pipeline wrapper class (the source we ingest)
#
# Wraps the sequential steps of biosppy.signals.ecg.ecg() into a
# stateful class whose self.* access patterns mirror the real data
# flow.  Placeholder functions avoid importing biosppy at parse time.
# ---------------------------------------------------------------------------

ECG_PROCESSOR_SOURCE = textwrap.dedent('''\
    """BioSPPy ECG pipeline wrapped as a stateful class for ingestion.

    Mirrors the data flow of biosppy.signals.ecg.ecg():
      signal -> bandpass filter -> R-peak detection -> peak correction
                   |                                      |
                   +------> template extraction <---------+
                   |                                      |
                   +------> heart rate computation <------+
    """

    import numpy as np


    def _apply_fir_bandpass(signal, order, freq, fs):
        """Placeholder for biosppy.tools.filter_signal FIR bandpass."""
        raise NotImplementedError


    def _hamilton_segmenter(signal, fs):
        """Placeholder for biosppy.signals.ecg.hamilton_segmenter."""
        raise NotImplementedError


    def _correct_rpeaks(signal, rpeaks, fs, tol=0.05):
        """Placeholder for biosppy.signals.ecg.correct_rpeaks."""
        raise NotImplementedError


    def _extract_heartbeats(signal, rpeaks, fs, before=0.2, after=0.4):
        """Placeholder for biosppy.signals.ecg.extract_heartbeats."""
        raise NotImplementedError


    def _get_heart_rate(beats, fs, smooth=True, size=3):
        """Placeholder for biosppy.tools.get_heart_rate."""
        raise NotImplementedError


    class ECGProcessor:
        """Stateful wrapper around the BioSPPy ECG processing pipeline.

        Wraps the sequential steps of biosppy.signals.ecg.ecg() into
        discrete methods whose self.* reads/writes mirror the real data
        flow, enabling the Smart Ingester AST extractor to trace them.
        """

        def __init__(self, signal, sampling_rate=1000.0):
            self.signal = np.array(signal)
            self.sampling_rate = float(sampling_rate)
            self.filtered = None
            self.rpeaks = None
            self.templates = None
            self.heart_rate = None
            self.hr_idx = None

        def filter_signal(self):
            """Apply FIR bandpass filter (3-45 Hz)."""
            order = int(0.3 * self.sampling_rate)
            self.filtered = _apply_fir_bandpass(
                self.signal, order, [3, 45], self.sampling_rate
            )

        def detect_rpeaks(self):
            """Detect R-peak locations using Hamilton segmenter."""
            self.rpeaks = _hamilton_segmenter(self.filtered, self.sampling_rate)

        def correct_peaks(self):
            """Correct R-peak positions to local maxima within tolerance."""
            self.rpeaks = _correct_rpeaks(
                self.filtered, self.rpeaks, self.sampling_rate
            )

        def extract_templates(self):
            """Extract heartbeat waveform templates around each R-peak."""
            self.templates, self.rpeaks = _extract_heartbeats(
                self.filtered, self.rpeaks, self.sampling_rate
            )

        def compute_heart_rate(self):
            """Compute instantaneous heart rate from R-R intervals."""
            self.hr_idx, self.heart_rate = _get_heart_rate(
                self.rpeaks, self.sampling_rate
            )
''')


# ---------------------------------------------------------------------------
# Known-correct LLM responses (encode the actual BioSPPy data flow)
#
# These are the responses a correct LLM would produce.  By injecting
# them deterministically we guarantee NO hallucinated data flows.
# ---------------------------------------------------------------------------

_CHUNK_RESPONSE = json.dumps(
    {
        "macro_atoms": [
            {
                "name": "Bandpass Filter",
                "description": (
                    "Apply FIR bandpass filter (3-45 Hz) to remove baseline "
                    "wander and high-frequency noise from the raw ECG signal"
                ),
                "method_names": ["filter_signal"],
                "inputs": [
                    {
                        "name": "signal",
                        "type_desc": "np.ndarray",
                        "constraints": "1D raw ECG signal",
                    },
                ],
                "outputs": [
                    {
                        "name": "filtered",
                        "type_desc": "np.ndarray",
                        "constraints": "bandpass-filtered ECG",
                    },
                ],
                "config_params": ["sampling_rate"],
                "concept_type": "signal_filter",
                "is_optional": False,
            },
            {
                "name": "R-Peak Detection",
                "description": (
                    "Detect R-peak locations in the filtered ECG signal "
                    "using the Hamilton segmenter algorithm"
                ),
                "method_names": ["detect_rpeaks"],
                "inputs": [
                    {
                        "name": "filtered",
                        "type_desc": "np.ndarray",
                        "constraints": "filtered ECG signal",
                    },
                ],
                "outputs": [
                    {
                        "name": "rpeaks",
                        "type_desc": "np.ndarray",
                        "constraints": "R-peak sample indices",
                    },
                ],
                "config_params": ["sampling_rate"],
                "concept_type": "custom",
                "is_optional": False,
            },
            {
                "name": "Peak Correction",
                "description": (
                    "Correct R-peak locations to the nearest local maximum "
                    "within a tolerance window"
                ),
                "method_names": ["correct_peaks"],
                "inputs": [
                    {
                        "name": "filtered",
                        "type_desc": "np.ndarray",
                        "constraints": "filtered ECG signal",
                    },
                    {
                        "name": "rpeaks",
                        "type_desc": "np.ndarray",
                        "constraints": "initial R-peak indices",
                    },
                ],
                "outputs": [
                    {
                        "name": "rpeaks_corrected",
                        "type_desc": "np.ndarray",
                        "constraints": "corrected R-peak indices",
                    },
                ],
                "config_params": ["sampling_rate"],
                "concept_type": "custom",
                "is_optional": False,
            },
            {
                "name": "Template Extraction",
                "description": (
                    "Extract individual heartbeat waveform templates around "
                    "each R-peak with configurable before/after windows"
                ),
                "method_names": ["extract_templates"],
                "inputs": [
                    {
                        "name": "filtered",
                        "type_desc": "np.ndarray",
                        "constraints": "filtered ECG signal",
                    },
                    {
                        "name": "rpeaks",
                        "type_desc": "np.ndarray",
                        "constraints": "corrected R-peak indices",
                    },
                ],
                "outputs": [
                    {
                        "name": "templates",
                        "type_desc": "np.ndarray",
                        "constraints": "2D array of heartbeat templates",
                    },
                    {
                        "name": "rpeaks_final",
                        "type_desc": "np.ndarray",
                        "constraints": "final R-peak indices after template extraction",
                    },
                ],
                "config_params": ["sampling_rate"],
                "concept_type": "custom",
                "is_optional": False,
            },
            {
                "name": "Heart Rate Computation",
                "description": (
                    "Compute instantaneous heart rate in bpm from R-R "
                    "intervals with optional smoothing"
                ),
                "method_names": ["compute_heart_rate"],
                "inputs": [
                    {
                        "name": "rpeaks",
                        "type_desc": "np.ndarray",
                        "constraints": "R-peak sample indices",
                    },
                ],
                "outputs": [
                    {
                        "name": "hr_idx",
                        "type_desc": "np.ndarray",
                        "constraints": "time indices for heart rate values",
                    },
                    {
                        "name": "heart_rate",
                        "type_desc": "np.ndarray",
                        "constraints": "instantaneous heart rate in bpm",
                    },
                ],
                "config_params": ["sampling_rate"],
                "concept_type": "custom",
                "is_optional": False,
            },
        ],
        "edges": [
            {
                "source_id": "bandpass_filter",
                "target_id": "r_peak_detection",
                "output_name": "filtered",
                "input_name": "filtered",
                "source_type": "np.ndarray",
                "target_type": "np.ndarray",
            },
            {
                "source_id": "bandpass_filter",
                "target_id": "peak_correction",
                "output_name": "filtered",
                "input_name": "filtered",
                "source_type": "np.ndarray",
                "target_type": "np.ndarray",
            },
            {
                "source_id": "r_peak_detection",
                "target_id": "peak_correction",
                "output_name": "rpeaks",
                "input_name": "rpeaks",
                "source_type": "np.ndarray",
                "target_type": "np.ndarray",
            },
            {
                "source_id": "bandpass_filter",
                "target_id": "template_extraction",
                "output_name": "filtered",
                "input_name": "filtered",
                "source_type": "np.ndarray",
                "target_type": "np.ndarray",
            },
            {
                "source_id": "peak_correction",
                "target_id": "template_extraction",
                "output_name": "rpeaks_corrected",
                "input_name": "rpeaks",
                "source_type": "np.ndarray",
                "target_type": "np.ndarray",
            },
            {
                "source_id": "peak_correction",
                "target_id": "heart_rate_computation",
                "output_name": "rpeaks_corrected",
                "input_name": "rpeaks",
                "source_type": "np.ndarray",
                "target_type": "np.ndarray",
            },
        ],
    }
)

# hoist_state: extract cross-window state model for intermediate pipeline data
_HOIST_RESPONSE = json.dumps(
    {
        "state_models": [
            {
                "model_name": "ECGPipelineState",
                "fields": [
                    ["filtered", "np.ndarray"],
                    ["rpeaks", "np.ndarray"],
                ],
                "source_attrs": ["filtered", "rpeaks"],
                "docstring": (
                    "Intermediate pipeline state carrying the filtered ECG "
                    "signal and detected R-peak indices between stages"
                ),
            },
        ],
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent() -> tuple[IngesterAgent, AsyncMock]:
    """Build an IngesterAgent with a mocked LLM returning known-correct data."""
    mock_llm = AsyncMock()
    _responses = iter([_CHUNK_RESPONSE, _HOIST_RESPONSE])

    async def _side_effect(*args, **kwargs):
        return next(_responses, "[]")

    mock_llm.complete.side_effect = _side_effect
    agent = IngesterAgent(llm=mock_llm)
    return agent, mock_llm


# ---------------------------------------------------------------------------
# Tests: Phase 1 — deterministic AST extraction
# ---------------------------------------------------------------------------


class TestPhase1Extraction:
    """Verify the deterministic AST extractor captures ECG data flow."""

    @pytest.fixture
    def ecg_source(self, tmp_path) -> str:
        src = tmp_path / "ecg_processor.py"
        src.write_text(ECG_PROCESSOR_SOURCE)
        return str(src)

    @pytest.mark.asyncio
    async def test_extracts_all_attributes(self, ecg_source):
        dfg = await extract_data_flow(ecg_source, "ECGProcessor")
        assert set(dfg.all_attributes.keys()) == {
            "signal",
            "sampling_rate",
            "filtered",
            "rpeaks",
            "templates",
            "heart_rate",
            "hr_idx",
        }

    @pytest.mark.asyncio
    async def test_extracts_all_methods(self, ecg_source):
        dfg = await extract_data_flow(ecg_source, "ECGProcessor")
        method_names = {m.name for m in dfg.methods}
        assert method_names == {
            "__init__",
            "filter_signal",
            "detect_rpeaks",
            "correct_peaks",
            "extract_templates",
            "compute_heart_rate",
        }

    @pytest.mark.asyncio
    async def test_filter_reads_signal_writes_filtered(self, ecg_source):
        dfg = await extract_data_flow(ecg_source, "ECGProcessor")
        fs = next(m for m in dfg.methods if m.name == "filter_signal")
        assert set(fs.reads) == {"signal", "sampling_rate"}
        assert set(fs.writes) == {"filtered"}

    @pytest.mark.asyncio
    async def test_detect_reads_filtered_writes_rpeaks(self, ecg_source):
        dfg = await extract_data_flow(ecg_source, "ECGProcessor")
        dr = next(m for m in dfg.methods if m.name == "detect_rpeaks")
        assert set(dr.reads) == {"filtered", "sampling_rate"}
        assert set(dr.writes) == {"rpeaks"}

    @pytest.mark.asyncio
    async def test_correct_reads_filtered_and_rpeaks(self, ecg_source):
        dfg = await extract_data_flow(ecg_source, "ECGProcessor")
        cp = next(m for m in dfg.methods if m.name == "correct_peaks")
        assert set(cp.reads) == {"filtered", "rpeaks", "sampling_rate"}
        assert set(cp.writes) == {"rpeaks"}

    @pytest.mark.asyncio
    async def test_extract_templates_tuple_unpack(self, ecg_source):
        """Tuple unpacking: self.templates, self.rpeaks = ...."""
        dfg = await extract_data_flow(ecg_source, "ECGProcessor")
        et = next(m for m in dfg.methods if m.name == "extract_templates")
        assert set(et.reads) == {"filtered", "rpeaks", "sampling_rate"}
        assert set(et.writes) == {"templates", "rpeaks"}

    @pytest.mark.asyncio
    async def test_compute_hr_tuple_unpack(self, ecg_source):
        """Tuple unpacking: self.hr_idx, self.heart_rate = ...."""
        dfg = await extract_data_flow(ecg_source, "ECGProcessor")
        ch = next(m for m in dfg.methods if m.name == "compute_heart_rate")
        assert set(ch.reads) == {"rpeaks", "sampling_rate"}
        assert set(ch.writes) == {"hr_idx", "heart_rate"}

    @pytest.mark.asyncio
    async def test_cross_window_state(self, ecg_source):
        """Cross-window attrs: filtered and rpeaks (read AND written in non-init)."""
        dfg = await extract_data_flow(ecg_source, "ECGProcessor")
        assert set(dfg.cross_window_attrs) == {"filtered", "rpeaks"}

    @pytest.mark.asyncio
    async def test_init_chain(self, ecg_source):
        dfg = await extract_data_flow(ecg_source, "ECGProcessor")
        # init_chain is sorted alphabetically (from _extract_method_fact)
        assert set(dfg.init_chain) == {
            "signal",
            "sampling_rate",
            "filtered",
            "rpeaks",
            "templates",
            "heart_rate",
            "hr_idx",
        }


# ---------------------------------------------------------------------------
# Tests: Full pipeline — CDG structure
# ---------------------------------------------------------------------------


class TestFullPipelineCDG:
    """Validate CDGExport from the full ingester pipeline."""

    @pytest.fixture
    def ecg_source(self, tmp_path) -> str:
        src = tmp_path / "ecg_processor.py"
        src.write_text(ECG_PROCESSOR_SOURCE)
        return str(src)

    @pytest.mark.asyncio
    async def test_cdg_has_correct_node_count(self, ecg_source):
        agent, _ = _make_agent()
        bundle = await agent.ingest(ecg_source, "ECGProcessor")

        # 1 root (DECOMPOSED) + 5 atomic leaves
        assert len(bundle.cdg.nodes) == 6

    @pytest.mark.asyncio
    async def test_root_is_decomposed(self, ecg_source):
        agent, _ = _make_agent()
        bundle = await agent.ingest(ecg_source, "ECGProcessor")

        root = next(n for n in bundle.cdg.nodes if n.depth == 0)
        assert root.status == NodeStatus.DECOMPOSED
        assert root.name == "ECGProcessor"
        assert len(root.children) == 5

    @pytest.mark.asyncio
    async def test_all_leaves_are_atomic(self, ecg_source):
        agent, _ = _make_agent()
        bundle = await agent.ingest(ecg_source, "ECGProcessor")

        leaves = [n for n in bundle.cdg.nodes if n.depth == 1]
        assert len(leaves) == 5
        for leaf in leaves:
            assert leaf.status == NodeStatus.ATOMIC

    @pytest.mark.asyncio
    async def test_bandpass_filter_is_signal_filter(self, ecg_source):
        agent, _ = _make_agent()
        bundle = await agent.ingest(ecg_source, "ECGProcessor")

        bp = next(n for n in bundle.cdg.nodes if n.node_id == "bandpass_filter")
        assert bp.concept_type == ConceptType.SIGNAL_FILTER

    @pytest.mark.asyncio
    async def test_all_leaves_have_type_signatures(self, ecg_source):
        agent, _ = _make_agent()
        bundle = await agent.ingest(ecg_source, "ECGProcessor")

        for node in bundle.cdg.nodes:
            if node.status == NodeStatus.ATOMIC:
                assert node.type_signature, f"{node.name} missing type_signature"
                assert node.inputs, f"{node.name} missing inputs"
                assert node.outputs, f"{node.name} missing outputs"


# ---------------------------------------------------------------------------
# Tests: No hallucinated data flows
# ---------------------------------------------------------------------------


class TestDataFlowEdges:
    """Ensure CDG edges exactly match the real BioSPPy pipeline."""

    @pytest.fixture
    def ecg_source(self, tmp_path) -> str:
        src = tmp_path / "ecg_processor.py"
        src.write_text(ECG_PROCESSOR_SOURCE)
        return str(src)

    @pytest.mark.asyncio
    async def test_edge_count(self, ecg_source):
        agent, _ = _make_agent()
        bundle = await agent.ingest(ecg_source, "ECGProcessor")
        # 6 data-flow edges + 10 state edges (filtered, rpeaks cross-window)
        data_edges = [
            e for e in bundle.cdg.edges if e.source_type != "ECGPipelineState"
        ]
        state_edges = [
            e for e in bundle.cdg.edges if e.source_type == "ECGPipelineState"
        ]
        assert len(data_edges) == 6
        assert len(state_edges) > 0

    @pytest.mark.asyncio
    async def test_no_phantom_edge_sources(self, ecg_source):
        agent, _ = _make_agent()
        bundle = await agent.ingest(ecg_source, "ECGProcessor")

        node_ids = {n.node_id for n in bundle.cdg.nodes}
        for edge in bundle.cdg.edges:
            assert edge.source_id in node_ids, f"phantom source: {edge.source_id}"

    @pytest.mark.asyncio
    async def test_no_phantom_edge_targets(self, ecg_source):
        agent, _ = _make_agent()
        bundle = await agent.ingest(ecg_source, "ECGProcessor")

        node_ids = {n.node_id for n in bundle.cdg.nodes}
        for edge in bundle.cdg.edges:
            assert edge.target_id in node_ids, f"phantom target: {edge.target_id}"

    @pytest.mark.asyncio
    async def test_edges_match_actual_data_flow(self, ecg_source):
        """Data-flow edges must reflect the real BioSPPy pipeline."""
        agent, _ = _make_agent()
        bundle = await agent.ingest(ecg_source, "ECGProcessor")

        # Filter to data-flow edges only (exclude state-typed edges)
        data_edge_set = {
            (e.source_id, e.target_id, e.output_name, e.input_name)
            for e in bundle.cdg.edges
            if e.source_type != "ECGPipelineState"
        }

        # These are the REAL data flows in biosppy.signals.ecg.ecg():
        #   filtered -> hamilton_segmenter, correct_rpeaks, extract_heartbeats
        #   rpeaks   -> correct_rpeaks, extract_heartbeats, get_heart_rate
        expected = {
            # filter -> detect (filtered signal)
            ("bandpass_filter", "r_peak_detection", "filtered", "filtered"),
            # filter -> correct (filtered signal for local-max search)
            ("bandpass_filter", "peak_correction", "filtered", "filtered"),
            # detect -> correct (initial rpeaks to refine)
            ("r_peak_detection", "peak_correction", "rpeaks", "rpeaks"),
            # filter -> extract (filtered signal for templates)
            ("bandpass_filter", "template_extraction", "filtered", "filtered"),
            # correct -> extract (corrected rpeaks for template windows)
            ("peak_correction", "template_extraction", "rpeaks_corrected", "rpeaks"),
            # correct -> heart_rate (corrected rpeaks for R-R intervals)
            ("peak_correction", "heart_rate_computation", "rpeaks_corrected", "rpeaks"),
        }

        assert data_edge_set == expected

    @pytest.mark.asyncio
    async def test_all_data_edges_are_ndarray(self, ecg_source):
        """All data-flow edge types in the ECG pipeline are np.ndarray."""
        agent, _ = _make_agent()
        bundle = await agent.ingest(ecg_source, "ECGProcessor")

        data_edges = [
            e for e in bundle.cdg.edges if e.source_type != "ECGPipelineState"
        ]
        for edge in data_edges:
            assert edge.source_type == "np.ndarray"
            assert edge.target_type == "np.ndarray"

    @pytest.mark.asyncio
    async def test_state_edges_are_ecg_pipeline_state(self, ecg_source):
        """State edges should be typed with ECGPipelineState."""
        agent, _ = _make_agent()
        bundle = await agent.ingest(ecg_source, "ECGProcessor")

        state_edges = [
            e for e in bundle.cdg.edges if e.source_type == "ECGPipelineState"
        ]
        assert len(state_edges) > 0
        for edge in state_edges:
            assert edge.target_type == "ECGPipelineState"


# ---------------------------------------------------------------------------
# Tests: Generated code quality
# ---------------------------------------------------------------------------


class TestGeneratedCode:
    """Validate generated atoms, witnesses, and state models."""

    @pytest.fixture
    def ecg_source(self, tmp_path) -> str:
        src = tmp_path / "ecg_processor.py"
        src.write_text(ECG_PROCESSOR_SOURCE)
        return str(src)

    @pytest.mark.asyncio
    async def test_atoms_are_valid_python(self, ecg_source):
        agent, _ = _make_agent()
        bundle = await agent.ingest(ecg_source, "ECGProcessor")
        ast.parse(bundle.generated_atoms)

    @pytest.mark.asyncio
    async def test_atoms_have_register_decorators(self, ecg_source):
        agent, _ = _make_agent()
        bundle = await agent.ingest(ecg_source, "ECGProcessor")
        assert bundle.generated_atoms.count("@register_atom") == 5

    @pytest.mark.asyncio
    async def test_atoms_have_five_functions(self, ecg_source):
        agent, _ = _make_agent()
        bundle = await agent.ingest(ecg_source, "ECGProcessor")

        tree = ast.parse(bundle.generated_atoms)
        fn_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        assert len(fn_names) == 5
        assert set(fn_names) == {
            "bandpass_filter",
            "r_peak_detection",
            "peak_correction",
            "template_extraction",
            "heart_rate_computation",
        }

    @pytest.mark.asyncio
    async def test_witnesses_are_valid_python(self, ecg_source):
        agent, _ = _make_agent()
        bundle = await agent.ingest(ecg_source, "ECGProcessor")
        ast.parse(bundle.generated_witnesses)

    @pytest.mark.asyncio
    async def test_witnesses_have_five_functions(self, ecg_source):
        agent, _ = _make_agent()
        bundle = await agent.ingest(ecg_source, "ECGProcessor")

        tree = ast.parse(bundle.generated_witnesses)
        fn_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        assert len(fn_names) == 5
        for name in fn_names:
            assert name.startswith("witness_")

    @pytest.mark.asyncio
    async def test_state_model_generated(self, ecg_source):
        agent, _ = _make_agent()
        bundle = await agent.ingest(ecg_source, "ECGProcessor")
        assert "ECGPipelineState" in bundle.generated_state_models
        assert "filtered" in bundle.generated_state_models
        assert "rpeaks" in bundle.generated_state_models
        ast.parse(bundle.generated_state_models)

    @pytest.mark.asyncio
    async def test_match_results_are_verified(self, ecg_source):
        agent, _ = _make_agent()
        bundle = await agent.ingest(ecg_source, "ECGProcessor")

        assert len(bundle.match_results) == 5
        for mr in bundle.match_results:
            assert mr.verified_match.verified is True


# ---------------------------------------------------------------------------
# Test: Write to ageo-atoms
# ---------------------------------------------------------------------------


class TestWriteToAgeoAtoms:
    """Write the ingestion output to ageo-atoms/ageoa/biosppy/."""

    @pytest.fixture
    def ecg_source(self, tmp_path) -> str:
        src = tmp_path / "ecg_processor.py"
        src.write_text(ECG_PROCESSOR_SOURCE)
        return str(src)

    @pytest.mark.asyncio
    async def test_write_artefacts(self, ecg_source, tmp_path):
        agent, _ = _make_agent()
        bundle = await agent.ingest(ecg_source, "ECGProcessor")

        output_dir = BIOSPPY_OUT
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            probe = output_dir / ".write_probe"
            probe.write_text("ok")
            probe.unlink()
        except OSError:
            # Sandbox/CI fallback: keep artefact writes inside temp workspace.
            output_dir = tmp_path / "biosppy_out"
            output_dir.mkdir(parents=True, exist_ok=True)

        # CDG
        cdg_path = output_dir / "ecg_cdg.json"
        save_json(bundle.cdg, cdg_path)
        assert cdg_path.exists()

        # Verify round-trip
        loaded = json.loads(cdg_path.read_text())
        assert len(loaded["nodes"]) == 6
        # 6 data-flow edges + state edges
        assert len(loaded["edges"]) >= 6

        # Atoms
        atoms_path = output_dir / "ecg.py"
        atoms_path.write_text(bundle.generated_atoms)
        assert atoms_path.exists()

        # Witnesses
        witnesses_path = output_dir / "ecg_witnesses.py"
        witnesses_path.write_text(bundle.generated_witnesses)
        assert witnesses_path.exists()

        # State models
        if bundle.generated_state_models:
            state_path = output_dir / "ecg_state.py"
            state_path.write_text(bundle.generated_state_models)
            assert state_path.exists()

        # __init__.py
        init_path = output_dir / "__init__.py"
        init_path.write_text(
            '"""BioSPPy ECG atoms ingested via the Smart Ingester."""\n'
        )
        assert init_path.exists()
