"""Tests for config-gated branch flattening and mypy repair loop.

Exercises:
1. Config branch detection in the AST extractor
2. Optional flag propagation from MacroAtomSpec → AlgorithmicNode in the CDG
3. The mypy verification/repair loop with a mock ProofEnvironment
"""

from __future__ import annotations

import json
import textwrap
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sciona.architect.models import NodeStatus
from sciona.ingester.extractor import extract_data_flow
from sciona.ingester.graph import IngesterAgent
from sciona.ingester.models import IngestionBundle

# ---------------------------------------------------------------------------
# Mock class source
# ---------------------------------------------------------------------------

SMOOTHED_ESTIMATOR_SOURCE = textwrap.dedent("""\
    class SmoothedEstimator:
        def __init__(self, data, options):
            self.options = options
            self.data = data
            self.smoothed = None
            self.result = None

        def preprocess(self):
            self.smoothed = self.data
            if self.options.smooth:
                self.smoothed = self._apply_smooth(self.smoothed)

        def _apply_smooth(self, sig):
            return sig

        def estimate(self):
            self.result = sum(self.smoothed) / len(self.smoothed)
            return self.result
""")


# ---------------------------------------------------------------------------
# Mock LLM responses
# ---------------------------------------------------------------------------

_CHUNK_RESPONSE = json.dumps(
    {
        "macro_atoms": [
            {
                "name": "Data Smoother",
                "description": "Preprocess and optionally smooth the input data",
                "method_names": ["preprocess", "_apply_smooth"],
                "inputs": [
                    {"name": "data", "type_desc": "list[float]", "constraints": ""},
                ],
                "outputs": [
                    {"name": "smoothed", "type_desc": "list[float]", "constraints": ""},
                ],
                "config_params": ["smooth"],
                "concept_type": "custom",
                "is_optional": True,
            },
            {
                "name": "Estimator",
                "description": "Compute the mean of the smoothed data",
                "method_names": ["estimate"],
                "inputs": [
                    {"name": "smoothed", "type_desc": "list[float]", "constraints": ""},
                ],
                "outputs": [
                    {"name": "result", "type_desc": "float", "constraints": ""},
                ],
                "config_params": [],
                "concept_type": "custom",
                "is_optional": False,
            },
        ],
        "edges": [
            {
                "source_id": "data_smoother",
                "target_id": "estimator",
                "output_name": "smoothed",
                "input_name": "smoothed",
                "source_type": "list[float]",
                "target_type": "list[float]",
            },
        ],
    }
)

# Hoist response: no cross-window state for this class
_HOIST_RESPONSE = json.dumps({"state_models": []})

# Repair response: a minimal type fix (used in repair-loop test)
_REPAIR_RESPONSE = json.dumps(
    [
        {
            "line_start": 1,
            "line_end": 1,
            "replacement": '"""Auto-generated atom wrappers following the sciona.atoms pattern."""',
        },
    ]
)

_ABSTRACT_RESPONSE = json.dumps(
    {
        "abstract_name": "ConceptualAtom",
        "conceptual_transform": "Transform data",
        "abstract_inputs": [],
        "abstract_outputs": [],
        "algorithmic_properties": [],
        "cross_disciplinary_applications": [],
    }
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_source(tmp_path):
    p = tmp_path / "smoothed_estimator.py"
    p.write_text(SMOOTHED_ESTIMATOR_SOURCE)
    return str(p)


# ---------------------------------------------------------------------------
# Test: config branch detection
# ---------------------------------------------------------------------------


class TestConfigBranchDetection:
    @pytest.mark.asyncio
    async def test_detects_options_smooth_branch(self, sample_source):
        dfg = await extract_data_flow(sample_source, "SmoothedEstimator")

        assert len(dfg.config_branches) == 1
        cb = dfg.config_branches[0]
        assert cb.config_attr == "smooth"
        assert cb.method == "preprocess"

    @pytest.mark.asyncio
    async def test_config_branch_captures_reads_writes(self, sample_source):
        dfg = await extract_data_flow(sample_source, "SmoothedEstimator")

        cb = dfg.config_branches[0]
        # Inside `if self.options.smooth:`, the body writes self.smoothed
        assert "smoothed" in cb.writes


# ---------------------------------------------------------------------------
# Test: optional flag propagation to CDG
# ---------------------------------------------------------------------------


class TestOptionalCDGNode:
    @pytest.mark.asyncio
    async def test_cdg_has_two_atomic_nodes(self, sample_source):
        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = [
            _CHUNK_RESPONSE,
            _HOIST_RESPONSE,
            _ABSTRACT_RESPONSE,
            _ABSTRACT_RESPONSE,
        ]

        mock_ghost_mod = MagicMock()
        with patch.dict("sys.modules", {"sciona.synthesizer.ghost_sim": mock_ghost_mod}):
            mock_ghost_mod.run_ghost_simulation.return_value = MagicMock(
                passed=True, ran=True, error=""
            )
            agent = IngesterAgent(llm=mock_llm)
            bundle = await agent.ingest(sample_source, "SmoothedEstimator")

        atomic = [n for n in bundle.cdg.nodes if n.status == NodeStatus.ATOMIC]
        assert len(atomic) == 2

    @pytest.mark.asyncio
    async def test_data_smoother_is_optional(self, sample_source):
        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = [
            _CHUNK_RESPONSE,
            _HOIST_RESPONSE,
            _ABSTRACT_RESPONSE,
            _ABSTRACT_RESPONSE,
        ]

        mock_ghost_mod = MagicMock()
        with patch.dict("sys.modules", {"sciona.synthesizer.ghost_sim": mock_ghost_mod}):
            mock_ghost_mod.run_ghost_simulation.return_value = MagicMock(
                passed=True, ran=True, error=""
            )
            agent = IngesterAgent(llm=mock_llm)
            bundle = await agent.ingest(sample_source, "SmoothedEstimator")

        smoother = next(
            (n for n in bundle.cdg.nodes if n.name == "Data Smoother"), None
        )
        assert smoother is not None
        assert smoother.is_optional is True

    @pytest.mark.asyncio
    async def test_estimator_is_not_optional(self, sample_source):
        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = [
            _CHUNK_RESPONSE,
            _HOIST_RESPONSE,
            _ABSTRACT_RESPONSE,
            _ABSTRACT_RESPONSE,
        ]

        mock_ghost_mod = MagicMock()
        with patch.dict("sys.modules", {"sciona.synthesizer.ghost_sim": mock_ghost_mod}):
            mock_ghost_mod.run_ghost_simulation.return_value = MagicMock(
                passed=True, ran=True, error=""
            )
            agent = IngesterAgent(llm=mock_llm)
            bundle = await agent.ingest(sample_source, "SmoothedEstimator")

        estimator = next((n for n in bundle.cdg.nodes if n.name == "Estimator"), None)
        assert estimator is not None
        assert estimator.is_optional is False


# ---------------------------------------------------------------------------
# Test: mypy verification/repair loop
# ---------------------------------------------------------------------------


class TestMypyRepairLoop:
    @pytest.mark.asyncio
    async def test_semantic_type_failure_does_not_enter_repair(self, sample_source):
        mock_proof_env = AsyncMock()
        mock_proof_env.prover_name = "python"
        mock_proof_env.check_proof.return_value = (
            False,
            'atoms.py:1: error: Too many arguments for "Estimator"',
        )
        mock_proof_env.close.return_value = None

        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = [
            _CHUNK_RESPONSE,
            _HOIST_RESPONSE,
            _ABSTRACT_RESPONSE,
            _ABSTRACT_RESPONSE,
        ]

        mock_ghost_mod = MagicMock()
        with patch.dict("sys.modules", {"sciona.synthesizer.ghost_sim": mock_ghost_mod}):
            mock_ghost_mod.run_ghost_simulation.return_value = MagicMock(
                passed=True, ran=True, error=""
            )
            agent = IngesterAgent(llm=mock_llm, proof_env=mock_proof_env)
            final_state = await agent.ingest_state(sample_source, "SmoothedEstimator")

        assert final_state["type_repair_count"] == 0
        assert final_state["type_failure_classification"]["reason_code"] == "semantic_signature"
        assert mock_proof_env.check_proof.call_count == 1
        assert mock_llm.complete.call_count == 4

    @pytest.mark.asyncio
    async def test_semantic_type_failure_preserves_partial_bundle(self, sample_source):
        mock_proof_env = AsyncMock()
        mock_proof_env.prover_name = "python"
        mock_proof_env.check_proof.return_value = (
            False,
            'atoms.py:1: error: Too many arguments for "Estimator"',
        )
        mock_proof_env.close.return_value = None

        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = [
            _CHUNK_RESPONSE,
            _HOIST_RESPONSE,
            _ABSTRACT_RESPONSE,
            _ABSTRACT_RESPONSE,
        ]

        mock_ghost_mod = MagicMock()
        with patch.dict("sys.modules", {"sciona.synthesizer.ghost_sim": mock_ghost_mod}):
            mock_ghost_mod.run_ghost_simulation.return_value = MagicMock(
                passed=True, ran=True, error=""
            )
            agent = IngesterAgent(llm=mock_llm, proof_env=mock_proof_env)
            bundle = await agent.ingest(sample_source, "SmoothedEstimator")

        assert len(bundle.cdg.nodes) > 0
        atomic = [n for n in bundle.cdg.nodes if n.status == NodeStatus.ATOMIC]
        assert len(atomic) == 2
        assert mock_proof_env.check_proof.call_count == 1
