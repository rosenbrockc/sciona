"""Tests for deep-learning black-box boundary detection in the Smart Ingester.

Covers:
- Phase 1 opaque detection (_detect_opaque_bases, _extract_opaque_boundary_fact)
- Full pipeline with opaque DL modules (chunker short-circuit, CDG propagation)
- LLM-drafted opaque witnesses and fallback behaviour
"""

from __future__ import annotations

import json
import textwrap
from unittest.mock import AsyncMock

import pytest

from ageom.architect.models import ConceptType, NodeStatus
from ageom.ingester import IngesterAgent, IngestionBundle
from ageom.ingester.emitter import (
    _opaque_witness_fallback,
    generate_opaque_witnesses,
)
from ageom.ingester.extractor import (
    extract_data_flow,
)
from ageom.ingester.models import MacroAtomSpec

# ---------------------------------------------------------------------------
# Mock source: mixed pipeline with transparent + opaque classes
# ---------------------------------------------------------------------------

MIXED_PIPELINE_SOURCE = textwrap.dedent("""\
    class Preprocessor:
        \"\"\"Transparent class: normalize signal.\"\"\"

        def __init__(self, signal):
            self.raw = signal
            self.normalized = None

        def normalize(self):
            self.normalized = self.raw / max(self.raw)
            return self.normalized


    class FeatureExtractor(nn.Module):
        \"\"\"Opaque PyTorch module.\"\"\"

        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3)

        def forward(self, x):
            \"\"\"Extract features from input tensor.\"\"\"
            return self.conv(x)


    class AlphaFoldBlock(hk.Module):
        \"\"\"Opaque Haiku module for MSA processing.\"\"\"

        def __init__(self, config):
            super().__init__(name="alphafold_block")
            self.config = config

        def __call__(self, msa_repr, pair_repr):
            \"\"\"Process MSA and pair representations.\"\"\"
            return msa_repr + pair_repr
""")


# Mock LLM response for opaque witness drafting
_OPAQUE_WITNESS_RESPONSE = json.dumps(
    {
        "witness_name": "witness_feature_extractor",
        "params": ["x: AbstractArray"],
        "return_type": "AbstractArray",
        "shape_transform": "(B, N, C_in) -> (B, N, C_out)",
        "witness_body": (
            "B, N, C_in = x.shape\n"
            "C_out = 64  # default output channels\n"
            'return AbstractArray(shape=(B, N, C_out), dtype="float32")'
        ),
    }
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_source(tmp_path):
    p = tmp_path / "mixed_pipeline.py"
    p.write_text(MIXED_PIPELINE_SOURCE)
    return str(p)


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    # Default: return chunker response (won't be called for opaque)
    llm.complete.return_value = "{}"
    return llm


@pytest.fixture
def mock_llm_with_witness():
    llm = AsyncMock()
    llm.complete.return_value = _OPAQUE_WITNESS_RESPONSE
    return llm


@pytest.fixture
def mock_llm_failing():
    llm = AsyncMock()
    llm.complete.side_effect = RuntimeError("LLM unavailable")
    return llm


# ---------------------------------------------------------------------------
# TestPhase1OpaqueDetection
# ---------------------------------------------------------------------------


class TestPhase1OpaqueDetection:
    """Test that Phase 1 correctly detects opaque DL base classes."""

    @pytest.mark.asyncio
    async def test_preprocessor_is_not_opaque(self, mixed_source):
        dfg = await extract_data_flow(mixed_source, "Preprocessor")
        assert dfg.is_opaque is False
        assert dfg.opaque_base_classes == []

    @pytest.mark.asyncio
    async def test_feature_extractor_is_opaque(self, mixed_source):
        dfg = await extract_data_flow(mixed_source, "FeatureExtractor")
        assert dfg.is_opaque is True
        assert "nn.Module" in dfg.opaque_base_classes

    @pytest.mark.asyncio
    async def test_alphafold_block_is_opaque(self, mixed_source):
        dfg = await extract_data_flow(mixed_source, "AlphaFoldBlock")
        assert dfg.is_opaque is True
        assert "hk.Module" in dfg.opaque_base_classes

    @pytest.mark.asyncio
    async def test_opaque_has_empty_reads_writes(self, mixed_source):
        dfg = await extract_data_flow(mixed_source, "FeatureExtractor")
        assert dfg.is_opaque is True
        # Opaque boundary fact should have empty reads/writes
        assert len(dfg.methods) == 1
        mf = dfg.methods[0]
        assert mf.reads == []
        assert mf.writes == []
        assert mf.calls == []

    @pytest.mark.asyncio
    async def test_forward_params_extracted(self, mixed_source):
        dfg = await extract_data_flow(mixed_source, "FeatureExtractor")
        mf = dfg.methods[0]
        assert mf.name == "forward"
        assert mf.params == ["x"]

    @pytest.mark.asyncio
    async def test_alphafold_call_params_extracted(self, mixed_source):
        dfg = await extract_data_flow(mixed_source, "AlphaFoldBlock")
        mf = dfg.methods[0]
        assert mf.name == "__call__"
        assert mf.params == ["msa_repr", "pair_repr"]

    @pytest.mark.asyncio
    async def test_opaque_boundary_fact_is_opaque(self, mixed_source):
        dfg = await extract_data_flow(mixed_source, "FeatureExtractor")
        assert dfg.methods[0].is_opaque is True

    @pytest.mark.asyncio
    async def test_transparent_has_normal_methods(self, mixed_source):
        dfg = await extract_data_flow(mixed_source, "Preprocessor")
        method_names = [m.name for m in dfg.methods]
        assert "__init__" in method_names
        assert "normalize" in method_names
        # Should have self.* reads and writes
        assert len(dfg.all_attributes) > 0


# ---------------------------------------------------------------------------
# TestFullPipelineOpaque
# ---------------------------------------------------------------------------


class TestFullPipelineOpaque:
    """Test the full ingester pipeline with opaque DL modules."""

    @pytest.mark.asyncio
    async def test_cdg_has_opaque_atomic_node(self, mixed_source, mock_llm):
        agent = IngesterAgent(llm=mock_llm, proof_env=None)
        bundle = await agent.ingest(mixed_source, "FeatureExtractor")

        assert isinstance(bundle, IngestionBundle)

        # Should have root + 1 child
        atomic_nodes = [n for n in bundle.cdg.nodes if n.status == NodeStatus.ATOMIC]
        assert len(atomic_nodes) == 1

        node = atomic_nodes[0]
        assert node.is_opaque is True
        assert node.concept_type == ConceptType.NEURAL_NETWORK

    @pytest.mark.asyncio
    async def test_witnesses_use_abstract_array(self, mixed_source, mock_llm):
        agent = IngesterAgent(llm=mock_llm, proof_env=None)
        bundle = await agent.ingest(mixed_source, "FeatureExtractor")

        assert "AbstractArray" in bundle.generated_witnesses

    @pytest.mark.asyncio
    async def test_generated_code_is_valid_python(self, mixed_source, mock_llm):
        agent = IngesterAgent(llm=mock_llm, proof_env=None)
        bundle = await agent.ingest(mixed_source, "FeatureExtractor")

        # Witness code should compile without errors
        compile(bundle.generated_witnesses, "<witness>", "exec")

    @pytest.mark.asyncio
    async def test_alphafold_block_has_two_inputs(self, mixed_source, mock_llm):
        agent = IngesterAgent(llm=mock_llm, proof_env=None)
        bundle = await agent.ingest(mixed_source, "AlphaFoldBlock")

        atomic_nodes = [n for n in bundle.cdg.nodes if n.status == NodeStatus.ATOMIC]
        assert len(atomic_nodes) == 1
        node = atomic_nodes[0]
        assert len(node.inputs) == 2
        input_names = [i.name for i in node.inputs]
        assert "msa_repr" in input_names
        assert "pair_repr" in input_names

    @pytest.mark.asyncio
    async def test_opaque_skips_llm_chunker(self, mixed_source, mock_llm):
        """LLM should NOT be called for opaque classes (deterministic chunking)."""
        agent = IngesterAgent(llm=mock_llm, proof_env=None)
        await agent.ingest(mixed_source, "FeatureExtractor")

        # The LLM mock should only be called for opaque witness drafting,
        # NOT for semantic chunking or state hoisting
        for call in mock_llm.complete.call_args_list:
            system_prompt = call[0][0] if call[0] else ""
            assert "group the methods" not in system_prompt.lower()


# ---------------------------------------------------------------------------
# TestLLMDraftedWitness
# ---------------------------------------------------------------------------


class TestLLMDraftedWitness:
    """Test LLM-drafted opaque witnesses and fallback."""

    @pytest.mark.asyncio
    async def test_llm_drafted_witness_body(self, mixed_source, mock_llm_with_witness):
        dfg = await extract_data_flow(mixed_source, "FeatureExtractor")

        from ageom.architect.models import IOSpec

        atoms = [
            MacroAtomSpec(
                name="FeatureExtractor",
                method_names=["forward"],
                inputs=[IOSpec(name="x", type_desc="Any")],
                outputs=[IOSpec(name="output", type_desc="Any")],
                concept_type=ConceptType.NEURAL_NETWORK,
                is_opaque=True,
            )
        ]

        witness_source, name_map = await generate_opaque_witnesses(
            atoms, dfg, mock_llm_with_witness
        )

        assert "witness_featureextractor" in name_map.values()
        # Should contain the LLM-drafted body
        assert "C_out = 64" in witness_source
        assert "AbstractArray" in witness_source

    @pytest.mark.asyncio
    async def test_fallback_when_llm_fails(self, mixed_source, mock_llm_failing):
        dfg = await extract_data_flow(mixed_source, "FeatureExtractor")

        from ageom.architect.models import IOSpec

        atoms = [
            MacroAtomSpec(
                name="FeatureExtractor",
                method_names=["forward"],
                inputs=[IOSpec(name="x", type_desc="Any")],
                outputs=[IOSpec(name="output", type_desc="Any")],
                concept_type=ConceptType.NEURAL_NETWORK,
                is_opaque=True,
            )
        ]

        witness_source, name_map = await generate_opaque_witnesses(
            atoms, dfg, mock_llm_failing
        )

        # Should fall back to shape-preserving default
        assert "witness_featureextractor" in name_map.values()
        assert "x.shape" in witness_source
        assert 'dtype="float32"' in witness_source

    def test_fallback_witness_valid_python(self):
        from ageom.architect.models import IOSpec

        atom = MacroAtomSpec(
            name="TestModule",
            method_names=["forward"],
            inputs=[IOSpec(name="x", type_desc="Any")],
            outputs=[IOSpec(name="output", type_desc="Any")],
            concept_type=ConceptType.NEURAL_NETWORK,
            is_opaque=True,
        )

        code = _opaque_witness_fallback(atom)
        # Should compile
        compile(code, "<fallback>", "exec")
        assert "witness_testmodule" in code
        assert "AbstractArray" in code

    @pytest.mark.asyncio
    async def test_full_pipeline_with_llm_witness(
        self, mixed_source, mock_llm_with_witness
    ):
        agent = IngesterAgent(llm=mock_llm_with_witness, proof_env=None)
        bundle = await agent.ingest(mixed_source, "FeatureExtractor")

        # LLM-drafted witness should be in the bundle
        assert "AbstractArray" in bundle.generated_witnesses
        # Generated witness code should be valid Python
        compile(bundle.generated_witnesses, "<witness>", "exec")
