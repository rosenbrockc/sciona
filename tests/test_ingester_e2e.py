"""End-to-end tests for the ingester pipeline with mocked LLM and mypy."""

from __future__ import annotations

import json
import textwrap
from unittest.mock import AsyncMock

import pytest

from sciona.architect.models import NodeStatus
from sciona.ingester import IngesterAgent, IngestionBundle

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SAMPLE_CLASS = textwrap.dedent("""\
    class SamplePipeline:
        def __init__(self, data, options):
            self.options = options
            self.raw = data
            self.processed = None

        def process(self):
            self.processed = self.raw * 2
            if self.options.normalize:
                self.processed = self.processed / max(self.processed)
            return self.processed

        def summarize(self):
            return sum(self.processed)
""")


def _llm_response_chunk() -> str:
    return json.dumps(
        {
            "macro_atoms": [
                {
                    "name": "Data Processor",
                    "description": "Transform raw data",
                    "method_names": ["__init__", "process"],
                    "inputs": [
                        {"name": "data", "type_desc": "list[float]", "constraints": ""}
                    ],
                    "outputs": [
                        {
                            "name": "processed",
                            "type_desc": "list[float]",
                            "constraints": "",
                        }
                    ],
                    "config_params": ["normalize"],
                    "concept_type": "custom",
                    "is_optional": False,
                },
                {
                    "name": "Summarizer",
                    "description": "Summarize processed data",
                    "method_names": ["summarize"],
                    "inputs": [
                        {
                            "name": "processed",
                            "type_desc": "list[float]",
                            "constraints": "",
                        }
                    ],
                    "outputs": [
                        {"name": "summary", "type_desc": "float", "constraints": ""}
                    ],
                    "config_params": [],
                    "concept_type": "custom",
                    "is_optional": False,
                },
            ],
            "edges": [
                {
                    "source_id": "data_processor",
                    "target_id": "summarizer",
                    "output_name": "processed",
                    "input_name": "processed",
                    "source_type": "list[float]",
                    "target_type": "list[float]",
                }
            ],
        }
    )


@pytest.fixture
def sample_source(tmp_path):
    p = tmp_path / "sample.py"
    p.write_text(SAMPLE_CLASS)
    return str(p)


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.complete.return_value = _llm_response_chunk()
    return llm


@pytest.fixture
def mock_proof_env():
    env = AsyncMock()
    env.prover_name = "python"
    env.check_proof.return_value = (True, "")
    env.close.return_value = None
    return env


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIngesterE2E:
    @pytest.mark.asyncio
    async def test_full_pipeline(self, sample_source, mock_llm, mock_proof_env):
        agent = IngesterAgent(
            llm=mock_llm,
            proof_env=mock_proof_env,
        )
        bundle = await agent.ingest(sample_source, "SamplePipeline")

        assert isinstance(bundle, IngestionBundle)
        assert len(bundle.cdg.nodes) > 0
        assert len(bundle.match_results) > 0

    @pytest.mark.asyncio
    async def test_cdg_has_correct_structure(
        self, sample_source, mock_llm, mock_proof_env
    ):
        agent = IngesterAgent(llm=mock_llm, proof_env=mock_proof_env)
        bundle = await agent.ingest(sample_source, "SamplePipeline")

        # Root should be DECOMPOSED
        root = next(
            (n for n in bundle.cdg.nodes if n.status == NodeStatus.DECOMPOSED),
            None,
        )
        assert root is not None
        assert root.name == "SamplePipeline"

        # Children should be ATOMIC
        atomic = [n for n in bundle.cdg.nodes if n.status == NodeStatus.ATOMIC]
        assert len(atomic) == 2

    @pytest.mark.asyncio
    async def test_match_results_compatible(
        self, sample_source, mock_llm, mock_proof_env
    ):
        """MatchResults should be serializable and have verified=True."""
        agent = IngesterAgent(llm=mock_llm, proof_env=mock_proof_env)
        bundle = await agent.ingest(sample_source, "SamplePipeline")

        for mr in bundle.match_results:
            assert mr.success is True
            # Should be JSON-serializable
            d = mr.to_dict()
            assert d["verified_match"]["verified"] is True

    @pytest.mark.asyncio
    async def test_generated_source_not_empty(
        self, sample_source, mock_llm, mock_proof_env
    ):
        agent = IngesterAgent(llm=mock_llm, proof_env=mock_proof_env)
        bundle = await agent.ingest(sample_source, "SamplePipeline")

        assert bundle.generated_atoms != ""
        assert bundle.generated_witnesses != ""

    @pytest.mark.asyncio
    async def test_handles_missing_class(self, sample_source, mock_llm, mock_proof_env):
        agent = IngesterAgent(llm=mock_llm, proof_env=mock_proof_env)
        bundle = await agent.ingest(sample_source, "NonExistentClass")

        # Should not crash; bundle CDG will be empty
        assert len(bundle.cdg.nodes) == 0

    @pytest.mark.asyncio
    async def test_no_proof_env(self, sample_source, mock_llm):
        """Pipeline should work without proof environment (skips mypy)."""
        agent = IngesterAgent(llm=mock_llm, proof_env=None)
        bundle = await agent.ingest(sample_source, "SamplePipeline")

        assert isinstance(bundle, IngestionBundle)
        assert len(bundle.cdg.nodes) > 0
