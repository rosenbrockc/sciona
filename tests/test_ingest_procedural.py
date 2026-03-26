"""Tests for Level 2 procedural SSA edge inference ingestion."""

from __future__ import annotations

import textwrap

import pytest

from sciona.architect.models import NodeStatus
from sciona.ingester.emitter import build_procedural_plan, emit_ingestion_bundle
from sciona.ingester.extractor import (
    _ProceduralBlockVisitor,
    extract_procedural_data_flow,
)

MOCK_SCRIPT = textwrap.dedent("""\
    import numpy as np

    def remove_baseline(signal: np.ndarray) -> np.ndarray:
        \"\"\"Remove baseline drift via high-pass filter.\"\"\"
        return signal - np.mean(signal)

    def fold_signal(signal: np.ndarray, period: float) -> np.ndarray:
        \"\"\"Fold a signal at the given period.\"\"\"
        n_bins = int(len(signal) / period)
        return signal[:n_bins * int(period)].reshape(n_bins, int(period)).mean(axis=0)

    def compute_snr(folded: np.ndarray) -> float:
        \"\"\"Compute signal-to-noise ratio.\"\"\"
        return float(np.max(folded) / np.std(folded))

    # --- Procedural pipeline ---
    raw = np.random.randn(10000)
    clean = remove_baseline(raw)
    folded = fold_signal(clean, 100.0)
    snr = compute_snr(folded)
""")


@pytest.fixture
def script_source(tmp_path):
    p = tmp_path / "pulsar_fold.py"
    p.write_text(MOCK_SCRIPT)
    return str(p)


# ---------------------------------------------------------------------------
# TestProceduralExtraction
# ---------------------------------------------------------------------------


class TestProceduralExtraction:
    @pytest.mark.asyncio
    async def test_finds_three_functions(self, script_source):
        dfg = await extract_procedural_data_flow(script_source)
        assert len(dfg.methods) == 3

    @pytest.mark.asyncio
    async def test_method_facts_names(self, script_source):
        dfg = await extract_procedural_data_flow(script_source)
        names = {m.name for m in dfg.methods}
        assert names == {"remove_baseline", "fold_signal", "compute_snr"}

    @pytest.mark.asyncio
    async def test_method_facts_have_params(self, script_source):
        dfg = await extract_procedural_data_flow(script_source)
        rb = next(m for m in dfg.methods if m.name == "remove_baseline")
        assert "signal" in rb.params

    @pytest.mark.asyncio
    async def test_call_sites_extracted(self, script_source):
        dfg = await extract_procedural_data_flow(script_source)
        # The visitor should have found call sites — verify via all_attributes
        assert "clean" in dfg.all_attributes
        assert "folded" in dfg.all_attributes

    @pytest.mark.asyncio
    async def test_var_producers(self, script_source):
        """Verify SSA var_producers maps variables to producing functions."""
        import ast as _ast
        from pathlib import Path

        source = Path(script_source).read_text()
        tree = _ast.parse(source)

        known = set()
        for node in tree.body:
            if isinstance(node, _ast.FunctionDef):
                known.add(node.name)

        visitor = _ProceduralBlockVisitor(known)
        for stmt in tree.body:
            if not isinstance(stmt, _ast.FunctionDef):
                visitor.visit(stmt)

        assert "clean" in visitor.var_producers
        assert visitor.var_producers["clean"].func_name == "remove_baseline"
        assert "folded" in visitor.var_producers
        assert visitor.var_producers["folded"].func_name == "fold_signal"
        assert "snr" in visitor.var_producers
        assert visitor.var_producers["snr"].func_name == "compute_snr"

    @pytest.mark.asyncio
    async def test_pipeline_name_defaults_to_stem(self, script_source):
        dfg = await extract_procedural_data_flow(script_source)
        assert dfg.class_name == "pulsar_fold"

    @pytest.mark.asyncio
    async def test_pipeline_name_override(self, script_source):
        dfg = await extract_procedural_data_flow(
            script_source, pipeline_name="MyPipeline"
        )
        assert dfg.class_name == "MyPipeline"


# ---------------------------------------------------------------------------
# TestSSAEdgeInference
# ---------------------------------------------------------------------------


class TestSSAEdgeInference:
    @pytest.mark.asyncio
    async def test_two_edges_inferred(self, script_source):
        dfg = await extract_procedural_data_flow(script_source)
        assert len(dfg.inferred_edges) == 2

    @pytest.mark.asyncio
    async def test_remove_baseline_to_fold_signal(self, script_source):
        dfg = await extract_procedural_data_flow(script_source)
        edge = next(
            (e for e in dfg.inferred_edges if e.source_id == "remove_baseline"),
            None,
        )
        assert edge is not None
        assert edge.target_id == "fold_signal"
        assert edge.output_name == "clean"

    @pytest.mark.asyncio
    async def test_fold_signal_to_compute_snr(self, script_source):
        dfg = await extract_procedural_data_flow(script_source)
        edge = next(
            (e for e in dfg.inferred_edges if e.source_id == "fold_signal"),
            None,
        )
        assert edge is not None
        assert edge.target_id == "compute_snr"
        assert edge.output_name == "folded"

    @pytest.mark.asyncio
    async def test_no_edge_from_raw(self, script_source):
        """raw is assigned from np.random.randn, not a known function."""
        dfg = await extract_procedural_data_flow(script_source)
        # No edge should have raw as an intermediary source from a known func
        # (np.random.randn is not in known_functions)
        assert all(
            e.source_id in {"remove_baseline", "fold_signal"}
            for e in dfg.inferred_edges
        )


# ---------------------------------------------------------------------------
# TestProceduralCDG
# ---------------------------------------------------------------------------


class TestProceduralCDG:
    @pytest.mark.asyncio
    async def test_procedural_plan_has_canonical_ir(self, script_source):
        dfg = await extract_procedural_data_flow(
            script_source, pipeline_name="PulsarFold"
        )
        plan = build_procedural_plan(dfg, "PulsarFold")

        assert plan.plan.canonical_ir is not None
        assert plan.plan.canonical_ir.source_language == "python"
        assert [op.operation_id for op in plan.plan.canonical_ir.operations] == [
            "remove_baseline",
            "fold_signal",
            "compute_snr",
        ]

    @pytest.mark.asyncio
    async def test_cdg_has_four_nodes(self, script_source):
        dfg = await extract_procedural_data_flow(
            script_source, pipeline_name="PulsarFold"
        )
        plan = build_procedural_plan(dfg, "PulsarFold")
        bundle = emit_ingestion_bundle(plan, "PulsarFold", script_source)
        # 1 DECOMPOSED root + 3 ATOMIC children
        assert len(bundle.cdg.nodes) == 4

    @pytest.mark.asyncio
    async def test_cdg_root_is_decomposed(self, script_source):
        dfg = await extract_procedural_data_flow(
            script_source, pipeline_name="PulsarFold"
        )
        plan = build_procedural_plan(dfg, "PulsarFold")
        bundle = emit_ingestion_bundle(plan, "PulsarFold", script_source)
        root = next(n for n in bundle.cdg.nodes if n.status == NodeStatus.DECOMPOSED)
        assert root is not None
        assert len(root.children) == 3

    @pytest.mark.asyncio
    async def test_cdg_has_two_edges(self, script_source):
        dfg = await extract_procedural_data_flow(
            script_source, pipeline_name="PulsarFold"
        )
        plan = build_procedural_plan(dfg, "PulsarFold")
        bundle = emit_ingestion_bundle(plan, "PulsarFold", script_source)
        assert len(bundle.cdg.edges) == 2

    @pytest.mark.asyncio
    async def test_cdg_edges_match_ssa(self, script_source):
        dfg = await extract_procedural_data_flow(
            script_source, pipeline_name="PulsarFold"
        )
        plan = build_procedural_plan(dfg, "PulsarFold")
        bundle = emit_ingestion_bundle(plan, "PulsarFold", script_source)
        edge_pairs = {(e.source_id, e.target_id) for e in bundle.cdg.edges}
        assert ("remove_baseline", "fold_signal") in edge_pairs
        assert ("fold_signal", "compute_snr") in edge_pairs

    @pytest.mark.asyncio
    async def test_cdg_metadata_source(self, script_source):
        dfg = await extract_procedural_data_flow(
            script_source, pipeline_name="PulsarFold"
        )
        plan = build_procedural_plan(dfg, "PulsarFold")
        bundle = emit_ingestion_bundle(plan, "PulsarFold", script_source)
        assert bundle.cdg.metadata.get("source") == "ingester"


# ---------------------------------------------------------------------------
# TestProceduralBundle
# ---------------------------------------------------------------------------


class TestProceduralBundle:
    @pytest.mark.asyncio
    async def test_bundle_has_generated_atoms(self, script_source):
        dfg = await extract_procedural_data_flow(
            script_source, pipeline_name="PulsarFold"
        )
        plan = build_procedural_plan(dfg, "PulsarFold")
        bundle = emit_ingestion_bundle(plan, "PulsarFold", script_source)
        assert bundle.generated_atoms
        assert len(bundle.generated_atoms) > 0

    @pytest.mark.asyncio
    async def test_bundle_cdg_edges_correct(self, script_source):
        dfg = await extract_procedural_data_flow(
            script_source, pipeline_name="PulsarFold"
        )
        plan = build_procedural_plan(dfg, "PulsarFold")
        bundle = emit_ingestion_bundle(plan, "PulsarFold", script_source)
        edge_pairs = {(e.source_id, e.target_id) for e in bundle.cdg.edges}
        assert ("remove_baseline", "fold_signal") in edge_pairs
        assert ("fold_signal", "compute_snr") in edge_pairs

    @pytest.mark.asyncio
    async def test_bundle_uses_canonical_edges_when_legacy_exports_are_empty(self, script_source):
        dfg = await extract_procedural_data_flow(
            script_source, pipeline_name="PulsarFold"
        )
        plan = build_procedural_plan(dfg, "PulsarFold")
        plan = plan.model_copy(
            update={
                "plan": plan.plan.model_copy(
                    update={
                        "macro_atoms": [],
                        "edge_definitions": [],
                    }
                ),
            }
        )
        assert plan.plan.macro_atoms == []
        assert plan.plan.edge_definitions == []

        bundle = emit_ingestion_bundle(plan, "PulsarFold", script_source)

        edge_pairs = {(e.source_id, e.target_id) for e in bundle.cdg.edges}
        assert ("remove_baseline", "fold_signal") in edge_pairs
        assert ("fold_signal", "compute_snr") in edge_pairs
        assert plan.plan.macro_atoms == []
        assert plan.plan.edge_definitions == []

    @pytest.mark.asyncio
    async def test_bundle_match_results_count(self, script_source):
        dfg = await extract_procedural_data_flow(
            script_source, pipeline_name="PulsarFold"
        )
        plan = build_procedural_plan(dfg, "PulsarFold")
        bundle = emit_ingestion_bundle(plan, "PulsarFold", script_source)
        assert len(bundle.match_results) == 3

    @pytest.mark.asyncio
    async def test_bundle_match_results_verified(self, script_source):
        dfg = await extract_procedural_data_flow(
            script_source, pipeline_name="PulsarFold"
        )
        plan = build_procedural_plan(dfg, "PulsarFold")
        bundle = emit_ingestion_bundle(plan, "PulsarFold", script_source)
        for mr in bundle.match_results:
            assert mr.verified_match.verified is True

    @pytest.mark.asyncio
    async def test_ingest_procedural_e2e(self, script_source):
        """Full pipeline via IngesterAgent.ingest_procedural."""
        from unittest.mock import AsyncMock

        from sciona.ingester.graph import IngesterAgent

        mock_llm = AsyncMock()
        agent = IngesterAgent(llm=mock_llm)
        bundle = await agent.ingest_procedural(script_source, "PulsarFold")

        assert bundle.generated_atoms
        assert len(bundle.cdg.edges) == 2
        assert len(bundle.match_results) == 3
        for mr in bundle.match_results:
            assert mr.verified_match.verified is True
