"""Tests for Rust tree-sitter extraction (sciona.ingester.treesitter_extractor)."""

from __future__ import annotations

import textwrap

import pytest

from sciona.ingester.base_extractor import SourceLanguage
from sciona.ingester.treesitter_extractor import TreeSitterExtractor

RUST_ORACLE_STRUCT = textwrap.dedent("""\
    struct State {
        x: f64,
    }

    struct Sampler<T> {
        target: T,
    }

    impl<T: BatchedGradientTarget> Sampler<T> {
        fn step(&self, state: State) {
            let lp = self.target.log_density(state.x);
            let grad = self.target.gradient(state.x);
        }
    }
""")


RUST_ORACLE_PROCEDURAL = textwrap.dedent("""\
    struct State {
        x: f64,
    }

    fn run_kernel<T>(target: &T, state: State)
    where
        T: BatchedGradientTarget,
    {
        let lp = target.log_density(state.x);
        let grad = target.gradient(state.x);
    }
""")


@pytest.fixture
def extractor():
    return TreeSitterExtractor(SourceLanguage.RUST)


@pytest.fixture
def rust_struct_source(tmp_path):
    p = tmp_path / "sampler.rs"
    p.write_text(RUST_ORACLE_STRUCT)
    return str(p)


@pytest.fixture
def rust_proc_source(tmp_path):
    p = tmp_path / "kernel.rs"
    p.write_text(RUST_ORACLE_PROCEDURAL)
    return str(p)


class TestRustOracleSubgraph:
    @pytest.mark.asyncio
    async def test_trait_bound_marks_oracle_method(self, extractor, rust_struct_source):
        dfg = await extractor.extract_class(rust_struct_source, "Sampler")
        step = next(m for m in dfg.methods if m.name == "step")
        assert step.is_oracle is True

    @pytest.mark.asyncio
    async def test_oracle_edges_and_flows(self, extractor, rust_struct_source):
        dfg = await extractor.extract_class(rust_struct_source, "Sampler")
        assert any(e.caller == "step" for e in dfg.oracle_edges)
        assert any(
            "oracle_subgraph::BatchedGradientTarget::step" in e.oracle_ref
            for e in dfg.oracle_edges
        )

        # state variables routed to oracle subgraph
        assert any(
            e.source_id.startswith("state:")
            and "oracle_subgraph::BatchedGradientTarget::step" in e.target_id
            for e in dfg.inferred_edges
        )
        # oracle outputs routed back to caller
        assert any(
            "oracle_subgraph::BatchedGradientTarget::step" in e.source_id
            and e.target_id == "step"
            and e.output_name == "log_prob"
            for e in dfg.inferred_edges
        )
        assert any(
            "oracle_subgraph::BatchedGradientTarget::step" in e.source_id
            and e.target_id == "step"
            and e.output_name == "gradient"
            for e in dfg.inferred_edges
        )

    @pytest.mark.asyncio
    async def test_procedural_where_clause_detection(self, extractor, rust_proc_source):
        dfg = await extractor.extract_procedural(rust_proc_source)
        run_kernel = next(m for m in dfg.methods if m.name == "run_kernel")
        assert run_kernel.is_oracle is True
        assert any(e.caller == "run_kernel" for e in dfg.oracle_edges)
        assert any(
            e.target_id == "run_kernel" and e.output_name == "gradient"
            for e in dfg.inferred_edges
        )
