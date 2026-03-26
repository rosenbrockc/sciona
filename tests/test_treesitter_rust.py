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


RUST_STATEFUL_STRUCT = textwrap.dedent("""\
    struct Integrator {
        position: f64,
        velocity: f64,
    }

    impl Integrator {
        fn step(&mut self, dt: f64) {
            self.position = self.position + self.velocity * dt;
        }

        fn get_position(&self) -> f64 {
            return self.position;
        }
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


@pytest.fixture
def rust_stateful_source(tmp_path):
    p = tmp_path / "integrator.rs"
    p.write_text(RUST_STATEFUL_STRUCT)
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

    @pytest.mark.asyncio
    async def test_oracle_call_facts_are_populated(self, extractor, rust_struct_source):
        dfg = await extractor.extract_class(rust_struct_source, "Sampler")
        step = next(m for m in dfg.methods if m.name == "step")
        assert {fact.resolved_target for fact in step.call_facts} >= {
            "log_density",
            "gradient",
        }


class TestRustSemanticFacts:
    @pytest.mark.asyncio
    async def test_signature_return_facts_and_roles(self, extractor, rust_stateful_source):
        dfg = await extractor.extract_class(rust_stateful_source, "Integrator")

        step = next(m for m in dfg.methods if m.name == "step")
        get_position = next(m for m in dfg.methods if m.name == "get_position")

        assert [param.name for param in step.signature] == ["dt"]
        assert step.signature[0].annotation == "f64"
        assert step.semantic_role == "fit_or_update"
        assert get_position.return_facts[0].kind == "attribute"
        assert get_position.return_facts[0].referenced_attrs == ["position"]
        assert get_position.semantic_role == "query_or_metadata"
