"""Tests for Phase 1 AST extraction (ageom.ingester.extractor)."""

from __future__ import annotations

import textwrap

import pytest

from ageom.ingester.extractor import extract_data_flow

# ---------------------------------------------------------------------------
# Fixtures: write minimal Python classes to tmp_path
# ---------------------------------------------------------------------------


SIMPLE_CLASS = textwrap.dedent("""\
    class SimpleProcessor:
        def __init__(self, data):
            self.raw = data
            self.processed = None

        def process(self):
            self.processed = self.raw * 2
            return self.processed
""")

STATEFUL_CLASS = textwrap.dedent("""\
    class StatefulPipeline:
        def __init__(self, signal, options):
            self.options = options
            self.raw = signal
            self.filtered = None
            self.history = []

        def preprocess(self):
            self.filtered = self.raw
            if self.options.apply_filter:
                self.filtered = self._apply_filter(self.filtered)

        def _apply_filter(self, sig):
            return sig

        def analyze(self):
            result = self.filtered + 1
            self.history = self.history + [result]
            return result
""")

NO_TARGET_CLASS = textwrap.dedent("""\
    class Foo:
        pass
""")


@pytest.fixture
def simple_source(tmp_path):
    p = tmp_path / "simple.py"
    p.write_text(SIMPLE_CLASS)
    return str(p)


@pytest.fixture
def stateful_source(tmp_path):
    p = tmp_path / "stateful.py"
    p.write_text(STATEFUL_CLASS)
    return str(p)


@pytest.fixture
def no_target_source(tmp_path):
    p = tmp_path / "no_target.py"
    p.write_text(NO_TARGET_CLASS)
    return str(p)


# ---------------------------------------------------------------------------
# Tests: self.* reads and writes
# ---------------------------------------------------------------------------


class TestSelfAccessTracking:
    @pytest.mark.asyncio
    async def test_init_writes(self, simple_source):
        dfg = await extract_data_flow(simple_source, "SimpleProcessor")
        init = next(m for m in dfg.methods if m.name == "__init__")
        assert "raw" in init.writes
        assert "processed" in init.writes

    @pytest.mark.asyncio
    async def test_process_reads_and_writes(self, simple_source):
        dfg = await extract_data_flow(simple_source, "SimpleProcessor")
        proc = next(m for m in dfg.methods if m.name == "process")
        assert "raw" in proc.reads
        assert "processed" in proc.writes

    @pytest.mark.asyncio
    async def test_all_attributes_index(self, simple_source):
        dfg = await extract_data_flow(simple_source, "SimpleProcessor")
        assert "raw" in dfg.all_attributes
        assert "processed" in dfg.all_attributes


# ---------------------------------------------------------------------------
# Tests: config branches
# ---------------------------------------------------------------------------


class TestConfigBranches:
    @pytest.mark.asyncio
    async def test_config_branch_detected(self, stateful_source):
        dfg = await extract_data_flow(stateful_source, "StatefulPipeline")
        assert len(dfg.config_branches) == 1
        cb = dfg.config_branches[0]
        assert cb.config_attr == "apply_filter"
        assert cb.method == "preprocess"

    @pytest.mark.asyncio
    async def test_config_branch_reads_writes(self, stateful_source):
        dfg = await extract_data_flow(stateful_source, "StatefulPipeline")
        cb = dfg.config_branches[0]
        assert "filtered" in cb.writes


# ---------------------------------------------------------------------------
# Tests: init chain
# ---------------------------------------------------------------------------


class TestInitChain:
    @pytest.mark.asyncio
    async def test_init_chain_order(self, stateful_source):
        dfg = await extract_data_flow(stateful_source, "StatefulPipeline")
        # Should capture sequential self.X = ... in __init__
        assert "raw" in dfg.init_chain
        assert "filtered" in dfg.init_chain
        assert "options" in dfg.init_chain


# ---------------------------------------------------------------------------
# Tests: cross-window attrs
# ---------------------------------------------------------------------------


class TestCrossWindowAttrs:
    @pytest.mark.asyncio
    async def test_cross_window_detection(self, stateful_source):
        dfg = await extract_data_flow(stateful_source, "StatefulPipeline")
        # 'history' is read and written in analyze() (non-init) -> cross-window
        assert "history" in dfg.cross_window_attrs

    @pytest.mark.asyncio
    async def test_filtered_is_cross_window(self, stateful_source):
        dfg = await extract_data_flow(stateful_source, "StatefulPipeline")
        # 'filtered' is written in preprocess and read in analyze
        assert "filtered" in dfg.cross_window_attrs


# ---------------------------------------------------------------------------
# Tests: internal call graph
# ---------------------------------------------------------------------------


class TestInternalCallGraph:
    @pytest.mark.asyncio
    async def test_internal_calls_detected(self, stateful_source):
        dfg = await extract_data_flow(stateful_source, "StatefulPipeline")
        assert "preprocess" in dfg.internal_call_graph
        assert "_apply_filter" in dfg.internal_call_graph["preprocess"]


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_missing_class(self, no_target_source):
        with pytest.raises(ValueError, match="not found"):
            await extract_data_flow(no_target_source, "MissingClass")

    @pytest.mark.asyncio
    async def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            await extract_data_flow("/nonexistent/path.py", "Foo")


# ---------------------------------------------------------------------------
# Tests: basic metadata
# ---------------------------------------------------------------------------


class TestBasicMetadata:
    @pytest.mark.asyncio
    async def test_class_name(self, simple_source):
        dfg = await extract_data_flow(simple_source, "SimpleProcessor")
        assert dfg.class_name == "SimpleProcessor"

    @pytest.mark.asyncio
    async def test_method_count(self, simple_source):
        dfg = await extract_data_flow(simple_source, "SimpleProcessor")
        assert len(dfg.methods) == 2  # __init__ and process

    @pytest.mark.asyncio
    async def test_method_params(self, simple_source):
        dfg = await extract_data_flow(simple_source, "SimpleProcessor")
        init = next(m for m in dfg.methods if m.name == "__init__")
        assert "data" in init.params
