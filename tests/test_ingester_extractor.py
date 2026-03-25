"""Tests for Phase 1 AST extraction (sciona.ingester.extractor)."""

from __future__ import annotations

import textwrap

import pytest

from sciona.ingester.extractor import extract_data_flow

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

SEMANTIC_CLASS = textwrap.dedent("""\
    class SemanticEstimator:
        def __init__(
            self,
            base_estimator,
            cv: int = 3,
            *,
            method: str = "sigmoid",
            n_jobs=None,
            **kwargs,
        ):
            self.base_estimator = base_estimator
            self.cv = cv
            self.method = method
            self.n_jobs = n_jobs
            self.extra = kwargs
            self.calibrators = []
            self.is_fitted_ = False

        def fit(self, X, y, *, sample_weight=None):
            self.calibrators = [
                self._fit_single(X, y, sample_weight=sample_weight)
            ]
            self.is_fitted_ = True
            return self

        def predict(self, X):
            return self._predict_impl(X)

        def expose_state(self):
            return self.calibrators

        def get_metadata_routing(self):
            return {"sample_weight": True}

        def __sklearn_tags__(self):
            return {"requires_fit": True}

        def passthrough(self, *args, **kwargs):
            return getattr(self, "missing_handler", None)(*args, **kwargs)

        def _fit_single(self, X, y, sample_weight=None):
            return (X, y, sample_weight)

        def _predict_impl(self, X):
            return self.calibrators[0]
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


@pytest.fixture
def semantic_source(tmp_path):
    p = tmp_path / "semantic_estimator.py"
    p.write_text(SEMANTIC_CLASS)
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


class TestSemanticFacts:
    @pytest.mark.asyncio
    async def test_exact_signature_facts(self, semantic_source):
        dfg = await extract_data_flow(semantic_source, "SemanticEstimator")
        init = next(m for m in dfg.methods if m.name == "__init__")

        assert init.params == [
            "base_estimator",
            "cv",
            "method",
            "n_jobs",
            "kwargs",
        ]
        assert [(param.name, param.kind) for param in init.signature] == [
            ("base_estimator", "positional_or_keyword"),
            ("cv", "positional_or_keyword"),
            ("method", "keyword_only"),
            ("n_jobs", "keyword_only"),
            ("kwargs", "kwarg"),
        ]
        by_name = {param.name: param for param in init.signature}
        assert by_name["cv"].annotation == "int"
        assert by_name["cv"].default_expression == "3"
        assert by_name["cv"].has_default is True
        assert by_name["method"].annotation == "str"
        assert by_name["method"].default_expression == "'sigmoid'"

    @pytest.mark.asyncio
    async def test_return_fact_classification(self, semantic_source):
        dfg = await extract_data_flow(semantic_source, "SemanticEstimator")
        fit = next(m for m in dfg.methods if m.name == "fit")
        predict = next(m for m in dfg.methods if m.name == "predict")
        expose_state = next(m for m in dfg.methods if m.name == "expose_state")

        assert [fact.kind for fact in fit.return_facts] == ["self"]
        assert [fact.kind for fact in predict.return_facts] == ["call_result"]
        assert predict.return_facts[0].referenced_callees == ["self._predict_impl"]
        assert [fact.kind for fact in expose_state.return_facts] == ["attribute"]
        assert expose_state.return_facts[0].referenced_attrs == ["calibrators"]

    @pytest.mark.asyncio
    async def test_role_and_attribute_inventory(self, semantic_source):
        dfg = await extract_data_flow(semantic_source, "SemanticEstimator")
        by_name = {method.name: method for method in dfg.methods}

        assert by_name["fit"].semantic_role == "fit_or_update"
        assert by_name["predict"].semantic_role == "predict_or_transform"
        assert by_name["get_metadata_routing"].semantic_role == "query_or_metadata"
        assert by_name["__sklearn_tags__"].semantic_role == "query_or_metadata"
        assert by_name["_fit_single"].semantic_role == "helper"
        assert "base_estimator" in dfg.config_attributes
        assert "cv" in dfg.config_attributes
        assert "calibrators" in dfg.fitted_attributes
        assert "is_fitted_" in dfg.fitted_attributes

        attr_by_name = {fact.attr_name: fact for fact in dfg.attribute_facts}
        assert attr_by_name["base_estimator"].is_config is True
        assert attr_by_name["calibrators"].is_fitted is True
        assert "fit" in attr_by_name["calibrators"].write_methods
        assert "predict" in attr_by_name["calibrators"].read_methods

    @pytest.mark.asyncio
    async def test_unknown_fact_emission(self, semantic_source):
        dfg = await extract_data_flow(semantic_source, "SemanticEstimator")
        passthrough = next(m for m in dfg.methods if m.name == "passthrough")
        reasons = {fact.reason for fact in passthrough.unknown_facts}

        assert "dynamic_getattr" in reasons
        assert "variadic_forwarding" in reasons
        assert {"dynamic_getattr", "variadic_forwarding"} <= {
            fact.reason for fact in dfg.semantic_unknowns
        }
