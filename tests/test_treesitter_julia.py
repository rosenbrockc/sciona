"""Tests for Julia tree-sitter extraction (sciona.ingester.treesitter_extractor)."""

from __future__ import annotations

import textwrap

import pytest

from sciona.ingester.base_extractor import SourceLanguage
from sciona.ingester.treesitter_extractor import TreeSitterExtractor

# ---------------------------------------------------------------------------
# Embedded Tempo.jl-like Julia source
# ---------------------------------------------------------------------------


EPOCH_JL = textwrap.dedent("""\
    struct Epoch{S,T}
        dur::T
        origin::S
    end

    function offset(e::Epoch, dt)
        return Epoch(e.dur + dt, e.origin)
    end

    function get_dur(e::Epoch)
        return e.dur
    end

    function set_origin!(e::Epoch, new_origin)
        e.origin = new_origin
    end
""")


SIMPLE_STRUCT_JL = textwrap.dedent("""\
    struct Point
        x::Float64
        y::Float64
    end

    function norm(p::Point)
        return sqrt(p.x^2 + p.y^2)
    end

    function translate(p::Point, dx, dy)
        return Point(p.x + dx, p.y + dy)
    end
""")


MULTI_STRUCT_JL = textwrap.dedent("""\
    struct Foo
        a::Int
    end

    struct Bar
        b::Float64
    end

    function process_foo(f::Foo)
        return f.a * 2
    end

    function process_bar(b::Bar)
        return b.b + 1.0
    end
""")


FREE_FUNCTIONS_JL = textwrap.dedent("""\
    struct Ignored
        x::Int
    end

    function add(a, b)
        return a + b
    end

    function multiply(x, y)
        return x * y
    end

    function process_ignored(i::Ignored)
        return i.x
    end
""")


ORACLE_DISPATCH_JL = textwrap.dedent("""\
    struct SamplerState
        theta::Float64
    end

    function AbstractMCMC.step(rng::AbstractRNG, model::DensityModel, state::SamplerState)
        lp = model.logdensity(state.theta)
        g = model.gradient(state.theta)
        return lp, g
    end
""")


@pytest.fixture
def extractor():
    return TreeSitterExtractor(SourceLanguage.JULIA)


@pytest.fixture
def epoch_source(tmp_path):
    p = tmp_path / "epoch.jl"
    p.write_text(EPOCH_JL)
    return str(p)


@pytest.fixture
def simple_source(tmp_path):
    p = tmp_path / "point.jl"
    p.write_text(SIMPLE_STRUCT_JL)
    return str(p)


@pytest.fixture
def multi_struct_source(tmp_path):
    p = tmp_path / "multi.jl"
    p.write_text(MULTI_STRUCT_JL)
    return str(p)


@pytest.fixture
def free_functions_source(tmp_path):
    p = tmp_path / "funcs.jl"
    p.write_text(FREE_FUNCTIONS_JL)
    return str(p)


@pytest.fixture
def oracle_dispatch_source(tmp_path):
    p = tmp_path / "oracle_step.jl"
    p.write_text(ORACLE_DISPATCH_JL)
    return str(p)


# ---------------------------------------------------------------------------
# Tests: struct name and type params
# ---------------------------------------------------------------------------


class TestJuliaStructName:
    @pytest.mark.asyncio
    async def test_struct_name(self, extractor, epoch_source):
        dfg = await extractor.extract_class(epoch_source, "Epoch")
        assert dfg.class_name == "Epoch"

    @pytest.mark.asyncio
    async def test_source_language(self, extractor, epoch_source):
        dfg = await extractor.extract_class(epoch_source, "Epoch")
        assert dfg.source_language == "julia"

    @pytest.mark.asyncio
    async def test_source_code_captured(self, extractor, epoch_source):
        dfg = await extractor.extract_class(epoch_source, "Epoch")
        assert "Epoch" in dfg.source_code


# ---------------------------------------------------------------------------
# Tests: field extraction
# ---------------------------------------------------------------------------


class TestJuliaFields:
    @pytest.mark.asyncio
    async def test_field_access_in_attributes(self, extractor, epoch_source):
        dfg = await extractor.extract_class(epoch_source, "Epoch")
        assert "dur" in dfg.all_attributes
        assert "origin" in dfg.all_attributes


# ---------------------------------------------------------------------------
# Tests: function association
# ---------------------------------------------------------------------------


class TestJuliaFunctionAssociation:
    @pytest.mark.asyncio
    async def test_methods_associated(self, extractor, epoch_source):
        dfg = await extractor.extract_class(epoch_source, "Epoch")
        names = [m.name for m in dfg.methods]
        assert "offset" in names
        assert "get_dur" in names
        assert "set_origin!" in names

    @pytest.mark.asyncio
    async def test_method_count(self, extractor, epoch_source):
        dfg = await extractor.extract_class(epoch_source, "Epoch")
        assert len(dfg.methods) == 3

    @pytest.mark.asyncio
    async def test_params_exclude_self(self, extractor, epoch_source):
        dfg = await extractor.extract_class(epoch_source, "Epoch")
        offset = next(m for m in dfg.methods if m.name == "offset")
        assert "dt" in offset.params
        # The self-like 'e' param should be excluded
        assert "e" not in offset.params

    @pytest.mark.asyncio
    async def test_multi_struct_association(self, extractor, multi_struct_source):
        dfg_foo = await extractor.extract_class(multi_struct_source, "Foo")
        dfg_bar = await extractor.extract_class(multi_struct_source, "Bar")
        foo_methods = [m.name for m in dfg_foo.methods]
        bar_methods = [m.name for m in dfg_bar.methods]
        assert "process_foo" in foo_methods
        assert "process_bar" not in foo_methods
        assert "process_bar" in bar_methods
        assert "process_foo" not in bar_methods


# ---------------------------------------------------------------------------
# Tests: field access tracking
# ---------------------------------------------------------------------------


class TestJuliaFieldAccess:
    @pytest.mark.asyncio
    async def test_field_read(self, extractor, epoch_source):
        dfg = await extractor.extract_class(epoch_source, "Epoch")
        get_dur = next(m for m in dfg.methods if m.name == "get_dur")
        assert "dur" in get_dur.reads

    @pytest.mark.asyncio
    async def test_field_read_in_expression(self, extractor, epoch_source):
        dfg = await extractor.extract_class(epoch_source, "Epoch")
        offset = next(m for m in dfg.methods if m.name == "offset")
        assert "dur" in offset.reads
        assert "origin" in offset.reads

    @pytest.mark.asyncio
    async def test_field_write(self, extractor, epoch_source):
        dfg = await extractor.extract_class(epoch_source, "Epoch")
        set_origin = next(m for m in dfg.methods if m.name == "set_origin!")
        assert "origin" in set_origin.writes


# ---------------------------------------------------------------------------
# Tests: return types
# ---------------------------------------------------------------------------


class TestJuliaReturnTypes:
    @pytest.mark.asyncio
    async def test_no_return_type(self, extractor, epoch_source):
        dfg = await extractor.extract_class(epoch_source, "Epoch")
        offset = next(m for m in dfg.methods if m.name == "offset")
        # No explicit return type in source
        assert offset.return_type == ""


# ---------------------------------------------------------------------------
# Tests: simple struct
# ---------------------------------------------------------------------------


class TestJuliaSimpleStruct:
    @pytest.mark.asyncio
    async def test_point_methods(self, extractor, simple_source):
        dfg = await extractor.extract_class(simple_source, "Point")
        names = [m.name for m in dfg.methods]
        assert "norm" in names
        assert "translate" in names

    @pytest.mark.asyncio
    async def test_point_field_reads(self, extractor, simple_source):
        dfg = await extractor.extract_class(simple_source, "Point")
        norm = next(m for m in dfg.methods if m.name == "norm")
        assert "x" in norm.reads
        assert "y" in norm.reads

    @pytest.mark.asyncio
    async def test_translate_params(self, extractor, simple_source):
        dfg = await extractor.extract_class(simple_source, "Point")
        translate = next(m for m in dfg.methods if m.name == "translate")
        assert "dx" in translate.params
        assert "dy" in translate.params


# ---------------------------------------------------------------------------
# Tests: procedural
# ---------------------------------------------------------------------------


class TestJuliaProcedural:
    @pytest.mark.asyncio
    async def test_free_functions_only(self, extractor, free_functions_source):
        dfg = await extractor.extract_procedural(free_functions_source)
        names = [m.name for m in dfg.methods]
        assert "add" in names
        assert "multiply" in names
        # Functions associated with structs should be excluded
        assert "process_ignored" not in names

    @pytest.mark.asyncio
    async def test_procedural_params(self, extractor, free_functions_source):
        dfg = await extractor.extract_procedural(free_functions_source)
        add = next(m for m in dfg.methods if m.name == "add")
        assert "a" in add.params
        assert "b" in add.params

    @pytest.mark.asyncio
    async def test_procedural_source_language(self, extractor, free_functions_source):
        dfg = await extractor.extract_procedural(free_functions_source)
        assert dfg.source_language == "julia"

    @pytest.mark.asyncio
    async def test_oracle_interface_dispatch_emits_oracle_subgraph(
        self, extractor, oracle_dispatch_source
    ):
        dfg = await extractor.extract_procedural(oracle_dispatch_source)

        step = next(m for m in dfg.methods if m.name == "step")
        assert step.is_oracle is True
        assert any(e.caller == "step" for e in dfg.oracle_edges)
        assert any(
            "oracle_subgraph::JuliaDispatch::step::model" in e.oracle_ref
            for e in dfg.oracle_edges
        )

        # state -> oracle routing edges
        assert any(
            e.source_id.startswith("state:")
            and "oracle_subgraph::JuliaDispatch::step::model" in e.target_id
            for e in dfg.inferred_edges
        )
        # oracle -> caller outputs
        assert any(
            "oracle_subgraph::JuliaDispatch::step::model" in e.source_id
            and e.target_id == "step"
            and e.output_name == "log_prob"
            for e in dfg.inferred_edges
        )
        assert any(
            "oracle_subgraph::JuliaDispatch::step::model" in e.source_id
            and e.target_id == "step"
            and e.output_name == "gradient"
            for e in dfg.inferred_edges
        )


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestJuliaErrors:
    @pytest.mark.asyncio
    async def test_missing_file(self, extractor):
        with pytest.raises(FileNotFoundError):
            await extractor.extract_class("/nonexistent/file.jl", "Foo")

    @pytest.mark.asyncio
    async def test_missing_struct(self, extractor, simple_source):
        with pytest.raises(ValueError, match="not found"):
            await extractor.extract_class(simple_source, "NonExistent")

    @pytest.mark.asyncio
    async def test_procedural_missing_file(self, extractor):
        with pytest.raises(FileNotFoundError):
            await extractor.extract_procedural("/nonexistent/file.jl")
