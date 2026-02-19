"""Tests for C++ tree-sitter extraction (ageom.ingester.treesitter_extractor)."""

from __future__ import annotations

import textwrap

import pytest

from ageom.ingester.base_extractor import SourceLanguage
from ageom.ingester.treesitter_extractor import TreeSitterExtractor


# ---------------------------------------------------------------------------
# Embedded Pronto-like C++ source
# ---------------------------------------------------------------------------


RIGID_BODY_STATE_CPP = textwrap.dedent("""\
    class RigidBodyState : public StateBase {
    public:
        Eigen::Vector3d position;
        Eigen::Quaterniond orientation;
        int64_t utime;

        RigidBodyState() : utime(0) {
            position = Eigen::Vector3d::Zero();
        }

        void predict(double dt) {
            this->position += this->velocity * dt;
            utime += static_cast<int64_t>(dt * 1e6);
        }

        Eigen::Vector3d getPosition() const {
            return this->position;
        }

    private:
        Eigen::Vector3d velocity;
    };
""")


SIMPLE_CLASS_CPP = textwrap.dedent("""\
    class SimpleProcessor {
    public:
        double value;

        SimpleProcessor() {
            value = 0.0;
        }

        void update(double input) {
            this->value = input * 2.0;
        }

        double get_value() {
            return this->value;
        }
    };
""")

MULTI_METHOD_CPP = textwrap.dedent("""\
    class Pipeline {
    public:
        double raw;
        double filtered;
        double output;

        void step1(double input) {
            this->raw = input;
            this->filtered = this->raw * 0.5;
        }

        void step2() {
            this->output = this->filtered + 1.0;
        }
    };
""")


FREE_FUNCTIONS_CPP = textwrap.dedent("""\
    double add(double a, double b) {
        return a + b;
    }

    int multiply(int x, int y) {
        return x * y;
    }
""")


@pytest.fixture
def extractor():
    return TreeSitterExtractor(SourceLanguage.CPP)


@pytest.fixture
def rigid_body_source(tmp_path):
    p = tmp_path / "rigid_body.cpp"
    p.write_text(RIGID_BODY_STATE_CPP)
    return str(p)


@pytest.fixture
def simple_source(tmp_path):
    p = tmp_path / "simple.cpp"
    p.write_text(SIMPLE_CLASS_CPP)
    return str(p)


@pytest.fixture
def multi_method_source(tmp_path):
    p = tmp_path / "pipeline.cpp"
    p.write_text(MULTI_METHOD_CPP)
    return str(p)


@pytest.fixture
def free_functions_source(tmp_path):
    p = tmp_path / "funcs.cpp"
    p.write_text(FREE_FUNCTIONS_CPP)
    return str(p)


# ---------------------------------------------------------------------------
# Tests: class name extraction
# ---------------------------------------------------------------------------


class TestCppClassName:
    @pytest.mark.asyncio
    async def test_class_name(self, extractor, rigid_body_source):
        dfg = await extractor.extract_class(rigid_body_source, "RigidBodyState")
        assert dfg.class_name == "RigidBodyState"

    @pytest.mark.asyncio
    async def test_source_language(self, extractor, rigid_body_source):
        dfg = await extractor.extract_class(rigid_body_source, "RigidBodyState")
        assert dfg.source_language == "cpp"

    @pytest.mark.asyncio
    async def test_source_code_captured(self, extractor, rigid_body_source):
        dfg = await extractor.extract_class(rigid_body_source, "RigidBodyState")
        assert "RigidBodyState" in dfg.source_code


# ---------------------------------------------------------------------------
# Tests: field extraction
# ---------------------------------------------------------------------------


class TestCppFields:
    @pytest.mark.asyncio
    async def test_fields_in_all_attributes(self, extractor, rigid_body_source):
        dfg = await extractor.extract_class(rigid_body_source, "RigidBodyState")
        # Fields should appear in all_attributes via method accesses
        assert "position" in dfg.all_attributes
        assert "utime" in dfg.all_attributes


# ---------------------------------------------------------------------------
# Tests: method extraction
# ---------------------------------------------------------------------------


class TestCppMethods:
    @pytest.mark.asyncio
    async def test_method_count(self, extractor, rigid_body_source):
        dfg = await extractor.extract_class(rigid_body_source, "RigidBodyState")
        # Constructor (__init__), predict, getPosition
        assert len(dfg.methods) == 3

    @pytest.mark.asyncio
    async def test_constructor_renamed(self, extractor, rigid_body_source):
        dfg = await extractor.extract_class(rigid_body_source, "RigidBodyState")
        names = [m.name for m in dfg.methods]
        assert "__init__" in names

    @pytest.mark.asyncio
    async def test_method_params(self, extractor, rigid_body_source):
        dfg = await extractor.extract_class(rigid_body_source, "RigidBodyState")
        predict = next(m for m in dfg.methods if m.name == "predict")
        assert "dt" in predict.params

    @pytest.mark.asyncio
    async def test_method_return_type(self, extractor, rigid_body_source):
        dfg = await extractor.extract_class(rigid_body_source, "RigidBodyState")
        get_pos = next(m for m in dfg.methods if m.name == "getPosition")
        assert "Vector3d" in get_pos.return_type or "Eigen" in get_pos.return_type

    @pytest.mark.asyncio
    async def test_method_source_code(self, extractor, simple_source):
        dfg = await extractor.extract_class(simple_source, "SimpleProcessor")
        update = next(m for m in dfg.methods if m.name == "update")
        assert "this->value" in update.source_code


# ---------------------------------------------------------------------------
# Tests: this->field reads/writes
# ---------------------------------------------------------------------------


class TestCppThisAccess:
    @pytest.mark.asyncio
    async def test_this_write(self, extractor, rigid_body_source):
        dfg = await extractor.extract_class(rigid_body_source, "RigidBodyState")
        predict = next(m for m in dfg.methods if m.name == "predict")
        assert "position" in predict.writes

    @pytest.mark.asyncio
    async def test_this_read(self, extractor, rigid_body_source):
        dfg = await extractor.extract_class(rigid_body_source, "RigidBodyState")
        get_pos = next(m for m in dfg.methods if m.name == "getPosition")
        assert "position" in get_pos.reads

    @pytest.mark.asyncio
    async def test_this_read_in_expression(self, extractor, rigid_body_source):
        dfg = await extractor.extract_class(rigid_body_source, "RigidBodyState")
        predict = next(m for m in dfg.methods if m.name == "predict")
        assert "velocity" in predict.reads


# ---------------------------------------------------------------------------
# Tests: bare member access
# ---------------------------------------------------------------------------


class TestCppBareMemberAccess:
    @pytest.mark.asyncio
    async def test_bare_member_write(self, extractor, rigid_body_source):
        dfg = await extractor.extract_class(rigid_body_source, "RigidBodyState")
        predict = next(m for m in dfg.methods if m.name == "predict")
        # utime += ... is a bare member write
        assert "utime" in predict.writes

    @pytest.mark.asyncio
    async def test_bare_member_in_constructor(self, extractor, rigid_body_source):
        dfg = await extractor.extract_class(rigid_body_source, "RigidBodyState")
        init = next(m for m in dfg.methods if m.name == "__init__")
        # position = ... in constructor body is a bare member write
        assert "position" in init.writes


# ---------------------------------------------------------------------------
# Tests: constructor as init
# ---------------------------------------------------------------------------


class TestCppConstructorInit:
    @pytest.mark.asyncio
    async def test_init_chain(self, extractor, rigid_body_source):
        dfg = await extractor.extract_class(rigid_body_source, "RigidBodyState")
        # Constructor initializes utime via initializer list and position in body
        assert "utime" in dfg.init_chain
        assert "position" in dfg.init_chain

    @pytest.mark.asyncio
    async def test_init_chain_from_initializer_list(self, extractor, rigid_body_source):
        dfg = await extractor.extract_class(rigid_body_source, "RigidBodyState")
        init = next(m for m in dfg.methods if m.name == "__init__")
        assert "utime" in init.writes


# ---------------------------------------------------------------------------
# Tests: inheritance
# ---------------------------------------------------------------------------


class TestCppInheritance:
    @pytest.mark.asyncio
    async def test_base_classes_not_in_dfg_yet(self, extractor, rigid_body_source):
        """Base classes are extracted by the visitor but not stored in DFG (future)."""
        dfg = await extractor.extract_class(rigid_body_source, "RigidBodyState")
        # The DFG is still created successfully
        assert dfg.class_name == "RigidBodyState"


# ---------------------------------------------------------------------------
# Tests: multiple methods with data flow
# ---------------------------------------------------------------------------


class TestCppMultiMethod:
    @pytest.mark.asyncio
    async def test_cross_method_data_flow(self, extractor, multi_method_source):
        dfg = await extractor.extract_class(multi_method_source, "Pipeline")
        step1 = next(m for m in dfg.methods if m.name == "step1")
        step2 = next(m for m in dfg.methods if m.name == "step2")
        assert "raw" in step1.writes
        assert "filtered" in step1.writes
        assert "filtered" in step2.reads
        assert "output" in step2.writes


# ---------------------------------------------------------------------------
# Tests: procedural
# ---------------------------------------------------------------------------


class TestCppProcedural:
    @pytest.mark.asyncio
    async def test_free_functions(self, extractor, free_functions_source):
        dfg = await extractor.extract_procedural(free_functions_source)
        names = [m.name for m in dfg.methods]
        assert "add" in names
        assert "multiply" in names

    @pytest.mark.asyncio
    async def test_procedural_params(self, extractor, free_functions_source):
        dfg = await extractor.extract_procedural(free_functions_source)
        add = next(m for m in dfg.methods if m.name == "add")
        assert "a" in add.params
        assert "b" in add.params

    @pytest.mark.asyncio
    async def test_procedural_return_type(self, extractor, free_functions_source):
        dfg = await extractor.extract_procedural(free_functions_source)
        add = next(m for m in dfg.methods if m.name == "add")
        assert add.return_type == "double"

    @pytest.mark.asyncio
    async def test_procedural_source_language(self, extractor, free_functions_source):
        dfg = await extractor.extract_procedural(free_functions_source)
        assert dfg.source_language == "cpp"


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestCppErrors:
    @pytest.mark.asyncio
    async def test_missing_file(self, extractor):
        with pytest.raises(FileNotFoundError):
            await extractor.extract_class("/nonexistent/file.cpp", "Foo")

    @pytest.mark.asyncio
    async def test_missing_class(self, extractor, simple_source):
        with pytest.raises(ValueError, match="not found"):
            await extractor.extract_class(simple_source, "NonExistent")

    @pytest.mark.asyncio
    async def test_python_language_rejected(self):
        with pytest.raises(ValueError, match="PythonASTExtractor"):
            TreeSitterExtractor(SourceLanguage.PYTHON)

    @pytest.mark.asyncio
    async def test_procedural_missing_file(self, extractor):
        with pytest.raises(FileNotFoundError):
            await extractor.extract_procedural("/nonexistent/file.cpp")
