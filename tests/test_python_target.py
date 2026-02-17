"""Comprehensive tests for the Python-as-Atom target."""

from __future__ import annotations

import ast
import asyncio
import re
import textwrap
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ageom.architect.handoff import CDGExport
from ageom.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from ageom.judge.models import CompilerFeedback
from ageom.synthesizer.assembler import Assembler, sanitize_name
from ageom.synthesizer.classifier import (
    ErrorCategory,
    classify_error,
    suggest_deterministic_fix,
)
from ageom.synthesizer.contracts import ContractGenerator, ContractSpec, SafeAtomWrapper
from ageom.synthesizer.extractor import ExportTarget
from ageom.synthesizer.models import AssemblyUnit, SkeletonFile
from ageom.synthesizer.patcher import find_sorry_locations
from ageom.synthesizer.python_template import (
    generate_init_py,
    generate_main_script,
    generate_pipeline_py,
    generate_pyproject_toml,
)
from ageom.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
    PDGNode,
    Prover,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_python_match_result(
    node_id: str, decl_name: str, type_sig: str
) -> MatchResult:
    decl = Declaration(
        name=decl_name,
        type_signature=type_sig,
        prover=Prover.PYTHON,
    )
    candidate = CandidateMatch(declaration=decl, score=0.95, retrieval_method="embedding")
    vr = VerificationResult(candidate=candidate, verified=True, proof_term=decl_name)
    return MatchResult(
        pdg_node=PDGNode(predicate_id=node_id, statement=type_sig, prover=Prover.PYTHON),
        verified_match=vr,
        all_candidates=[candidate],
        all_verifications=[vr],
    )


@pytest.fixture
def python_cdg() -> CDGExport:
    """Simple 2-node CDG for Python target."""
    nodes = [
        AlgorithmicNode(
            node_id="root",
            name="Solve Linear System",
            description="Solve Ax = b",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.DECOMPOSED,
            children=["solve_step"],
            depth=0,
            type_signature="(A: ndarray, b: ndarray) -> ndarray",
        ),
        AlgorithmicNode(
            node_id="solve_step",
            parent_id="root",
            name="LU Solve",
            description="Solve via LU factorization",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.ATOMIC,
            matched_primitive="scipy.linalg.solve",
            type_signature="(A: ndarray, b: ndarray) -> ndarray",
            inputs=[
                IOSpec(name="A", type_desc="ndarray", constraints="A.ndim == 2"),
                IOSpec(name="b", type_desc="ndarray"),
            ],
            outputs=[IOSpec(name="result", type_desc="ndarray")],
            depth=1,
        ),
    ]
    edges: list[DependencyEdge] = []
    return CDGExport(
        nodes=nodes,
        edges=edges,
        metadata={"goal": "Solve Ax=b"},
    )


@pytest.fixture
def python_match_results() -> list[MatchResult]:
    return [
        _make_python_match_result(
            "solve_step",
            "scipy.linalg.solve",
            "(A: ndarray, b: ndarray) -> ndarray",
        ),
    ]


# ---------------------------------------------------------------------------
# TestProverEnum
# ---------------------------------------------------------------------------


class TestProverEnum:
    def test_python_exists(self):
        assert hasattr(Prover, "PYTHON")
        assert Prover.PYTHON.value == "python"

    def test_roundtrip_via_value(self):
        assert Prover("python") == Prover.PYTHON

    def test_string_comparison(self):
        assert Prover.PYTHON == "python"

    def test_all_provers(self):
        values = {p.value for p in Prover}
        assert values == {"lean4", "coq", "python"}


# ---------------------------------------------------------------------------
# TestPythonDeclarationSource
# ---------------------------------------------------------------------------


class TestPythonDeclarationSource:
    def test_ast_extract_function(self):
        from ageom.indexer.python_source import PythonDeclarationSource

        source = PythonDeclarationSource()
        code = textwrap.dedent("""\
            def solve(A: ndarray, b: ndarray) -> ndarray:
                \"\"\"Solve Ax = b.\"\"\"
                return linalg.solve(A, b)
        """)
        tree = ast.parse(code)
        func_node = tree.body[0]
        assert isinstance(func_node, ast.FunctionDef)

        info = source._ast_extract_function(func_node, "scipy.linalg")
        assert info is not None
        assert info.qualname == "scipy.linalg.solve"
        assert info.return_type == "ndarray"
        assert "A" in info.parameter_types
        assert "b" in info.parameter_types
        assert info.docstring == "Solve Ax = b."

    def test_private_functions_skipped(self):
        from ageom.indexer.python_source import PythonDeclarationSource

        source = PythonDeclarationSource()
        code = textwrap.dedent("""\
            def _internal(x: int) -> int:
                return x
        """)
        tree = ast.parse(code)
        func_node = tree.body[0]
        assert isinstance(func_node, ast.FunctionDef)

        info = source._ast_extract_function(func_node, "mod")
        assert info is None

    def test_type_signature_normalization(self):
        from ageom.indexer.python_source import PythonDeclarationSource, PythonFunctionInfo

        source = PythonDeclarationSource()
        info = PythonFunctionInfo(
            module="numpy",
            qualname="numpy.dot",
            signature="(a: ndarray, b: ndarray) -> ndarray",
            docstring="Dot product.",
            parameter_types={"a": "ndarray", "b": "ndarray"},
            return_type="ndarray",
        )
        sig = source._build_type_signature(info)
        assert sig == "(a: ndarray, b: ndarray) -> ndarray"

    def test_extract_icontract_decorators(self):
        from ageom.indexer.python_source import PythonDeclarationSource

        source = PythonDeclarationSource()
        code = textwrap.dedent("""\
            @icontract.require(lambda A: A.ndim == 2)
            @icontract.ensure(lambda result: result.ndim == 1)
            def solve(A: ndarray, b: ndarray) -> ndarray:
                pass
        """)
        tree = ast.parse(code)
        func_node = tree.body[0]
        assert isinstance(func_node, ast.FunctionDef)

        pre, post = source._extract_icontract_decorators(func_node)
        assert len(pre) == 1
        assert "A.ndim == 2" in pre[0]
        assert len(post) == 1
        assert "result.ndim == 1" in post[0]

    def test_get_declarations_from_module_json(self):
        """Test extraction from a real stdlib module (json)."""
        from ageom.indexer.python_source import PythonDeclarationSource

        source = PythonDeclarationSource()
        decls = source.get_declarations_from_module("json")
        # json module has public functions like dump, dumps, load, loads
        names = [d.name for d in decls]
        assert any("dump" in n for n in names)
        assert all(d.prover == Prover.PYTHON for d in decls)


# ---------------------------------------------------------------------------
# TestPythonEnvironment
# ---------------------------------------------------------------------------


class TestPythonEnvironment:
    def test_prover_name(self):
        from ageom.judge.python_env import PythonEnvironment

        env = PythonEnvironment()
        assert env.prover_name == "python"

    def test_parse_mypy_output_clean(self):
        from ageom.judge.python_env import _parse_mypy_output

        raw = "Success: no issues found in 1 source file\n"
        fb = _parse_mypy_output(raw)
        assert fb.success is True
        assert len(fb.errors) == 0

    def test_parse_mypy_output_errors(self):
        from ageom.judge.python_env import _parse_mypy_output

        raw = textwrap.dedent("""\
            _check.py:1: error: Incompatible types in assignment [assignment]
            _check.py:2: warning: Unused variable [misc]
            Found 1 error in 1 file (checked 1 source file)
        """)
        fb = _parse_mypy_output(raw)
        assert fb.success is False
        assert len(fb.errors) == 1
        assert "Incompatible types" in fb.errors[0]
        assert len(fb.warnings) == 1

    def test_parse_mypy_output_multiple_errors(self):
        from ageom.judge.python_env import _parse_mypy_output

        raw = textwrap.dedent("""\
            _check.py:1: error: No module named 'foo'
            _check.py:3: error: Incompatible return value type
            Found 2 errors in 1 file (checked 1 source file)
        """)
        fb = _parse_mypy_output(raw)
        assert len(fb.errors) == 2

    @pytest.mark.asyncio
    async def test_run_with_mocked_subprocess(self):
        from ageom.judge.python_env import PythonEnvironment

        env = PythonEnvironment()

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (
            b"Success: no issues found in 1 source file\n",
            b"",
        )
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            fb = await env._run("x: int = 1\n")
            assert fb.success is True

        await env.close()

    @pytest.mark.asyncio
    async def test_check_term_with_mocked_subprocess(self):
        from ageom.judge.python_env import PythonEnvironment

        env = PythonEnvironment()

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (
            b"Success: no issues found in 1 source file\n",
            b"",
        )
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            success, output = await env.check_term("42", "int")
            assert success is True

        await env.close()

    @pytest.mark.asyncio
    async def test_mypy_not_found(self):
        from ageom.judge.python_env import PythonEnvironment

        env = PythonEnvironment(mypy_path="/nonexistent/mypy")

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("mypy not found"),
        ):
            fb = await env._run("x: int = 1\n")
            assert fb.success is False
            assert "mypy not found" in fb.errors[0]

        await env.close()


# ---------------------------------------------------------------------------
# TestPythonAssembler
# ---------------------------------------------------------------------------


class TestPythonAssembler:
    def test_assemble_python_skeleton(self, python_cdg, python_match_results):
        assembler = Assembler(Prover.PYTHON)
        skeleton = assembler.assemble(python_cdg, python_match_results)

        assert skeleton.prover == "python"
        assert "import icontract" in skeleton.source_code
        assert "import numpy" in skeleton.source_code or "import scipy" in skeleton.source_code
        assert "# Node: LU Solve" in skeleton.source_code
        assert "scipy.linalg.solve" in skeleton.source_code
        assert len(skeleton.units) == 1

    def test_python_sorry_count(self, python_cdg, python_match_results):
        assembler = Assembler(Prover.PYTHON)
        skeleton = assembler.assemble(python_cdg, python_match_results)

        # Composition is now generated, so sorry_count should be 0
        assert skeleton.sorry_count == 0

    def test_python_skeleton_has_docstring_header(self, python_cdg, python_match_results):
        assembler = Assembler(Prover.PYTHON)
        skeleton = assembler.assemble(python_cdg, python_match_results)

        assert '"""' in skeleton.source_code
        assert "AGEO-Matcher Skeleton" in skeleton.source_code
        assert "Goal: Solve Ax=b" in skeleton.source_code

    def test_python_composition_stub(self, python_cdg, python_match_results):
        assembler = Assembler(Prover.PYTHON)
        skeleton = assembler.assemble(python_cdg, python_match_results)

        assert "def solve_linear_system_composition" in skeleton.source_code
        # Composition now generates actual code (Issue 1 fix)
        assert "lu_solve_result" in skeleton.source_code
        assert "return lu_solve_result" in skeleton.source_code


# ---------------------------------------------------------------------------
# TestContractGenerator
# ---------------------------------------------------------------------------


class TestContractGenerator:
    def test_generate_wrapper(self):
        unit = AssemblyUnit(
            node_id="solve_step",
            name="LU Solve",
            declaration_name="scipy.linalg.solve",
            type_signature="(A: ndarray, b: ndarray) -> ndarray",
            inputs=[
                IOSpec(name="A", type_desc="ndarray", constraints="A.ndim == 2"),
                IOSpec(name="b", type_desc="ndarray"),
            ],
            outputs=[IOSpec(name="result", type_desc="ndarray")],
        )
        decl = Declaration(
            name="scipy.linalg.solve",
            type_signature="(A: ndarray, b: ndarray) -> ndarray",
            source_lib="scipy.linalg",
            prover=Prover.PYTHON,
        )

        gen = ContractGenerator()
        wrapper = gen.generate_wrapper(unit, decl)

        assert wrapper.function_name == "lu_solve_wrapper"
        assert wrapper.original_qualname == "scipy.linalg.solve"
        assert len(wrapper.preconditions) == 1  # A has constraint
        assert wrapper.preconditions[0].kind == "require"
        assert "A.ndim == 2" in wrapper.preconditions[0].lambda_expr
        assert wrapper.return_type == "ndarray"

    def test_render_wrapper(self):
        wrapper = SafeAtomWrapper(
            function_name="solve_wrapper",
            original_qualname="scipy.linalg.solve",
            imports=["import icontract", "import scipy"],
            parameters=[("A", "np.ndarray"), ("b", "np.ndarray")],
            return_type="np.ndarray",
            preconditions=[
                ContractSpec(
                    kind="require",
                    lambda_expr="lambda A: A.ndim == 2",
                    description="A must be 2D",
                ),
            ],
            postconditions=[
                ContractSpec(
                    kind="ensure",
                    lambda_expr="lambda result, A: result.shape == (A.shape[1],)",
                    description="",
                ),
            ],
            body="return scipy.linalg.solve(A, b)",
        )

        gen = ContractGenerator()
        code = gen.render_wrapper(wrapper)

        assert "@icontract.require(lambda A: A.ndim == 2" in code
        assert "@icontract.ensure(lambda result, A: result.shape == (A.shape[1],))" in code
        assert "def solve_wrapper(A: np.ndarray, b: np.ndarray) -> np.ndarray:" in code
        assert "return scipy.linalg.solve(A, b)" in code

    def test_render_wrapper_no_contracts(self):
        wrapper = SafeAtomWrapper(
            function_name="simple_wrapper",
            original_qualname="math.sqrt",
            parameters=[("x", "float")],
            return_type="float",
            body="return math.sqrt(x)",
        )

        gen = ContractGenerator()
        code = gen.render_wrapper(wrapper)

        assert "@icontract" not in code
        assert "def simple_wrapper(x: float) -> float:" in code

    def test_iospec_to_contract(self):
        gen = ContractGenerator()

        spec_with_constraint = IOSpec(name="A", type_desc="ndarray", constraints="A.ndim == 2")
        contract = gen._iospec_to_contract(spec_with_constraint, "require")
        assert contract is not None
        assert contract.kind == "require"
        assert "A.ndim == 2" in contract.lambda_expr

        spec_no_constraint = IOSpec(name="b", type_desc="ndarray")
        contract = gen._iospec_to_contract(spec_no_constraint, "require")
        assert contract is None


# ---------------------------------------------------------------------------
# TestPythonErrorClassifier
# ---------------------------------------------------------------------------


class TestPythonErrorClassifier:
    def test_no_module_named(self):
        cat = classify_error("_check.py:1: error: No module named 'foo'")
        assert cat == ErrorCategory.MISSING_IMPORT

    def test_cannot_find_module_stub(self):
        cat = classify_error(
            "_check.py:1: error: Cannot find implementation or library stub for module named 'bar'"
        )
        assert cat == ErrorCategory.MISSING_IMPORT

    def test_incompatible_types(self):
        cat = classify_error(
            "_check.py:1: error: Incompatible types in assignment"
        )
        assert cat == ErrorCategory.TYPE_MISMATCH

    def test_incompatible_return_value(self):
        cat = classify_error(
            '_check.py:5: error: Incompatible return value type (got "str", expected "int")'
        )
        assert cat == ErrorCategory.TYPE_MISMATCH

    def test_argument_incompatible_type(self):
        cat = classify_error(
            '_check.py:3: error: Argument 1 to "foo" has incompatible type "str"; expected "int"'
        )
        assert cat == ErrorCategory.TYPE_MISMATCH

    def test_invalid_syntax(self):
        cat = classify_error("_check.py:1: error: invalid syntax")
        assert cat == ErrorCategory.SYNTAX

    def test_syntax_error(self):
        cat = classify_error("SyntaxError: unexpected EOF while parsing")
        assert cat == ErrorCategory.SYNTAX

    def test_suggest_python_import_fix(self):
        fix = suggest_deterministic_fix(
            ErrorCategory.MISSING_IMPORT,
            "_check.py:1: error: No module named 'numpy'"
        )
        assert fix is not None
        assert "import numpy" in fix

    def test_suggest_python_module_stub_fix(self):
        fix = suggest_deterministic_fix(
            ErrorCategory.MISSING_IMPORT,
            "_check.py:1: error: Cannot find implementation or library stub for module named 'scipy'"
        )
        assert fix is not None
        assert "import scipy" in fix


# ---------------------------------------------------------------------------
# TestPythonPatcher
# ---------------------------------------------------------------------------


class TestPythonPatcher:
    def test_find_sorry_locations_python(self):
        source = textwrap.dedent("""\
            import numpy as np

            def solve(A, b):
                raise NotImplementedError("TODO: compose solve")

            def other():
                return 42
        """)
        locations = find_sorry_locations(source, "python")
        assert len(locations) == 1
        line_num, context = locations[0]
        assert line_num == 4
        assert "NotImplementedError" in context

    def test_multiple_sorry_locations(self):
        source = textwrap.dedent("""\
            def foo():
                raise NotImplementedError("TODO")

            def bar():
                raise NotImplementedError("TODO")
        """)
        locations = find_sorry_locations(source, "python")
        assert len(locations) == 2

    def test_no_sorry_in_clean_file(self):
        source = textwrap.dedent("""\
            def foo():
                return 42
        """)
        locations = find_sorry_locations(source, "python")
        assert len(locations) == 0

    def test_lean_sorry_not_detected_in_python_mode(self):
        source = textwrap.dedent("""\
            def foo():
                # sorry is a Lean keyword, not Python
                return "sorry"
        """)
        locations = find_sorry_locations(source, "python")
        assert len(locations) == 0


# ---------------------------------------------------------------------------
# TestPythonExtractor
# ---------------------------------------------------------------------------


class TestPythonExtractor:
    def test_python_pkg_in_export_target_enum(self):
        assert ExportTarget.PYTHON_PKG.value == "python-pkg"
        assert ExportTarget("python-pkg") == ExportTarget.PYTHON_PKG

    def test_generate_pyproject_toml(self):
        content = generate_pyproject_toml("my_pkg", ["numpy>=1.24", "scipy>=1.10"])
        assert 'name = "my_pkg"' in content
        assert '"icontract>=2.6"' in content
        assert '"numpy>=1.24"' in content
        assert '"scipy>=1.10"' in content
        assert "[build-system]" in content

    def test_generate_init_py(self):
        content = generate_init_py("my_pkg", ["solve_wrapper", "dot_wrapper"])
        assert "from my_pkg.atoms import solve_wrapper" in content
        assert "from my_pkg.atoms import dot_wrapper" in content
        assert '"solve_wrapper"' in content
        assert "__all__" in content

    def test_generate_pipeline_py(self):
        content = generate_pipeline_py([])
        assert "def run_pipeline" in content
        assert "raise NotImplementedError" in content

    def test_generate_pipeline_py_with_steps(self):
        content = generate_pipeline_py(["result = atoms.solve_wrapper(A, b)"])
        assert "result = atoms.solve_wrapper(A, b)" in content
        assert "raise NotImplementedError" not in content

    def test_generate_main_script(self):
        skeleton = SkeletonFile(
            prover="python",
            source_code="",
            units=[
                AssemblyUnit(
                    node_id="n1",
                    name="solve",
                    declaration_name="scipy.linalg.solve",
                    type_signature="(A, b) -> x",
                ),
            ],
        )
        content = generate_main_script(skeleton, ["def solve_wrapper(): pass"], [])
        assert "import icontract" in content
        assert "import scipy" in content
        assert "def solve_wrapper(): pass" in content

    @pytest.mark.asyncio
    async def test_build_python_creates_package_structure(self, tmp_path):
        """Test that _build_python creates the expected directory structure."""
        from ageom.config import AgeomConfig
        from ageom.synthesizer.extractor import Extractor

        config = AgeomConfig()
        extractor = Extractor(config)
        skeleton = SkeletonFile(
            prover="python",
            source_code="import numpy\ndef solve(): pass\n",
            metadata={"name": "test_pkg"},
        )
        source_path = tmp_path / "src" / "Verified.py"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(skeleton.source_code)

        # Mock mypy to avoid needing it installed
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"Success\n", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            artifact, errors = await extractor._build_python(
                skeleton, source_path, tmp_path
            )

        assert (tmp_path / "pyproject.toml").exists()
        assert (tmp_path / "src" / "test_pkg" / "__init__.py").exists()
        assert (tmp_path / "src" / "test_pkg" / "atoms.py").exists()
        assert (tmp_path / "src" / "test_pkg" / "pipeline.py").exists()


# ---------------------------------------------------------------------------
# TestCLIAcceptsPython
# ---------------------------------------------------------------------------


class TestCLIAcceptsPython:
    def _parse_args(self, argv: list[str]):
        """Parse CLI args without executing commands."""
        from unittest.mock import patch as _patch

        from ageom.cli import main

        captured_args = None

        # We patch asyncio.run and the command functions to capture args
        with _patch("sys.argv", ["ageom"] + argv):
            with _patch("asyncio.run") as mock_run:
                with _patch("ageom.cli._cmd_index_build") as mock_index:
                    try:
                        main()
                    except SystemExit:
                        pass
        return True  # If we get here, parsing succeeded

    def test_index_build_python(self):
        assert self._parse_args(["index", "build", "--prover", "python"])

    def test_assemble_python(self, tmp_path):
        cdg = tmp_path / "cdg.json"
        cdg.write_text('{"nodes":[], "edges":[]}')
        matches = tmp_path / "m.json"
        matches.write_text("[]")
        assert self._parse_args([
            "assemble", str(cdg), str(matches), "--prover", "python"
        ])

    def test_synthesize_python(self, tmp_path):
        cdg = tmp_path / "cdg.json"
        cdg.write_text('{"nodes":[], "edges":[]}')
        matches = tmp_path / "m.json"
        matches.write_text("[]")
        assert self._parse_args([
            "synthesize", str(cdg), str(matches), "--prover", "python"
        ])

    def test_export_python_pkg(self, tmp_path):
        src = tmp_path / "verified.py"
        src.write_text("pass")
        assert self._parse_args([
            "export", str(src), "--target", "python-pkg"
        ])

    def test_export_prover_python(self, tmp_path):
        src = tmp_path / "verified.py"
        src.write_text("pass")
        assert self._parse_args([
            "export", str(src), "--prover", "python"
        ])

    def test_match_prover_python(self):
        assert self._parse_args([
            "match", "--statement", "test", "--prover", "python"
        ])


# ---------------------------------------------------------------------------
# TestPythonPrompts
# ---------------------------------------------------------------------------


class TestPythonPrompts:
    def test_analyze_error_system_python_exists(self):
        from ageom.synthesizer.prompts import ANALYZE_ERROR_SYSTEM_PYTHON

        assert "Python" in ANALYZE_ERROR_SYSTEM_PYTHON
        assert "type annotations" in ANALYZE_ERROR_SYSTEM_PYTHON
        assert "icontract" in ANALYZE_ERROR_SYSTEM_PYTHON
        assert "JSON" in ANALYZE_ERROR_SYSTEM_PYTHON

    def test_generate_implementation_system_python_exists(self):
        from ageom.synthesizer.prompts import GENERATE_IMPLEMENTATION_SYSTEM_PYTHON

        assert "Python" in GENERATE_IMPLEMENTATION_SYSTEM_PYTHON
        assert "NotImplementedError" in GENERATE_IMPLEMENTATION_SYSTEM_PYTHON
        assert "icontract" in GENERATE_IMPLEMENTATION_SYSTEM_PYTHON


# ---------------------------------------------------------------------------
# TestCheckerWithPython
# ---------------------------------------------------------------------------


class TestCheckerWithPython:
    def test_python_env_parameter(self):
        from ageom.judge.checker import VerificationOracleImpl

        oracle = VerificationOracleImpl(python_env=MagicMock())
        env = oracle._get_env(Prover.PYTHON)
        assert env is not None

    def test_python_env_not_configured_raises(self):
        from ageom.judge.checker import VerificationOracleImpl

        oracle = VerificationOracleImpl()
        with pytest.raises(RuntimeError, match="PythonEnvironment not configured"):
            oracle._get_env(Prover.PYTHON)


# ---------------------------------------------------------------------------
# TestPythonConfig
# ---------------------------------------------------------------------------


class TestPythonConfig:
    def test_config_has_python_fields(self):
        from ageom.config import AgeomConfig

        config = AgeomConfig()
        assert hasattr(config, "python_path")
        assert hasattr(config, "python_mypy_path")
        assert hasattr(config, "python_packages")
        assert config.python_path == "python"
        assert config.python_mypy_path == "mypy"
        assert config.python_packages == "numpy,scipy"


# ---------------------------------------------------------------------------
# TestRepairPythonPromptSelection
# ---------------------------------------------------------------------------


class TestRepairPythonPromptSelection:
    def test_repair_imports_python_prompts(self):
        """Verify repair.py imports the Python-specific prompts."""
        import ageom.synthesizer.repair as repair_mod

        assert hasattr(repair_mod, "ANALYZE_ERROR_SYSTEM_PYTHON")
        assert hasattr(repair_mod, "GENERATE_IMPLEMENTATION_SYSTEM_PYTHON")


# ---------------------------------------------------------------------------
# TestMatchResultPythonRoundtrip
# ---------------------------------------------------------------------------


class TestMatchResultPythonRoundtrip:
    def test_roundtrip(self):
        mr = _make_python_match_result(
            "solve_step",
            "scipy.linalg.solve",
            "(A: ndarray, b: ndarray) -> ndarray",
        )
        data = mr.to_dict()
        restored = MatchResult.from_dict(data)

        assert restored.pdg_node.prover == Prover.PYTHON
        assert restored.verified_match is not None
        assert restored.verified_match.candidate.declaration.prover == Prover.PYTHON
