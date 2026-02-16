"""Tests for the synthesizer assembler (Phase 1 — Round 3)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

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
from ageom.synthesizer.assembler import Assembler, AssemblyError, sanitize_name
from ageom.synthesizer.compiler import SkeletonCompiler
from ageom.synthesizer.models import AssemblyUnit, SkeletonFile
from ageom.synthesizer.pipeline import assemble_and_check
from ageom.synthesizer.toposort import toposort_nodes
from ageom.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
    PDGNode,
    Prover,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_cdg() -> CDGExport:
    """3-node CDG: root -> sort_step -> search_step with one data-flow edge."""
    nodes = [
        AlgorithmicNode(
            node_id="root",
            name="Sort and Search",
            description="Sort an array then binary search",
            concept_type=ConceptType.DIVIDE_AND_CONQUER,
            status=NodeStatus.DECOMPOSED,
            children=["sort_step", "search_step"],
            depth=0,
            type_signature="list Nat -> Nat -> Option Nat",
        ),
        AlgorithmicNode(
            node_id="sort_step",
            parent_id="root",
            name="Heapsort",
            description="Sort array using heapsort",
            concept_type=ConceptType.SORTING,
            status=NodeStatus.ATOMIC,
            matched_primitive="heapsort",
            type_signature="list Nat -> list Nat",
            inputs=[IOSpec(name="arr", type_desc="list Nat")],
            outputs=[IOSpec(name="sorted", type_desc="list Nat")],
            depth=1,
        ),
        AlgorithmicNode(
            node_id="search_step",
            parent_id="root",
            name="Binary Search",
            description="Search sorted array for target",
            concept_type=ConceptType.SEARCHING,
            status=NodeStatus.ATOMIC,
            matched_primitive="binary_search",
            type_signature="list Nat -> Nat -> Option Nat",
            inputs=[
                IOSpec(name="arr", type_desc="sorted list Nat"),
                IOSpec(name="target", type_desc="Nat"),
            ],
            outputs=[IOSpec(name="index", type_desc="Option Nat")],
            depth=1,
        ),
    ]
    edges = [
        DependencyEdge(
            source_id="sort_step",
            target_id="search_step",
            output_name="sorted",
            input_name="arr",
            source_type="list Nat",
            target_type="sorted list Nat",
        ),
    ]
    return CDGExport(
        nodes=nodes,
        edges=edges,
        metadata={"goal": "Sort and search", "paradigm": "divide_and_conquer"},
    )


@pytest.fixture
def sample_cdg_with_glue(sample_cdg: CDGExport) -> CDGExport:
    """Same CDG but the edge has requires_glue=True."""
    edges = [
        DependencyEdge(
            source_id="sort_step",
            target_id="search_step",
            output_name="sorted",
            input_name="arr",
            source_type="list Nat",
            target_type="sorted list Nat",
            requires_glue=True,
        ),
    ]
    return CDGExport(
        nodes=sample_cdg.nodes,
        edges=edges,
        metadata=sample_cdg.metadata,
    )


def _make_match_result(node_id: str, decl_name: str, type_sig: str) -> MatchResult:
    decl = Declaration(
        name=decl_name,
        type_signature=type_sig,
        prover=Prover.LEAN4,
    )
    candidate = CandidateMatch(declaration=decl, score=0.95, retrieval_method="embedding")
    vr = VerificationResult(candidate=candidate, verified=True, proof_term=f"@{decl_name}")
    return MatchResult(
        pdg_node=PDGNode(predicate_id=node_id, statement=type_sig),
        verified_match=vr,
        all_candidates=[candidate],
        all_verifications=[vr],
    )


@pytest.fixture
def sample_match_results() -> list[MatchResult]:
    return [
        _make_match_result("sort_step", "List.mergeSort", "list Nat -> list Nat"),
        _make_match_result("search_step", "List.binSearch", "list Nat -> Nat -> Option Nat"),
    ]


@pytest.fixture
def mock_env_success():
    """Mock ProofEnvironment that always succeeds."""
    env = AsyncMock()
    env.prover_name = "lean4"
    env._run = AsyncMock(
        return_value=CompilerFeedback(raw_output="ok", errors=[], warnings=[])
    )
    env.check_term = AsyncMock(return_value=(True, "ok"))
    env.check_proof = AsyncMock(return_value=(True, "ok"))
    env.close = AsyncMock()
    return env


@pytest.fixture
def mock_env_failure():
    """Mock ProofEnvironment that always fails compilation."""
    env = AsyncMock()
    env.prover_name = "lean4"
    env._run = AsyncMock(
        return_value=CompilerFeedback(
            raw_output="error: type mismatch",
            errors=["error: type mismatch"],
        )
    )
    env.check_term = AsyncMock(return_value=(False, "error: type mismatch"))
    env.check_proof = AsyncMock(return_value=(False, "error: type mismatch"))
    env.close = AsyncMock()
    return env


# ---------------------------------------------------------------------------
# TestToposort
# ---------------------------------------------------------------------------


class TestToposort:
    def test_linear_chain(self):
        nodes = [
            AlgorithmicNode(node_id="A", name="A", description="", concept_type=ConceptType.CUSTOM),
            AlgorithmicNode(node_id="B", name="B", description="", concept_type=ConceptType.CUSTOM),
            AlgorithmicNode(node_id="C", name="C", description="", concept_type=ConceptType.CUSTOM),
        ]
        edges = [
            DependencyEdge(
                source_id="A", target_id="B",
                output_name="x", input_name="x",
                source_type="T", target_type="T",
            ),
            DependencyEdge(
                source_id="B", target_id="C",
                output_name="y", input_name="y",
                source_type="T", target_type="T",
            ),
        ]
        result = toposort_nodes(nodes, edges)
        assert result.index("A") < result.index("B") < result.index("C")

    def test_diamond(self):
        nodes = [
            AlgorithmicNode(node_id="A", name="A", description="", concept_type=ConceptType.CUSTOM),
            AlgorithmicNode(node_id="B", name="B", description="", concept_type=ConceptType.CUSTOM),
            AlgorithmicNode(node_id="C", name="C", description="", concept_type=ConceptType.CUSTOM),
            AlgorithmicNode(node_id="D", name="D", description="", concept_type=ConceptType.CUSTOM),
        ]
        edges = [
            DependencyEdge(source_id="A", target_id="B", output_name="x", input_name="x", source_type="T", target_type="T"),
            DependencyEdge(source_id="A", target_id="C", output_name="x", input_name="x", source_type="T", target_type="T"),
            DependencyEdge(source_id="B", target_id="D", output_name="x", input_name="x", source_type="T", target_type="T"),
            DependencyEdge(source_id="C", target_id="D", output_name="x", input_name="x", source_type="T", target_type="T"),
        ]
        result = toposort_nodes(nodes, edges)
        assert result.index("A") < result.index("B")
        assert result.index("A") < result.index("C")
        assert result.index("B") < result.index("D")
        assert result.index("C") < result.index("D")

    def test_single_node(self):
        nodes = [
            AlgorithmicNode(node_id="X", name="X", description="", concept_type=ConceptType.CUSTOM),
        ]
        result = toposort_nodes(nodes, [])
        assert result == ["X"]

    def test_cycle_raises(self):
        nodes = [
            AlgorithmicNode(node_id="A", name="A", description="", concept_type=ConceptType.CUSTOM),
            AlgorithmicNode(node_id="B", name="B", description="", concept_type=ConceptType.CUSTOM),
        ]
        edges = [
            DependencyEdge(source_id="A", target_id="B", output_name="x", input_name="x", source_type="T", target_type="T"),
            DependencyEdge(source_id="B", target_id="A", output_name="x", input_name="x", source_type="T", target_type="T"),
        ]
        with pytest.raises(ValueError, match="Cycle detected"):
            toposort_nodes(nodes, edges)


# ---------------------------------------------------------------------------
# TestAssembler
# ---------------------------------------------------------------------------


class TestAssembler:
    def test_assemble_lean4_skeleton(self, sample_cdg, sample_match_results):
        assembler = Assembler(Prover.LEAN4)
        skeleton = assembler.assemble(sample_cdg, sample_match_results)

        assert skeleton.prover == "lean4"
        assert "import Mathlib" in skeleton.source_code
        assert "#check @List.mergeSort" in skeleton.source_code
        assert "#check @List.binSearch" in skeleton.source_code
        assert "-- Node: Heapsort" in skeleton.source_code
        assert "-- Node: Binary Search" in skeleton.source_code
        assert len(skeleton.units) == 2

    def test_assemble_coq_skeleton(self, sample_cdg, sample_match_results):
        assembler = Assembler(Prover.COQ)
        skeleton = assembler.assemble(sample_cdg, sample_match_results)

        assert skeleton.prover == "coq"
        assert "Check @List.mergeSort." in skeleton.source_code
        assert "Definition" in skeleton.source_code
        assert "(* Node: Heapsort" in skeleton.source_code

    def test_missing_match_raises(self, sample_cdg):
        """Atomic leaf without a match should raise AssemblyError."""
        # Only provide match for sort_step, not search_step
        partial_matches = [
            _make_match_result("sort_step", "List.mergeSort", "list Nat -> list Nat"),
        ]
        assembler = Assembler(Prover.LEAN4)
        with pytest.raises(AssemblyError, match="Missing verified matches"):
            assembler.assemble(sample_cdg, partial_matches)

    def test_sorry_count(self, sample_cdg, sample_match_results):
        """Sorry count should match number of decomposed nodes with type signatures."""
        assembler = Assembler(Prover.LEAN4)
        skeleton = assembler.assemble(sample_cdg, sample_match_results)

        # root node is decomposed and has a type_signature -> 1 sorry
        assert skeleton.sorry_count == 1
        assert "sorry" in skeleton.source_code

    def test_sorry_count_coq(self, sample_cdg, sample_match_results):
        assembler = Assembler(Prover.COQ)
        skeleton = assembler.assemble(sample_cdg, sample_match_results)

        assert skeleton.sorry_count == 1
        assert "Admitted." in skeleton.source_code

    def test_glue_edges_flagged(self, sample_cdg_with_glue, sample_match_results):
        assembler = Assembler(Prover.LEAN4)
        skeleton = assembler.assemble(sample_cdg_with_glue, sample_match_results)

        # search_step should be flagged as requires_glue
        search_unit = next(u for u in skeleton.units if u.node_id == "search_step")
        assert search_unit.requires_glue is True

        # sort_step should NOT be flagged
        sort_unit = next(u for u in skeleton.units if u.node_id == "sort_step")
        assert sort_unit.requires_glue is False

    def test_sanitize_name(self):
        assert sanitize_name("Merge Sort") == "merge_sort"
        assert sanitize_name("binary-search") == "binary_search"
        assert sanitize_name("123abc") == "n_123abc"
        assert sanitize_name("hello___world") == "hello_world"
        assert sanitize_name("") == "unnamed"
        assert sanitize_name("  spaces  ") == "spaces"
        assert sanitize_name("CamelCase") == "camelcase"

    def test_units_in_topological_order(self, sample_cdg, sample_match_results):
        """Units should be ordered so sort_step comes before search_step."""
        assembler = Assembler(Prover.LEAN4)
        skeleton = assembler.assemble(sample_cdg, sample_match_results)

        unit_ids = [u.node_id for u in skeleton.units]
        assert unit_ids.index("sort_step") < unit_ids.index("search_step")

    def test_metadata_preserved(self, sample_cdg, sample_match_results):
        assembler = Assembler(Prover.LEAN4)
        skeleton = assembler.assemble(sample_cdg, sample_match_results)

        assert skeleton.metadata["goal"] == "Sort and search"
        assert "timestamp" in skeleton.metadata


# ---------------------------------------------------------------------------
# TestSkeletonCompiler
# ---------------------------------------------------------------------------


class TestSkeletonCompiler:
    @pytest.mark.asyncio
    async def test_compile_success(self, mock_env_success, sample_cdg, sample_match_results):
        assembler = Assembler(Prover.LEAN4)
        skeleton = assembler.assemble(sample_cdg, sample_match_results)

        compiler = SkeletonCompiler(mock_env_success)
        result = await compiler.compile(skeleton)

        assert result.compiled_ok is True
        assert result.feedback is not None
        assert result.feedback.success is True

    @pytest.mark.asyncio
    async def test_compile_failure(self, mock_env_failure, sample_cdg, sample_match_results):
        assembler = Assembler(Prover.LEAN4)
        skeleton = assembler.assemble(sample_cdg, sample_match_results)

        compiler = SkeletonCompiler(mock_env_failure)
        result = await compiler.compile(skeleton)

        assert result.compiled_ok is False
        assert result.feedback is not None
        assert len(result.feedback.errors) > 0

    @pytest.mark.asyncio
    async def test_check_unit_isolation(self, mock_env_success, sample_match_results):
        unit = AssemblyUnit(
            node_id="sort_step",
            name="Heapsort",
            declaration_name="List.mergeSort",
            type_signature="list Nat -> list Nat",
        )
        compiler = SkeletonCompiler(mock_env_success)
        feedback = await compiler.check_unit(unit)

        assert feedback.success is True


# ---------------------------------------------------------------------------
# TestPipeline
# ---------------------------------------------------------------------------


class TestPipeline:
    @pytest.mark.asyncio
    async def test_assemble_and_check_happy_path(
        self, sample_cdg, sample_match_results, mock_env_success
    ):
        result = await assemble_and_check(sample_cdg, sample_match_results, mock_env_success)

        assert result.compiled_ok is True
        assert result.skeleton.prover == "lean4"
        assert len(result.skeleton.units) == 2

    @pytest.mark.asyncio
    async def test_assemble_and_check_compile_failure(
        self, sample_cdg, sample_match_results, mock_env_failure
    ):
        result = await assemble_and_check(sample_cdg, sample_match_results, mock_env_failure)

        assert result.compiled_ok is False
        assert result.feedback is not None
        assert len(result.feedback.errors) > 0


# ---------------------------------------------------------------------------
# TestMatchResultSerialization
# ---------------------------------------------------------------------------


class TestMatchResultSerialization:
    def test_roundtrip(self):
        mr = _make_match_result("node1", "Nat.add_comm", "Nat -> Nat -> Nat")
        data = mr.to_dict()
        restored = MatchResult.from_dict(data)

        assert restored.pdg_node.predicate_id == "node1"
        assert restored.success is True
        assert restored.verified_match is not None
        assert restored.verified_match.candidate.declaration.name == "Nat.add_comm"

    def test_roundtrip_no_verified_match(self):
        mr = MatchResult(
            pdg_node=PDGNode(predicate_id="p1", statement="T"),
        )
        data = mr.to_dict()
        restored = MatchResult.from_dict(data)

        assert restored.verified_match is None
        assert restored.success is False

    def test_json_serializable(self):
        mr = _make_match_result("node1", "Nat.add_comm", "Nat -> Nat -> Nat")
        data = mr.to_dict()
        # Should be JSON-serializable
        text = json.dumps(data)
        loaded = json.loads(text)
        restored = MatchResult.from_dict(loaded)
        assert restored.success is True


# ---------------------------------------------------------------------------
# TestCLIParserAcceptsAssemble
# ---------------------------------------------------------------------------


class TestCLIParserAcceptsAssemble:
    def test_assemble_args(self, tmp_path):
        """Parser should accept 'assemble' with required args."""
        from unittest.mock import patch

        from ageom.cli import main

        cdg = tmp_path / "cdg.json"
        cdg.write_text('{"nodes":[], "edges":[]}')
        matches = tmp_path / "matches.json"
        matches.write_text("[]")

        with patch("sys.argv", ["ageom", "assemble", str(cdg), str(matches)]):
            with patch("ageom.cli._cmd_assemble") as mock_cmd:
                # _cmd_assemble is async so we mock the coroutine
                mock_cmd.return_value = None
                # asyncio.run will be called on it, so we need to mock at a higher level
                with patch("asyncio.run") as mock_run:
                    main()
                    mock_run.assert_called_once()
