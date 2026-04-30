"""Tests for AOT codegen integration in the assembler.

Verifies that when atoms have symbolic metadata, the assembler emits
inline NumPy code instead of _sciona_call, and the generated skeleton
has zero sympy imports.
"""

from __future__ import annotations

import pytest
import sympy as sp

from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.ghost.dimensions import DIMENSIONLESS
from sciona.ghost.registry import REGISTRY
from sciona.ghost.symbolic import SymbolicExpression, serialize_expr
from sciona.synthesizer.assembler import Assembler
from sciona.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
    PDGNode,
    VerificationResult,
)


def _cleanup_registry(*names: str):
    for n in names:
        REGISTRY.pop(n, None)


def _register_symbolic_atom(name: str, expr, variables: dict):
    """Register a fake symbolic atom."""

    def _witness(**kwargs):
        pass

    def _impl(**kwargs):
        pass

    sym = SymbolicExpression(
        srepr_str=serialize_expr(expr),
        variables=variables,
        dim_map={k: DIMENSIONLESS for k in variables if variables[k] == "input"},
    )
    REGISTRY[name] = {
        "impl": _impl,
        "witness": _witness,
        "doc": f"test: {name}",
        "signature": {},
        "heavy_signature": {},
        "module": "test",
        "name": name,
        "dim_signature": {},
        "symbolic": sym,
    }


def _make_match_result(node_id: str, decl_name: str) -> MatchResult:
    decl = Declaration(
        name=decl_name,
        type_signature="float -> float",
        docstring="test",
        prover="python",
    )
    candidate = CandidateMatch(declaration=decl, score=1.0, retrieval_method="test")
    verification = VerificationResult(candidate=candidate, verified=True, compiler_output="ok")
    return MatchResult(
        pdg_node=PDGNode(predicate_id=node_id, statement="test"),
        verified_match=verification,
        all_candidates=[candidate],
        all_verifications=[verification],
    )


class TestAOTInAssembler:
    def test_symbolic_atom_emits_inline_numpy(self):
        """When an atom has symbolic metadata, the assembler emits inline NumPy."""
        x = sp.Symbol("x")
        expr = x**2 + 1
        _register_symbolic_atom("test.square_plus_one", expr, {"x": "input"})

        try:
            nodes = [
                AlgorithmicNode(
                    node_id="root",
                    name="Pipeline",
                    description="test",
                    concept_type=ConceptType.CUSTOM,
                    status=NodeStatus.DECOMPOSED,
                    children=["leaf1"],
                    inputs=[IOSpec(name="x", type_desc="float")],
                    outputs=[IOSpec(name="y", type_desc="float")],
                ),
                AlgorithmicNode(
                    node_id="leaf1",
                    name="Square Plus One",
                    description="test",
                    concept_type=ConceptType.ARITHMETIC,
                    status=NodeStatus.ATOMIC,
                    matched_primitive="test.square_plus_one",
                    inputs=[IOSpec(name="x", type_desc="float")],
                    outputs=[IOSpec(name="y", type_desc="float")],
                ),
            ]
            edges = [
                DependencyEdge(
                    source_id="leaf1",
                    target_id="root",
                    output_name="y",
                    input_name="y",
                    source_type="float",
                    target_type="float",
                ),
            ]

            cdg = CDGExport(
                goal="test",
                nodes=nodes,
                edges=edges,
            )

            match_results = [_make_match_result("leaf1", "test.square_plus_one")]

            assembler = Assembler("python")
            skeleton = assembler.assemble(cdg, match_results)

            # The generated source should contain inline NumPy, not _sciona_call
            source = skeleton.source_code
            assert "AOT-compiled from SymPy" in source
            assert "import sympy" not in source

        finally:
            _cleanup_registry("test.square_plus_one")

    def test_non_symbolic_atom_uses_sciona_call(self):
        """When an atom has no symbolic metadata, standard _sciona_call is used."""
        def _witness():
            pass

        def _impl():
            pass

        REGISTRY["test.plain_atom"] = {
            "impl": _impl,
            "witness": _witness,
            "doc": "test",
            "signature": {},
            "heavy_signature": {},
            "module": "test",
            "name": "test.plain_atom",
            "dim_signature": {},
            "symbolic": None,
        }

        try:
            nodes = [
                AlgorithmicNode(
                    node_id="root",
                    name="Pipeline",
                    description="test",
                    concept_type=ConceptType.CUSTOM,
                    status=NodeStatus.DECOMPOSED,
                    children=["leaf1"],
                    inputs=[IOSpec(name="x", type_desc="float")],
                    outputs=[IOSpec(name="y", type_desc="float")],
                ),
                AlgorithmicNode(
                    node_id="leaf1",
                    name="Plain Atom",
                    description="test",
                    concept_type=ConceptType.ARITHMETIC,
                    status=NodeStatus.ATOMIC,
                    matched_primitive="test.plain_atom",
                    inputs=[IOSpec(name="x", type_desc="float")],
                    outputs=[IOSpec(name="y", type_desc="float")],
                ),
            ]
            edges = [
                DependencyEdge(
                    source_id="leaf1",
                    target_id="root",
                    output_name="y",
                    input_name="y",
                    source_type="float",
                    target_type="float",
                ),
            ]

            cdg = CDGExport(goal="test", nodes=nodes, edges=edges)
            match_results = [_make_match_result("leaf1", "test.plain_atom")]

            assembler = Assembler("python")
            skeleton = assembler.assemble(cdg, match_results)

            source = skeleton.source_code
            assert "_sciona_call" in source
            assert "AOT-compiled" not in source

        finally:
            _cleanup_registry("test.plain_atom")
