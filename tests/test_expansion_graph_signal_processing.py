"""Tests for the Graph Signal Processing expansion rules and runtime atoms."""

import numpy as np
import pytest

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules.graph_signal_processing import GraphSignalProcessingExpansionRuleSet
from sciona.expansion_atoms.runtime_graph_signal_processing import (
    validate_graph_connectivity, check_laplacian_symmetry,
    analyze_spectral_gap, validate_filter_response,
)


def _node(nid, name, concept=ConceptType.CUSTOM, primitive=None):
    return AlgorithmicNode(
        node_id=nid, name=name, description=name, concept_type=concept,
        status=NodeStatus.ATOMIC, matched_primitive=primitive,
        inputs=[IOSpec(name="in", type_desc="ndarray")],
        outputs=[IOSpec(name="out", type_desc="ndarray")],
        type_signature=f"{name} -> r",
    )

def _edge(src, tgt):
    return DependencyEdge(source_id=src, target_id=tgt, output_name="out", input_name="in", source_type="ndarray", target_type="ndarray")

def _cdg(nodes, edges):
    return CDGExport(nodes=nodes, edges=edges, metadata={})

def _gsp_cdg():
    return _cdg(
        [_node("src", "Source"), _node("bg", "Build Graph", ConceptType.GRAPH_SIGNAL_PROCESSING),
         _node("cl", "Compute Laplacian", ConceptType.GRAPH_SIGNAL_PROCESSING),
         _node("gft", "GFT", ConceptType.GRAPH_SIGNAL_PROCESSING),
         _node("gf", "Graph Filter/Diffuse", ConceptType.GRAPH_SIGNAL_PROCESSING),
         _node("out", "Output")],
        [_edge("src", "bg"), _edge("bg", "cl"), _edge("cl", "gft"), _edge("gft", "gf"), _edge("gf", "out")],
    )


class TestValidateGraphConnectivity:
    def test_connected(self):
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        n_comp, connected = validate_graph_connectivity(A)
        assert connected
        assert n_comp == 1

    def test_disconnected(self):
        A = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=float)
        n_comp, connected = validate_graph_connectivity(A)
        assert not connected
        assert n_comp == 2

    def test_empty(self):
        n_comp, connected = validate_graph_connectivity(np.array([]).reshape(0, 0))
        assert connected


class TestCheckLaplacianSymmetry:
    def test_symmetric(self):
        L = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=float)
        asym, sym = check_laplacian_symmetry(L)
        assert sym
        assert asym == 0.0

    def test_asymmetric(self):
        L = np.array([[2, -1], [-0.5, 1]], dtype=float)
        asym, sym = check_laplacian_symmetry(L)
        assert not sym

    def test_empty(self):
        asym, sym = check_laplacian_symmetry(np.array([]).reshape(0, 0))
        assert sym


class TestAnalyzeSpectralGap:
    def test_well_connected(self):
        eigs = np.array([0.0, 0.5, 1.5, 2.0])
        gap, well = analyze_spectral_gap(eigs)
        assert well
        assert gap == 0.5

    def test_near_disconnected(self):
        eigs = np.array([0.0, 0.001, 1.5, 2.0])
        gap, well = analyze_spectral_gap(eigs)
        assert not well

    def test_single_eigenvalue(self):
        gap, well = analyze_spectral_gap(np.array([0.0]))
        assert not well


class TestValidateFilterResponse:
    def test_stable(self):
        h = np.array([1.0, 0.5, 0.1, 0.01])
        eigs = np.array([0.0, 0.5, 1.0, 2.0])
        gain, stable = validate_filter_response(h, eigs)
        assert stable
        assert gain == 1.0

    def test_unstable(self):
        h = np.array([1.0, 200.0, 0.1])
        eigs = np.array([0.0, 0.5, 1.0])
        gain, stable = validate_filter_response(h, eigs)
        assert not stable

    def test_empty(self):
        gain, stable = validate_filter_response(np.array([]), np.array([]))
        assert stable


class TestGSPRules:
    def _get_rules(self):
        return {r.name: r for r in GraphSignalProcessingExpansionRuleSet().rules()}

    def test_connectivity_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_connectivity_validation_after_build_graph"], _gsp_cdg())
        assert not result.is_failure
        assert "validate_graph_connectivity" in {n.matched_primitive for n in result.unwrap().nodes if n.matched_primitive}

    def test_symmetry_check_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_laplacian_symmetry_check_after_compute_laplacian"], _gsp_cdg())
        assert not result.is_failure

    def test_spectral_gap_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_spectral_gap_analysis_after_gft"], _gsp_cdg())
        assert not result.is_failure

    def test_filter_response_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_filter_response_validation_after_graph_filter"], _gsp_cdg())
        assert not result.is_failure


class TestGSPDiagnostics:
    def test_diagnose_disconnected(self):
        diags = GraphSignalProcessingExpansionRuleSet().diagnose(_gsp_cdg(), ExpansionContext(intermediates={"n_graph_components": 3}))
        assert "insert_connectivity_validation_after_build_graph" in {d.rule_name for d in diags}

    def test_connected_no_trigger(self):
        diags = GraphSignalProcessingExpansionRuleSet().diagnose(_gsp_cdg(), ExpansionContext(intermediates={"n_graph_components": 1}))
        assert not [d for d in diags if d.rule_name == "insert_connectivity_validation_after_build_graph"]

    def test_diagnose_asymmetry(self):
        diags = GraphSignalProcessingExpansionRuleSet().diagnose(_gsp_cdg(), ExpansionContext(intermediates={"laplacian_max_asymmetry": 0.01}))
        assert "insert_laplacian_symmetry_check_after_compute_laplacian" in {d.rule_name for d in diags}

    def test_diagnose_spectral_gap(self):
        diags = GraphSignalProcessingExpansionRuleSet().diagnose(_gsp_cdg(), ExpansionContext(intermediates={"spectral_gap": 0.001}))
        assert "insert_spectral_gap_analysis_after_gft" in {d.rule_name for d in diags}

    def test_diagnose_filter_gain(self):
        diags = GraphSignalProcessingExpansionRuleSet().diagnose(_gsp_cdg(), ExpansionContext(intermediates={"max_filter_gain": 500.0}))
        assert "insert_filter_response_validation_after_graph_filter" in {d.rule_name for d in diags}

    def test_no_data_returns_nothing(self):
        assert GraphSignalProcessingExpansionRuleSet().diagnose(_gsp_cdg(), ExpansionContext()) == []


class TestGSPIntegration:
    def test_full_expansion(self):
        result = ExpansionEngine([GraphSignalProcessingExpansionRuleSet()]).expand(
            _gsp_cdg(), ExpansionContext(intermediates={"n_graph_components": 2, "spectral_gap": 0.005}))
        assert result.expanded
