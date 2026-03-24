"""Tests for the Dimensionality Reduction expansion rules and runtime atoms."""

import numpy as np
import pytest

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules.dimensionality_reduction import DimensionalityReductionExpansionRuleSet
from sciona.expansion_atoms.runtime_dimensionality_reduction import (
    analyze_explained_variance, detect_crowding,
    check_reconstruction_error, validate_orthogonality,
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

def _dimred_cdg():
    return _cdg(
        [_node("src", "Source"),
         _node("center", "Center/Scale", ConceptType.DIMENSIONALITY_REDUCTION),
         _node("proj", "Project", ConceptType.DIMENSIONALITY_REDUCTION),
         _node("val", "Validate Reconstruction", ConceptType.DIMENSIONALITY_REDUCTION),
         _node("out", "Output")],
        [_edge("src", "center"), _edge("center", "proj"), _edge("proj", "val"),
         _edge("val", "out")],
    )


class TestAnalyzeExplainedVariance:
    def test_sufficient(self):
        eigs = np.array([10.0, 5.0, 1.0])
        ratio, ok = analyze_explained_variance(eigs)
        assert ok
        assert ratio == pytest.approx(1.0)

    def test_empty(self):
        ratio, ok = analyze_explained_variance(np.array([]))
        assert not ok


class TestDetectCrowding:
    def test_trustworthy(self):
        ranks = np.arange(10, dtype=float)
        trust, ok = detect_crowding(ranks, ranks)
        assert ok
        assert trust == pytest.approx(1.0)

    def test_empty(self):
        trust, ok = detect_crowding(np.array([]), np.array([]))
        assert ok


class TestCheckReconstructionError:
    def test_exact(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        err, ok = check_reconstruction_error(X, X)
        assert ok
        assert err == pytest.approx(0.0, abs=1e-15)

    def test_lossy(self):
        X = np.eye(3)
        X_rec = np.zeros((3, 3))
        err, ok = check_reconstruction_error(X, X_rec)
        assert not ok
        assert err > 0.1

    def test_empty(self):
        err, ok = check_reconstruction_error(np.array([]), np.array([]))
        assert ok


class TestValidateOrthogonality:
    def test_orthogonal(self):
        C = np.eye(3)
        off, ok = validate_orthogonality(C)
        assert ok
        assert off == pytest.approx(0.0, abs=1e-15)

    def test_not_orthogonal(self):
        C = np.ones((3, 3))
        off, ok = validate_orthogonality(C)
        assert not ok
        assert off > 1e-6

    def test_empty(self):
        off, ok = validate_orthogonality(np.array([]).reshape(0, 0))
        assert ok


class TestDimensionalityReductionRules:
    def _get_rules(self):
        return {r.name: r for r in DimensionalityReductionExpansionRuleSet().rules()}

    def test_explained_variance_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_explained_variance_after_project"], _dimred_cdg())
        assert not result.is_failure
        assert "analyze_explained_variance" in {n.matched_primitive for n in result.unwrap().nodes if n.matched_primitive}

    def test_crowding_detection_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_crowding_detection_after_project"], _dimred_cdg())
        assert not result.is_failure

    def test_reconstruction_error_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_reconstruction_error_after_validate"], _dimred_cdg())
        assert not result.is_failure

    def test_orthogonality_validation_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_orthogonality_validation_before_project"], _dimred_cdg())
        assert not result.is_failure


class TestDimensionalityReductionDiagnostics:
    def test_diagnose_explained_variance(self):
        diags = DimensionalityReductionExpansionRuleSet().diagnose(_dimred_cdg(), ExpansionContext(intermediates={"cumulative_variance_ratio": 0.7}))
        assert "insert_explained_variance_after_project" in {d.rule_name for d in diags}

    def test_sufficient_variance_no_trigger(self):
        diags = DimensionalityReductionExpansionRuleSet().diagnose(_dimred_cdg(), ExpansionContext(intermediates={"cumulative_variance_ratio": 0.99}))
        assert not [d for d in diags if d.rule_name == "insert_explained_variance_after_project"]

    def test_diagnose_crowding(self):
        diags = DimensionalityReductionExpansionRuleSet().diagnose(_dimred_cdg(), ExpansionContext(intermediates={"trustworthiness": 0.5}))
        assert "insert_crowding_detection_after_project" in {d.rule_name for d in diags}

    def test_diagnose_reconstruction_error(self):
        diags = DimensionalityReductionExpansionRuleSet().diagnose(_dimred_cdg(), ExpansionContext(intermediates={"reconstruction_relative_error": 0.5}))
        assert "insert_reconstruction_error_after_validate" in {d.rule_name for d in diags}

    def test_diagnose_orthogonality(self):
        diags = DimensionalityReductionExpansionRuleSet().diagnose(_dimred_cdg(), ExpansionContext(intermediates={"max_off_diagonal": 0.1}))
        assert "insert_orthogonality_validation_before_project" in {d.rule_name for d in diags}

    def test_no_data_returns_nothing(self):
        assert DimensionalityReductionExpansionRuleSet().diagnose(_dimred_cdg(), ExpansionContext()) == []


class TestDimensionalityReductionIntegration:
    def test_full_expansion(self):
        result = ExpansionEngine([DimensionalityReductionExpansionRuleSet()]).expand(
            _dimred_cdg(), ExpansionContext(intermediates={"cumulative_variance_ratio": 0.7, "reconstruction_relative_error": 0.5}))
        assert result.expanded
