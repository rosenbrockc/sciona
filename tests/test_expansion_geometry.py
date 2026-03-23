"""Tests for the Geometry expansion rules and runtime atoms."""

import numpy as np
import pytest

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules.geometry import GeometryExpansionRuleSet
from sciona.expansion_atoms.runtime_geometry import (
    detect_collinear_points, analyze_numeric_precision,
    detect_duplicate_points, validate_convexity,
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

def _geometry_cdg():
    return _cdg(
        [_node("src", "Source"), _node("pp", "Preprocess Points", ConceptType.GEOMETRY),
         _node("co", "Construct", ConceptType.GEOMETRY), _node("vi", "Verify Invariant", ConceptType.GEOMETRY),
         _node("out", "Output")],
        [_edge("src", "pp"), _edge("pp", "co"), _edge("co", "vi"), _edge("vi", "out")],
    )


class TestDetectCollinearPoints:
    def test_non_collinear(self):
        pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=float)
        n_col, frac = detect_collinear_points(pts)
        assert isinstance(frac, float)

    def test_all_collinear(self):
        pts = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
        n_col, frac = detect_collinear_points(pts)
        assert frac > 0.5

    def test_too_few_points(self):
        pts = np.array([[0, 0], [1, 1]], dtype=float)
        n_col, frac = detect_collinear_points(pts)
        assert n_col == 0


class TestAnalyzeNumericPrecision:
    def test_well_conditioned(self):
        pts = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        cond, risky = analyze_numeric_precision(pts)
        assert not risky

    def test_single_point(self):
        cond, risky = analyze_numeric_precision(np.array([[0, 0]], dtype=float))
        assert not risky


class TestDetectDuplicatePoints:
    def test_no_duplicates(self):
        pts = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        n_dup, frac = detect_duplicate_points(pts)
        assert n_dup == 0

    def test_has_duplicates(self):
        pts = np.array([[0, 0], [0, 0], [1, 0]], dtype=float)
        n_dup, frac = detect_duplicate_points(pts)
        assert n_dup >= 1

    def test_single_point(self):
        n_dup, frac = detect_duplicate_points(np.array([[0, 0]], dtype=float))
        assert n_dup == 0


class TestValidateConvexity:
    def test_convex_square(self):
        pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        v, is_convex = validate_convexity(pts)
        assert v == 0
        assert is_convex

    def test_non_convex(self):
        pts = np.array([[0, 0], [2, 0], [1, 0.5], [2, 1], [0, 1]], dtype=float)
        v, is_convex = validate_convexity(pts)
        assert v > 0
        assert not is_convex

    def test_triangle(self):
        v, is_convex = validate_convexity(np.array([[0, 0], [1, 0], [0.5, 1]], dtype=float))
        assert is_convex


class TestGeometryRules:
    def _get_rules(self):
        return {r.name: r for r in GeometryExpansionRuleSet().rules()}

    def test_collinearity_detection_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_collinearity_detection_before_preprocess"], _geometry_cdg())
        assert not result.is_failure
        assert "detect_collinear_points" in {n.matched_primitive for n in result.unwrap().nodes if n.matched_primitive}

    def test_duplicate_detection_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_duplicate_point_detection_before_preprocess"], _geometry_cdg())
        assert not result.is_failure

    def test_precision_analysis_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_numeric_precision_analysis_after_construct"], _geometry_cdg())
        assert not result.is_failure

    def test_convexity_validation_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_convexity_validation_after_verify"], _geometry_cdg())
        assert not result.is_failure


class TestGeometryDiagnostics:
    def test_diagnose_collinearity(self):
        diags = GeometryExpansionRuleSet().diagnose(_geometry_cdg(), ExpansionContext(intermediates={"collinear_fraction": 0.3}))
        assert "insert_collinearity_detection_before_preprocess" in {d.rule_name for d in diags}

    def test_low_collinearity_no_trigger(self):
        diags = GeometryExpansionRuleSet().diagnose(_geometry_cdg(), ExpansionContext(intermediates={"collinear_fraction": 0.01}))
        assert not [d for d in diags if d.rule_name == "insert_collinearity_detection_before_preprocess"]

    def test_diagnose_duplicates(self):
        diags = GeometryExpansionRuleSet().diagnose(_geometry_cdg(), ExpansionContext(intermediates={"duplicate_fraction": 0.05}))
        assert "insert_duplicate_point_detection_before_preprocess" in {d.rule_name for d in diags}

    def test_diagnose_precision(self):
        diags = GeometryExpansionRuleSet().diagnose(_geometry_cdg(), ExpansionContext(intermediates={"geometric_condition_number": 1e12}))
        assert "insert_numeric_precision_analysis_after_construct" in {d.rule_name for d in diags}

    def test_diagnose_convexity(self):
        diags = GeometryExpansionRuleSet().diagnose(_geometry_cdg(), ExpansionContext(intermediates={"convexity_violations": 2}))
        assert "insert_convexity_validation_after_verify" in {d.rule_name for d in diags}

    def test_no_data_returns_nothing(self):
        assert GeometryExpansionRuleSet().diagnose(_geometry_cdg(), ExpansionContext()) == []


class TestGeometryIntegration:
    def test_full_expansion(self):
        result = ExpansionEngine([GeometryExpansionRuleSet()]).expand(
            _geometry_cdg(), ExpansionContext(intermediates={"collinear_fraction": 0.5, "convexity_violations": 3}))
        assert result.expanded
