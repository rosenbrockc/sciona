"""Tests for baseline-analysis expansion rules and diagnostics."""

from __future__ import annotations

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.architect.skeletons import get_skeleton, instantiate_skeleton
from sciona.principal.expansion import ExpansionContext
from sciona.principal.expansion_rules import default_rule_sets
from sciona.principal.expansion_rules.baseline_analysis import (
    BaselineAnalysisExpansionRuleSet,
)


def _node(
    nid: str,
    name: str,
    concept: ConceptType = ConceptType.CUSTOM,
    primitive: str | None = None,
    *,
    map_window_size: int = 0,
    map_hop_size: int = 0,
    is_opaque: bool = False,
) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=nid,
        name=name,
        description=name,
        concept_type=concept,
        status=NodeStatus.ATOMIC,
        matched_primitive=primitive,
        inputs=[IOSpec(name="in", type_desc="ndarray")],
        outputs=[IOSpec(name="out", type_desc="ndarray")],
        type_signature=f"{name} -> r",
        map_window_size=map_window_size,
        map_hop_size=map_hop_size,
        is_opaque=is_opaque,
    )


def _edge(src: str, tgt: str) -> DependencyEdge:
    return DependencyEdge(
        source_id=src,
        target_id=tgt,
        output_name="out",
        input_name="in",
        source_type="ndarray",
        target_type="ndarray",
    )


def _cdg(nodes: list[AlgorithmicNode], edges: list[DependencyEdge]) -> CDGExport:
    return CDGExport(nodes=nodes, edges=edges, metadata={})


def _baseline_cdg() -> CDGExport:
    return _cdg(
        [
            _node("src", "Source"),
            _node("acquire", "Acquire Data", ConceptType.BASELINE_ANALYSIS),
            _node("preprocess", "Preprocess", ConceptType.BASELINE_ANALYSIS),
            _node(
                "windowed",
                "Windowed Analysis",
                ConceptType.MAP_OVER,
                map_window_size=1024,
                map_hop_size=512,
            ),
            _node(
                "fit",
                "Fit",
                ConceptType.BASELINE_ANALYSIS,
                primitive="baseline_fit_stack",
                is_opaque=True,
            ),
            _node(
                "transform",
                "Output Transform",
                ConceptType.BASELINE_ANALYSIS,
            ),
            _node("normalize", "Normalize", ConceptType.BASELINE_ANALYSIS),
            _node("combine", "Combine", ConceptType.BASELINE_ANALYSIS),
            _node("regionize", "Regionize", ConceptType.BASELINE_ANALYSIS),
            _node("out", "Output"),
        ],
        [
            _edge("src", "acquire"),
            _edge("acquire", "preprocess"),
            _edge("preprocess", "windowed"),
            _edge("windowed", "fit"),
            _edge("fit", "transform"),
            _edge("transform", "normalize"),
            _edge("normalize", "combine"),
            _edge("combine", "regionize"),
            _edge("regionize", "out"),
        ],
    )


class TestBaselineAnalysisRuleSet:
    def test_rule_set_metadata(self):
        rule_set = BaselineAnalysisExpansionRuleSet()
        assert rule_set.name == "baseline_analysis"
        assert rule_set.domain == "baseline_analysis"

    def test_rule_names(self):
        names = {rule.name for rule in BaselineAnalysisExpansionRuleSet().rules()}
        assert names == {
            "insert_onset_coverage_check_after_fit",
            "insert_padding_saturation_after_transform",
            "insert_normalization_clipping_after_normalize",
            "insert_component_balance_after_combine",
        }

    def test_default_rule_sets_register_baseline_analysis(self):
        names = {rule_set.name for rule_set in default_rule_sets()}
        assert "baseline_analysis" in names


class TestBaselineAnalysisRuleApplication:
    def _get_rules(self):
        return {rule.name: rule for rule in BaselineAnalysisExpansionRuleSet().rules()}

    def test_onset_rule_applies(self):
        result = GraphRewriter().apply_rule(
            self._get_rules()["insert_onset_coverage_check_after_fit"],
            _baseline_cdg(),
        )
        assert not result.is_failure
        assert "check_onset_coverage" in {
            node.matched_primitive for node in result.unwrap().nodes if node.matched_primitive
        }

    def test_padding_rule_applies(self):
        result = GraphRewriter().apply_rule(
            self._get_rules()["insert_padding_saturation_after_transform"],
            _baseline_cdg(),
        )
        assert not result.is_failure
        assert "detect_padding_saturation" in {
            node.matched_primitive for node in result.unwrap().nodes if node.matched_primitive
        }

    def test_clipping_rule_applies(self):
        result = GraphRewriter().apply_rule(
            self._get_rules()["insert_normalization_clipping_after_normalize"],
            _baseline_cdg(),
        )
        assert not result.is_failure
        assert "monitor_normalization_clipping" in {
            node.matched_primitive for node in result.unwrap().nodes if node.matched_primitive
        }

    def test_component_balance_rule_applies(self):
        result = GraphRewriter().apply_rule(
            self._get_rules()["insert_component_balance_after_combine"],
            _baseline_cdg(),
        )
        assert not result.is_failure
        assert "validate_component_balance" in {
            node.matched_primitive for node in result.unwrap().nodes if node.matched_primitive
        }


class TestBaselineAnalysisDiagnostics:
    def test_onset_coverage_diagnostic(self):
        diagnostics = BaselineAnalysisExpansionRuleSet().diagnose(
            _baseline_cdg(),
            ExpansionContext(intermediates={"onset_density": 1e-6}),
        )
        assert "insert_onset_coverage_check_after_fit" in {
            diagnostic.rule_name for diagnostic in diagnostics
        }

    def test_padding_saturation_diagnostic(self):
        diagnostics = BaselineAnalysisExpansionRuleSet().diagnose(
            _baseline_cdg(),
            ExpansionContext(intermediates={"padding_overlap_fraction": 0.8}),
        )
        assert "insert_padding_saturation_after_transform" in {
            diagnostic.rule_name for diagnostic in diagnostics
        }

    def test_normalization_clipping_diagnostic(self):
        diagnostics = BaselineAnalysisExpansionRuleSet().diagnose(
            _baseline_cdg(),
            ExpansionContext(intermediates={"clipped_fraction": 0.6}),
        )
        assert "insert_normalization_clipping_after_normalize" in {
            diagnostic.rule_name for diagnostic in diagnostics
        }

    def test_component_balance_diagnostic(self):
        diagnostics = BaselineAnalysisExpansionRuleSet().diagnose(
            _baseline_cdg(),
            ExpansionContext(intermediates={"component_entropy": 0.1}),
        )
        assert "insert_component_balance_after_combine" in {
            diagnostic.rule_name for diagnostic in diagnostics
        }

    def test_all_good_intermediates_return_no_diagnostics(self):
        diagnostics = BaselineAnalysisExpansionRuleSet().diagnose(
            _baseline_cdg(),
            ExpansionContext(
                intermediates={
                    "onset_density": 1e-3,
                    "padding_overlap_fraction": 0.1,
                    "clipped_fraction": 0.01,
                    "component_entropy": 0.9,
                }
            ),
        )
        assert diagnostics == []

    def test_all_bad_intermediates_return_four_diagnostics(self):
        diagnostics = BaselineAnalysisExpansionRuleSet().diagnose(
            _baseline_cdg(),
            ExpansionContext(
                intermediates={
                    "onset_density": 1e-8,
                    "padding_overlap_fraction": 0.9,
                    "clipped_fraction": 0.8,
                    "component_entropy": 0.05,
                }
            ),
        )
        assert len(diagnostics) == 4


class TestBaselineAnalysisSkeleton:
    def test_fit_node_has_opaque_primitive(self):
        skeleton = get_skeleton(ConceptType.BASELINE_ANALYSIS)
        assert skeleton is not None
        fit = next(node for node in skeleton.template_nodes if node.name == "Fit")
        assert fit.is_opaque is True
        assert fit.matched_primitive == "baseline_fit_stack"

    def test_windowed_analysis_has_map_fields(self):
        skeleton = get_skeleton(ConceptType.BASELINE_ANALYSIS)
        assert skeleton is not None
        windowed = next(
            node for node in skeleton.template_nodes if node.name == "Windowed Analysis"
        )
        assert windowed.map_window_size == 1024
        assert windowed.map_hop_size == 512

    def test_instantiated_nodes_preserve_metadata(self):
        skeleton = get_skeleton(ConceptType.BASELINE_ANALYSIS)
        assert skeleton is not None
        nodes, _edges = instantiate_skeleton(skeleton, "baseline test")
        fit = next(node for node in nodes if node.name == "Fit")
        windowed = next(node for node in nodes if node.name == "Windowed Analysis")
        assert fit.is_opaque is True
        assert fit.matched_primitive == "baseline_fit_stack"
        assert windowed.map_window_size == 1024
        assert windowed.map_hop_size == 512
