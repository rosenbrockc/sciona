"""Tests for the String Matching expansion rules and runtime atoms."""

import numpy as np
import pytest

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules.string_matching import (
    StringMatchingExpansionRuleSet,
)
from sciona.expansion_atoms.runtime_string_matching import (
    analyze_alphabet_size,
    check_pattern_text_ratio,
    measure_hash_collision_rate,
    validate_failure_function,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(nid, name, concept=ConceptType.CUSTOM, primitive=None):
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
    )


def _edge(src, tgt):
    return DependencyEdge(
        source_id=src,
        target_id=tgt,
        output_name="out",
        input_name="in",
        source_type="ndarray",
        target_type="ndarray",
    )


def _cdg(nodes, edges):
    return CDGExport(nodes=nodes, edges=edges, metadata={})


def _string_matching_cdg():
    """Build a minimal string matching CDG matching the skeleton topology."""
    return _cdg(
        [
            _node("src", "Source", ConceptType.CUSTOM),
            _node("pp", "Preprocess", ConceptType.STRING_MATCHING),
            _node("sc", "Scan", ConceptType.STRING_MATCHING),
            _node("ma", "Match/Advance", ConceptType.STRING_MATCHING),
            _node("out", "Output", ConceptType.CUSTOM),
        ],
        [
            _edge("src", "pp"),
            _edge("pp", "sc"),
            _edge("sc", "ma"),
            _edge("ma", "out"),
        ],
    )


# ---------------------------------------------------------------------------
# Runtime atom tests
# ---------------------------------------------------------------------------


class TestAnalyzeAlphabetSize:
    def test_basic(self):
        text = np.array([65, 66, 67, 65, 66])  # A B C A B
        pattern = np.array([65, 66])  # A B
        t_size, p_size, overlap = analyze_alphabet_size(text, pattern)
        assert t_size == 3
        assert p_size == 2
        assert overlap == 1.0

    def test_no_overlap(self):
        text = np.array([65, 66, 67])
        pattern = np.array([68, 69])
        _, _, overlap = analyze_alphabet_size(text, pattern)
        assert overlap == 0.0

    def test_partial_overlap(self):
        text = np.array([65, 66])
        pattern = np.array([65, 67])
        _, _, overlap = analyze_alphabet_size(text, pattern)
        assert overlap == 0.5

    def test_empty_pattern(self):
        text = np.array([65, 66])
        pattern = np.array([], dtype=np.int64)
        t_size, p_size, overlap = analyze_alphabet_size(text, pattern)
        assert p_size == 0
        assert overlap == 1.0

    def test_empty_both(self):
        t_size, p_size, overlap = analyze_alphabet_size(
            np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        )
        assert t_size == 0
        assert p_size == 0


class TestCheckPatternTextRatio:
    def test_normal(self):
        ratio, assessment = check_pattern_text_ratio(10, 100)
        assert ratio == 0.1
        assert assessment == "normal"

    def test_pattern_longer(self):
        ratio, assessment = check_pattern_text_ratio(100, 50)
        assert ratio == 2.0
        assert assessment == "error"

    def test_very_short_pattern(self):
        ratio, assessment = check_pattern_text_ratio(1, 10000)
        assert ratio < 0.01
        assert assessment == "short"

    def test_long_pattern(self):
        ratio, assessment = check_pattern_text_ratio(60, 100)
        assert ratio == 0.6
        assert assessment == "long"

    def test_empty_text(self):
        ratio, assessment = check_pattern_text_ratio(5, 0)
        assert assessment == "error"

    def test_both_empty(self):
        ratio, assessment = check_pattern_text_ratio(0, 0)
        assert assessment == "normal"


class TestMeasureHashCollisionRate:
    def test_no_collisions(self):
        rate, is_exc = measure_hash_collision_rate(10, 10)
        assert rate == 0.0
        assert not is_exc

    def test_high_collisions(self):
        rate, is_exc = measure_hash_collision_rate(100, 10)
        assert rate == 0.9
        assert is_exc

    def test_no_matches(self):
        rate, is_exc = measure_hash_collision_rate(0, 0)
        assert rate == 0.0
        assert not is_exc

    def test_all_spurious(self):
        rate, is_exc = measure_hash_collision_rate(50, 0)
        assert rate == 1.0
        assert is_exc


class TestValidateFailureFunction:
    def test_valid_table(self):
        # KMP failure for "ABAB": [0, 0, 1, 2]
        table = np.array([0, 0, 1, 2])
        violations, is_valid = validate_failure_function(table, 4)
        assert violations == 0
        assert is_valid

    def test_invalid_first_entry(self):
        table = np.array([1, 0, 0])
        violations, is_valid = validate_failure_function(table, 3)
        assert violations >= 1
        assert not is_valid

    def test_value_exceeds_index(self):
        table = np.array([0, 2, 0])  # failure[1]=2 >= 1
        violations, is_valid = validate_failure_function(table, 3)
        assert violations >= 1
        assert not is_valid

    def test_wrong_length(self):
        table = np.array([0, 0])
        violations, is_valid = validate_failure_function(table, 5)
        assert violations >= 1
        assert not is_valid

    def test_empty(self):
        violations, is_valid = validate_failure_function(
            np.array([], dtype=np.int64), 0
        )
        assert violations == 0
        assert is_valid


# ---------------------------------------------------------------------------
# DPO rule application tests
# ---------------------------------------------------------------------------


class TestStringMatchingRules:
    def _get_rules(self):
        rs = StringMatchingExpansionRuleSet()
        return {r.name: r for r in rs.rules()}

    def test_alphabet_analysis_applies(self):
        rules = self._get_rules()
        rule = rules["insert_alphabet_analysis_before_preprocess"]
        rw = GraphRewriter()
        cdg = _string_matching_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "analyze_alphabet_size" in prims
        assert len(g.nodes) == 6  # 5 + 1

    def test_pattern_text_ratio_check_applies(self):
        rules = self._get_rules()
        rule = rules["insert_pattern_text_ratio_check_before_preprocess"]
        rw = GraphRewriter()
        cdg = _string_matching_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "check_pattern_text_ratio" in prims

    def test_hash_collision_detection_applies(self):
        rules = self._get_rules()
        rule = rules["insert_hash_collision_detection_after_scan"]
        rw = GraphRewriter()
        cdg = _string_matching_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "measure_hash_collision_rate" in prims

    def test_failure_function_validation_applies(self):
        rules = self._get_rules()
        rule = rules["insert_failure_function_validation_after_preprocess"]
        rw = GraphRewriter()
        cdg = _string_matching_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "validate_failure_function" in prims


# ---------------------------------------------------------------------------
# Diagnostic tests
# ---------------------------------------------------------------------------


class TestStringMatchingDiagnostics:
    def test_diagnose_low_alphabet_overlap(self):
        rs = StringMatchingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"alphabet_overlap_ratio": 0.5}
        )
        cdg = _string_matching_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_alphabet_analysis_before_preprocess" in names

    def test_full_overlap_no_trigger(self):
        rs = StringMatchingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"alphabet_overlap_ratio": 1.0}
        )
        cdg = _string_matching_cdg()
        diags = rs.diagnose(cdg, ctx)
        alpha_diags = [
            d for d in diags
            if d.rule_name == "insert_alphabet_analysis_before_preprocess"
        ]
        assert len(alpha_diags) == 0

    def test_diagnose_pattern_longer_than_text(self):
        rs = StringMatchingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"pattern_length": 100, "text_length": 50}
        )
        cdg = _string_matching_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_pattern_text_ratio_check_before_preprocess" in names

    def test_diagnose_very_short_pattern(self):
        rs = StringMatchingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"pattern_length": 1, "text_length": 10000}
        )
        cdg = _string_matching_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_pattern_text_ratio_check_before_preprocess" in names

    def test_normal_ratio_no_trigger(self):
        rs = StringMatchingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"pattern_length": 10, "text_length": 100}
        )
        cdg = _string_matching_cdg()
        diags = rs.diagnose(cdg, ctx)
        ratio_diags = [
            d for d in diags
            if d.rule_name == "insert_pattern_text_ratio_check_before_preprocess"
        ]
        assert len(ratio_diags) == 0

    def test_diagnose_high_collisions(self):
        rs = StringMatchingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"hash_collision_rate": 0.8}
        )
        cdg = _string_matching_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_hash_collision_detection_after_scan" in names

    def test_low_collisions_no_trigger(self):
        rs = StringMatchingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"hash_collision_rate": 0.1}
        )
        cdg = _string_matching_cdg()
        diags = rs.diagnose(cdg, ctx)
        coll_diags = [
            d for d in diags
            if d.rule_name == "insert_hash_collision_detection_after_scan"
        ]
        assert len(coll_diags) == 0

    def test_diagnose_failure_function_violations(self):
        rs = StringMatchingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"failure_function_violations": 3}
        )
        cdg = _string_matching_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_failure_function_validation_after_preprocess" in names

    def test_valid_failure_function_no_trigger(self):
        rs = StringMatchingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"failure_function_violations": 0}
        )
        cdg = _string_matching_cdg()
        diags = rs.diagnose(cdg, ctx)
        ff_diags = [
            d for d in diags
            if d.rule_name == "insert_failure_function_validation_after_preprocess"
        ]
        assert len(ff_diags) == 0

    def test_no_data_returns_nothing(self):
        rs = StringMatchingExpansionRuleSet()
        cdg = _string_matching_cdg()
        diags = rs.diagnose(cdg, ExpansionContext())
        assert diags == []


# ---------------------------------------------------------------------------
# Integration: full expansion engine
# ---------------------------------------------------------------------------


class TestStringMatchingIntegration:
    def test_full_expansion_with_all_diagnostics(self):
        """End-to-end: diagnostics fire, engine expands string matching CDG."""
        rs = StringMatchingExpansionRuleSet()
        engine = ExpansionEngine([rs])

        ctx = ExpansionContext(
            intermediates={
                "alphabet_overlap_ratio": 0.5,
                "pattern_length": 100,
                "text_length": 50,
                "hash_collision_rate": 0.8,
                "failure_function_violations": 2,
            }
        )
        cdg = _string_matching_cdg()
        result = engine.expand(cdg, ctx)

        assert result.expanded
        assert len(result.applied_rules) >= 1
        prims = {n.matched_primitive for n in result.cdg.nodes if n.matched_primitive}
        expansion_atoms = prims & {
            "analyze_alphabet_size",
            "check_pattern_text_ratio",
            "measure_hash_collision_rate",
            "validate_failure_function",
        }
        assert len(expansion_atoms) >= 1

    def test_cross_domain_with_sorting_rules(self):
        """String matching + Sorting rules both available; only relevant ones fire."""
        from sciona.principal.expansion_rules.sorting import (
            SortingExpansionRuleSet,
        )

        engine = ExpansionEngine([
            StringMatchingExpansionRuleSet(),
            SortingExpansionRuleSet(),
        ])

        ctx = ExpansionContext(
            intermediates={"alphabet_overlap_ratio": 0.5}
        )
        cdg = _string_matching_cdg()
        result = engine.expand(cdg, ctx)

        sorting_atoms = {
            "measure_presortedness",
            "analyze_comparison_count",
            "analyze_swap_count",
            "validate_stability",
        }
        applied_prims = {
            n.matched_primitive for n in result.cdg.nodes if n.matched_primitive
        }
        assert not (applied_prims & sorting_atoms)
