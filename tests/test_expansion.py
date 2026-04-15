"""Tests for the expansion engine and cross-domain expansion."""

import pytest

from sciona.architect.graph_rewriter import GraphRewriter, Morphism, RewriteRule
from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.principal.expansion import (
    ExpansionContext,
    ExpansionDiagnostic,
    ExpansionEngine,
    ExpansionResult,
    ExpansionRuleSet,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(nid, primitive=None, concept=ConceptType.CUSTOM):
    return AlgorithmicNode(
        node_id=nid,
        name=nid,
        description=nid,
        concept_type=concept,
        status=NodeStatus.ATOMIC,
        matched_primitive=primitive,
        inputs=[IOSpec(name="in", type_desc="np.ndarray")],
        outputs=[IOSpec(name="out", type_desc="np.ndarray")],
        type_signature=f"{nid} -> r",
    )


def _edge(src, tgt):
    return DependencyEdge(
        source_id=src,
        target_id=tgt,
        output_name="out",
        input_name="in",
        source_type="np.ndarray",
        target_type="np.ndarray",
    )


def _cdg(nodes, edges):
    return CDGExport(nodes=nodes, edges=edges, metadata={})


def _interpose_rule(name, target_prim, new_prim):
    """Generic interposition rule: insert new_prim before target_prim."""
    src = _node("src")
    tgt = _node("tgt", primitive=target_prim)
    new = _node("new", primitive=new_prim)

    lhs = _cdg([src, tgt], [_edge("src", "tgt")])
    k = _cdg([src, tgt], [])
    rhs = _cdg([src, new, tgt], [_edge("src", "new"), _edge("new", "tgt")])

    morph = Morphism(node_map={"src": "src", "tgt": "tgt"}, edge_map={})
    return RewriteRule(name, lhs, rhs, k, morph, morph)


# ---------------------------------------------------------------------------
# Mock rule sets
# ---------------------------------------------------------------------------


class MockRuleSet:
    def __init__(self, name, domain, diagnostics_fn, rule_list):
        self.name = name
        self.domain = domain
        self._diag_fn = diagnostics_fn
        self._rules = rule_list

    def diagnose(self, cdg, context):
        return self._diag_fn(cdg, context)

    def rules(self):
        return self._rules


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExpansionEngine:
    def test_runtime_inputs_and_signal_data_alias_each_other(self):
        ctx = ExpansionContext(runtime_inputs={"features": [1.0, 2.0]})
        assert ctx.signal_data == {"features": [1.0, 2.0]}

        legacy = ExpansionContext(signal_data={"signal": [0.0, 1.0]})
        assert legacy.runtime_inputs == {"signal": [0.0, 1.0]}

    def test_no_diagnostics_no_expansion(self):
        rs = MockRuleSet("empty", "test", lambda c, ctx: [], [])
        engine = ExpansionEngine([rs])
        cdg = _cdg([_node("a")], [])
        result = engine.expand(cdg, ExpansionContext())
        assert not result.expanded
        assert result.applied_rules == ()

    def test_applies_rule_on_diagnostic(self):
        rule = _interpose_rule("insert_X", "B", "X")
        diag = ExpansionDiagnostic(
            rule_name="insert_X",
            severity=0.8,
            evidence="test",
            metric_name="m",
            metric_value=1.0,
            threshold=0.5,
            asset_id="asset.insert_x.v1",
            asset_version="v1",
            asset_family="test_family",
            asset_source_kind="local_asset",
            asset_review_status="transitional",
            asset_operation="insert_X",
        )
        rs = MockRuleSet("sig", "signal", lambda c, ctx: [diag], [rule])
        engine = ExpansionEngine([rs])

        cdg = _cdg(
            [_node("a"), _node("b", primitive="B")],
            [_edge("a", "b")],
        )
        result = engine.expand(cdg, ExpansionContext())
        assert result.expanded
        assert "insert_X" in result.applied_rules
        assert result.applied_assets[0]["asset_id"] == "asset.insert_x.v1"
        assert len(result.cdg.nodes) == 3

    def test_severity_ordering(self):
        """Higher severity diagnostics are applied first."""
        rule_high = _interpose_rule("r_high", "B", "H")
        rule_low = _interpose_rule("r_low", "B", "L")
        diag_low = ExpansionDiagnostic(
            rule_name="r_low", severity=0.4, evidence="",
            metric_name="m", metric_value=1.0, threshold=0.5,
        )
        diag_high = ExpansionDiagnostic(
            rule_name="r_high", severity=0.9, evidence="",
            metric_name="m", metric_value=1.0, threshold=0.5,
        )
        rs = MockRuleSet(
            "test", "test",
            lambda c, ctx: [diag_low, diag_high],
            [rule_high, rule_low],
        )
        engine = ExpansionEngine([rs])
        cdg = _cdg(
            [_node("a"), _node("b", primitive="B")],
            [_edge("a", "b")],
        )
        result = engine.expand(cdg, ExpansionContext())
        # Higher severity rule applied first
        assert result.applied_rules[0] == "r_high"

    def test_below_threshold_ignored(self):
        rule = _interpose_rule("insert_X", "B", "X")
        diag = ExpansionDiagnostic(
            rule_name="insert_X", severity=0.1, evidence="weak",
            metric_name="m", metric_value=0.1, threshold=0.5,
        )
        rs = MockRuleSet("sig", "signal", lambda c, ctx: [diag], [rule])
        engine = ExpansionEngine([rs], activation_threshold=0.3)
        cdg = _cdg([_node("a"), _node("b", primitive="B")], [_edge("a", "b")])
        result = engine.expand(cdg, ExpansionContext())
        assert not result.expanded

    def test_default_engine_does_not_suppress_emitted_low_severity_diagnostic(self):
        rule = _interpose_rule("insert_X", "B", "X")
        diag = ExpansionDiagnostic(
            rule_name="insert_X", severity=0.1, evidence="triggered",
            metric_name="m", metric_value=0.6, threshold=0.5,
        )
        rs = MockRuleSet("sig", "signal", lambda c, ctx: [diag], [rule])
        engine = ExpansionEngine([rs])
        cdg = _cdg([_node("a"), _node("b", primitive="B")], [_edge("a", "b")])
        result = engine.expand(cdg, ExpansionContext())
        assert result.expanded
        assert result.applied_rules == ("insert_X",)

    def test_graceful_failure_no_match(self):
        """Rule that doesn't match CDG topology is silently skipped."""
        rule = _interpose_rule("insert_X", "MISSING", "X")
        diag = ExpansionDiagnostic(
            rule_name="insert_X", severity=0.9, evidence="",
            metric_name="m", metric_value=1.0, threshold=0.5,
        )
        rs = MockRuleSet("sig", "signal", lambda c, ctx: [diag], [rule])
        engine = ExpansionEngine([rs])
        cdg = _cdg([_node("a", primitive="A")], [])
        result = engine.expand(cdg, ExpansionContext())
        assert not result.expanded

    def test_cross_domain_expansion(self):
        """Rules from different domains can both fire on the same CDG."""
        rule_sig = _interpose_rule("sig_rule", "B", "SIG_NEW")
        rule_stats = _interpose_rule("stats_rule", "SIG_NEW", "STATS_NEW")

        diag_sig = ExpansionDiagnostic(
            rule_name="sig_rule", severity=0.9, evidence="",
            metric_name="m", metric_value=1.0, threshold=0.5,
            source_domain="signal",
        )
        diag_stats = ExpansionDiagnostic(
            rule_name="stats_rule", severity=0.7, evidence="",
            metric_name="m", metric_value=1.0, threshold=0.5,
            source_domain="stats",
        )

        rs_sig = MockRuleSet("sig", "signal", lambda c, ctx: [diag_sig], [rule_sig])
        rs_stats = MockRuleSet("stats", "stats", lambda c, ctx: [diag_stats], [rule_stats])

        engine = ExpansionEngine([rs_sig, rs_stats])
        cdg = _cdg(
            [_node("a"), _node("b", primitive="B")],
            [_edge("a", "b")],
        )
        result = engine.expand(cdg, ExpansionContext())
        assert result.expanded
        # Both rules should have applied: sig_rule inserts SIG_NEW,
        # then stats_rule inserts STATS_NEW before SIG_NEW
        assert len(result.applied_rules) == 2
        assert len(result.cdg.nodes) == 4  # a, SIG_NEW, STATS_NEW, b

    def test_register_adds_rule_set(self):
        engine = ExpansionEngine()
        rule = _interpose_rule("r", "B", "X")
        diag = ExpansionDiagnostic(
            rule_name="r", severity=0.9, evidence="",
            metric_name="m", metric_value=1.0, threshold=0.5,
        )
        rs = MockRuleSet("late", "test", lambda c, ctx: [diag], [rule])
        engine.register(rs)

        cdg = _cdg([_node("a"), _node("b", primitive="B")], [_edge("a", "b")])
        result = engine.expand(cdg, ExpansionContext())
        assert result.expanded

    def test_diagnostic_exception_is_caught(self):
        """If a rule set's diagnose() throws, others still run."""
        def explode(cdg, ctx):
            raise RuntimeError("kaboom")

        rule = _interpose_rule("r", "B", "X")
        diag = ExpansionDiagnostic(
            rule_name="r", severity=0.9, evidence="",
            metric_name="m", metric_value=1.0, threshold=0.5,
        )
        rs_bad = MockRuleSet("bad", "test", explode, [])
        rs_good = MockRuleSet("good", "test", lambda c, ctx: [diag], [rule])

        engine = ExpansionEngine([rs_bad, rs_good])
        cdg = _cdg([_node("a"), _node("b", primitive="B")], [_edge("a", "b")])
        result = engine.expand(cdg, ExpansionContext())
        assert result.expanded
