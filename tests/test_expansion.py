"""Tests for the expansion engine and cross-domain expansion."""

import numpy as np
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
from sciona.synthesizer.assembler import Assembler
from sciona.types import (
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


class TestSignalEventRateExpansion:
    """Integration tests for the signal-event-rate expansion rules."""

    def _pipeline_cdg(self, use_smoothed=False, rate_primitive=None):
        """Build a minimal filter → detect → rate CDG."""
        rate_prim = rate_primitive or (
            "compute_event_rate_smoothed" if use_smoothed else "compute_event_rate"
        )
        return _cdg(
            [
                _node("src"),
                _node("filt", primitive="filter_signal_for_detection",
                      concept=ConceptType.SIGNAL_FILTER),
                _node("det", primitive="detect_peaks_in_signal",
                      concept=ConceptType.DATA_EXTRACTION),
                _node("rate", primitive=rate_prim, concept=ConceptType.ANALYSIS),
            ],
            [
                _edge("src", "filt"),
                _edge("filt", "det"),
                _edge("det", "rate"),
            ],
        )

    def _root_boundary_pipeline_cdg(self):
        return _cdg(
            [
                AlgorithmicNode(
                    node_id="root",
                    name="ECG HR",
                    description="Top-level ECG HR pipeline",
                    concept_type=ConceptType.ANALYSIS,
                    status=NodeStatus.DECOMPOSED,
                    children=["filt", "det", "rate"],
                    depth=0,
                    inputs=[
                        IOSpec(name="signal", type_desc="np.ndarray"),
                        IOSpec(name="sampling_rate", type_desc="float"),
                    ],
                    outputs=[IOSpec(name="rate", type_desc="np.ndarray")],
                ),
                AlgorithmicNode(
                    node_id="filt",
                    parent_id="root",
                    name="Filter Signal",
                    description="Condition the signal",
                    concept_type=ConceptType.SIGNAL_FILTER,
                    status=NodeStatus.ATOMIC,
                    matched_primitive="filter_signal_for_detection",
                    depth=1,
                    inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
                    outputs=[IOSpec(name="signal", type_desc="np.ndarray")],
                ),
                AlgorithmicNode(
                    node_id="det",
                    parent_id="root",
                    name="Detect Peaks",
                    description="Detect peaks",
                    concept_type=ConceptType.DATA_EXTRACTION,
                    status=NodeStatus.ATOMIC,
                    matched_primitive="detect_peaks_in_signal",
                    depth=1,
                    inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
                    outputs=[IOSpec(name="events", type_desc="np.ndarray")],
                ),
                AlgorithmicNode(
                    node_id="rate",
                    parent_id="root",
                    name="Compute Event Rate",
                    description="Compute rate",
                    concept_type=ConceptType.ANALYSIS,
                    status=NodeStatus.ATOMIC,
                    matched_primitive="compute_event_rate",
                    depth=1,
                    inputs=[IOSpec(name="events", type_desc="np.ndarray")],
                    outputs=[IOSpec(name="rate", type_desc="np.ndarray")],
                ),
            ],
            [
                DependencyEdge(
                    source_id="filt",
                    target_id="det",
                    output_name="signal",
                    input_name="signal",
                    source_type="np.ndarray",
                    target_type="np.ndarray",
                ),
                DependencyEdge(
                    source_id="det",
                    target_id="rate",
                    output_name="events",
                    input_name="events",
                    source_type="np.ndarray",
                    target_type="np.ndarray",
                ),
            ],
        )

    def test_jump_removal_rule_applies(self):
        from sciona.principal.expansion_rules.signal_event_rate import (
            SignalEventRateExpansionRuleSet,
        )

        rs = SignalEventRateExpansionRuleSet()
        rules_by_name = {r.name: r for r in rs.rules()}
        rule = rules_by_name["insert_jump_removal_before_filter"]

        rw = GraphRewriter()
        cdg = self._pipeline_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure

        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "remove_signal_jumps" in prims
        assert len(g.nodes) == 5  # original 4 + jump removal

    def test_jump_removal_rule_applies_without_explicit_source_node(self):
        from sciona.principal.expansion_rules.signal_event_rate import (
            SignalEventRateExpansionRuleSet,
        )

        rs = SignalEventRateExpansionRuleSet()
        rules_by_name = {r.name: r for r in rs.rules()}
        rule = rules_by_name["insert_jump_removal_before_filter"]

        rw = GraphRewriter()
        cdg = self._root_boundary_pipeline_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure

        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "remove_signal_jumps" in prims
        filter_incoming = [
            edge for edge in g.edges if edge.target_id == "filt" and edge.input_name == "signal"
        ]
        assert len(filter_incoming) == 1
        assert filter_incoming[0].source_id != "filt"

    def test_sqi_rule_applies(self):
        from sciona.principal.expansion_rules.signal_event_rate import (
            SignalEventRateExpansionRuleSet,
        )

        rs = SignalEventRateExpansionRuleSet()
        rules_by_name = {r.name: r for r in rs.rules()}
        rule = rules_by_name["insert_sqi_before_filter"]

        rw = GraphRewriter()
        cdg = self._pipeline_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure

        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "assess_signal_quality" in prims

    def test_outlier_rejection_rule_applies(self):
        from sciona.principal.expansion_rules.signal_event_rate import (
            _biosppy_ecg_primitive_fqdn,
            SignalEventRateExpansionRuleSet,
        )

        rs = SignalEventRateExpansionRuleSet()
        rules_by_name = {r.name: r for r in rs.rules()}
        rule = rules_by_name["insert_outlier_rejection_after_detection"]

        rw = GraphRewriter()
        cdg = self._pipeline_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure

        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert _biosppy_ecg_primitive_fqdn("reject_outlier_intervals") in prims

    def test_peak_correction_rule_applies(self):
        from sciona.principal.expansion_rules.signal_event_rate import (
            SignalEventRateExpansionRuleSet,
        )

        rs = SignalEventRateExpansionRuleSet()
        rules_by_name = {r.name: r for r in rs.rules()}
        rule = rules_by_name["insert_peak_correction_after_detection"]

        rw = GraphRewriter()
        cdg = self._pipeline_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure

        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "peak_correction" in prims
        correction_inputs = {
            (edge.source_id, edge.target_id, edge.input_name)
            for edge in g.edges
            if edge.target_id != "rate"
        }
        assert ("filt", next(node.node_id for node in g.nodes if node.matched_primitive == "peak_correction"), "filtered") in correction_inputs

    def test_outlier_rejection_smoothed_variant(self):
        from sciona.principal.expansion_rules.signal_event_rate import (
            SignalEventRateExpansionRuleSet,
        )

        rs = SignalEventRateExpansionRuleSet()
        rules_by_name = {r.name: r for r in rs.rules()}
        rule = rules_by_name["insert_outlier_rejection_after_detection_smoothed"]

        rw = GraphRewriter()
        cdg = self._pipeline_cdg(use_smoothed=True)
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure

    def test_outlier_rejection_median_smoothed_variant(self):
        from sciona.principal.expansion_rules.signal_event_rate import (
            _biosppy_ecg_primitive_fqdn,
            SignalEventRateExpansionRuleSet,
        )

        rs = SignalEventRateExpansionRuleSet()
        rules_by_name = {r.name: r for r in rs.rules()}
        rule = rules_by_name["insert_outlier_rejection_after_detection_median_smoothed"]

        rw = GraphRewriter()
        cdg = self._pipeline_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure

        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert _biosppy_ecg_primitive_fqdn("reject_outlier_intervals") in prims
        assert _biosppy_ecg_primitive_fqdn(
            "heart_rate_computation_median_smoothed"
        ) in prims

    def test_outlier_rejection_median_smoothed_variant_preserves_root_children(self):
        from sciona.principal.expansion_rules.signal_event_rate import (
            _biosppy_ecg_primitive_fqdn,
            SignalEventRateExpansionRuleSet,
        )

        rs = SignalEventRateExpansionRuleSet()
        rules_by_name = {r.name: r for r in rs.rules()}
        rule = rules_by_name["insert_outlier_rejection_after_detection_median_smoothed"]

        rw = GraphRewriter()
        cdg = self._root_boundary_pipeline_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure

        g = result.unwrap()
        root = next(node for node in g.nodes if node.node_id == "root")
        robust_node = next(
            node
            for node in g.nodes
            if node.matched_primitive
            == _biosppy_ecg_primitive_fqdn(
                "heart_rate_computation_median_smoothed"
            )
        )
        assert robust_node.parent_id == "root"
        assert robust_node.node_id in root.children
        assert [port.name for port in robust_node.outputs] == ["rate"]

    def test_biosppy_ecg_primitive_fqdn_prefers_pilot_namespace(self, tmp_path, monkeypatch):
        from sciona.principal.expansion_rules.signal_event_rate import (
            _biosppy_ecg_primitive_fqdn,
        )

        pilot_ecg = (
            tmp_path
            / "src"
            / "sciona"
            / "atoms"
            / "signal_processing"
            / "biosppy"
            / "ecg.py"
        )
        pilot_ecg.parent.mkdir(parents=True)
        pilot_ecg.write_text("# pilot atom\n", encoding="utf-8")
        monkeypatch.setattr(
            "sciona.principal.expansion_rules.signal_event_rate.candidate_atom_provider_roots",
            lambda: [tmp_path],
        )

        assert _biosppy_ecg_primitive_fqdn(
            "reject_outlier_intervals"
        ) == "sciona.atoms.signal_processing.biosppy.ecg.reject_outlier_intervals"

    def test_biosppy_ecg_primitive_fqdn_falls_back_to_legacy_namespace(self, tmp_path, monkeypatch):
        from sciona.principal.expansion_rules.signal_event_rate import (
            _biosppy_ecg_primitive_fqdn,
        )

        monkeypatch.setattr(
            "sciona.principal.expansion_rules.signal_event_rate.candidate_atom_provider_roots",
            lambda: [tmp_path],
        )

        assert _biosppy_ecg_primitive_fqdn(
            "heart_rate_computation_median_smoothed"
        ) == "ageoa.biosppy.ecg.heart_rate_computation_median_smoothed"

    def test_outlier_rejection_median_smoothed_variant_preserves_rate_output_tags_in_assembly(self):
        from sciona.principal.expansion_rules.signal_event_rate import (
            SignalEventRateExpansionRuleSet,
        )

        rs = SignalEventRateExpansionRuleSet()
        rules_by_name = {r.name: r for r in rs.rules()}
        rule = rules_by_name["insert_outlier_rejection_after_detection_median_smoothed"]

        rw = GraphRewriter()
        cdg = self._root_boundary_pipeline_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure

        g = result.unwrap()
        matches: list[MatchResult] = []
        for index, node in enumerate(g.nodes):
            if node.status != NodeStatus.ATOMIC:
                continue
            decl = Declaration(
                name=f"fake_runtime.atom_{index}",
                type_signature=node.type_signature or "",
                prover=Prover.PYTHON,
            )
            candidate = CandidateMatch(
                declaration=decl,
                score=1.0,
                retrieval_method="test",
            )
            verified = VerificationResult(candidate=candidate, verified=True)
            matches.append(
                MatchResult(
                    pdg_node=PDGNode(
                        predicate_id=node.node_id,
                        statement=node.type_signature or "",
                    ),
                    verified_match=verified,
                    all_candidates=[candidate],
                    all_verifications=[verified],
                )
            )

        source = Assembler(Prover.PYTHON).assemble(
            g,
            matches,
            with_telemetry=True,
        ).source_code

        assert "def compute_robust_smoothed_rate" in source
        assert "output_names=('rate',)" in source
        assert "return compute_robust_smoothed_rate_result" in source

    def test_diagnose_jump_discontinuities(self):
        from sciona.principal.expansion_rules.signal_event_rate import (
            SignalEventRateExpansionRuleSet,
        )

        rs = SignalEventRateExpansionRuleSet()
        # Create signal with obvious jumps
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(5000)
        # Insert 10 large jumps
        for i in range(10):
            signal[500 * (i + 1) :] += 50.0

        ctx = ExpansionContext(signal_data={"signal": signal, "sampling_rate": 500.0})
        cdg = self._pipeline_cdg()
        diags = rs.diagnose(cdg, ctx)
        rule_names = {d.rule_name for d in diags}
        assert "insert_jump_removal_before_filter" in rule_names
        jump_diag = next(
            diag for diag in diags if diag.rule_name == "insert_jump_removal_before_filter"
        )
        assert jump_diag.asset_id == "family.signal_event_rate.expansions.v1"
        assert jump_diag.asset_operation == "insert_jump_removal_before_filter"

    def test_diagnose_jump_discontinuities_from_summary_telemetry(self):
        from sciona.principal.expansion_rules.signal_event_rate import (
            SignalEventRateExpansionRuleSet,
        )

        rs = SignalEventRateExpansionRuleSet()
        ctx = ExpansionContext(
            runtime_evidence={
                "telemetry_summary": {
                    "signal": {
                        "count": 38943.0,
                        "discontinuity_count": 3716.0,
                    }
                }
            }
        )
        cdg = self._pipeline_cdg()
        diags = rs.diagnose(cdg, ctx)
        rule_names = {d.rule_name for d in diags}
        assert "insert_jump_removal_before_filter" in rule_names

    def test_diagnose_interval_outliers(self):
        from sciona.principal.expansion_rules.signal_event_rate import (
            SignalEventRateExpansionRuleSet,
        )

        rs = SignalEventRateExpansionRuleSet()
        # Events with varied intervals including clear outliers.
        # Normal intervals ~480-520, outliers at 10 and 2500.
        events = np.array([
            0, 480, 1000, 1490, 2010, 2020, 2530, 3020, 3500, 6000, 6510,
        ])
        ctx = ExpansionContext(intermediates={"events": events})
        cdg = self._pipeline_cdg()
        diags = rs.diagnose(cdg, ctx)
        rule_names = {d.rule_name for d in diags}
        assert "insert_outlier_rejection_after_detection_median_smoothed" in rule_names

    def test_diagnose_interval_outliers_from_summary_telemetry(self):
        from sciona.principal.expansion_rules.signal_event_rate import (
            SignalEventRateExpansionRuleSet,
        )

        rs = SignalEventRateExpansionRuleSet()
        ctx = ExpansionContext(
            runtime_evidence={
                "telemetry_summary": {
                    "events": {
                        "count": 438.0,
                        "outlier_fraction": 0.22,
                        "interval_median_samples": 88.0,
                    }
                }
            }
        )
        cdg = self._pipeline_cdg()
        diags = rs.diagnose(cdg, ctx)
        rule_names = {d.rule_name for d in diags}
        assert "insert_outlier_rejection_after_detection_median_smoothed" in rule_names

    def test_diagnose_peak_correction_need_from_summary_telemetry(self):
        from sciona.principal.expansion_rules.signal_event_rate import (
            SignalEventRateExpansionRuleSet,
        )

        rs = SignalEventRateExpansionRuleSet()
        ctx = ExpansionContext(
            runtime_evidence={
                "telemetry_summary": {
                    "signal": {
                        "count": 38943.0,
                    },
                    "events": {
                        "count": 438.0,
                        "outlier_fraction": 0.06,
                    },
                },
                "canonical_runtime_context": {
                    "canonical_inputs": {
                        "signal": {"raw_key": "h10_ecg_value"},
                        "sampling_rate": {"raw_key": "ecg_sampling_rate"},
                    }
                },
            }
        )
        cdg = self._pipeline_cdg()
        diags = rs.diagnose(cdg, ctx)
        rule_names = {d.rule_name for d in diags}
        assert "insert_peak_correction_after_detection" in rule_names

    def test_diagnose_no_signal_data_returns_nothing(self):
        from sciona.principal.expansion_rules.signal_event_rate import (
            SignalEventRateExpansionRuleSet,
        )

        rs = SignalEventRateExpansionRuleSet()
        cdg = self._pipeline_cdg()
        diags = rs.diagnose(cdg, ExpansionContext())
        assert diags == []

    def test_full_expansion_integration(self):
        """End-to-end: diagnostics fire → engine applies rules → CDG expanded."""
        from sciona.principal.expansion_rules.signal_event_rate import (
            SignalEventRateExpansionRuleSet,
        )

        rs = SignalEventRateExpansionRuleSet()
        engine = ExpansionEngine([rs])

        # Signal with jumps
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(5000)
        for i in range(10):
            signal[500 * (i + 1) :] += 50.0

        # Events with outlier intervals
        events = np.array([0, 500, 1000, 1500, 1510, 2000, 2500, 3000, 5500, 6000])

        ctx = ExpansionContext(
            signal_data={"signal": signal, "sampling_rate": 500.0},
            intermediates={"events": events},
        )
        cdg = self._pipeline_cdg()
        result = engine.expand(cdg, ctx)

        assert result.expanded
        assert len(result.applied_rules) >= 1
        assert result.applied_assets
        assert (
            result.applied_assets[0]["asset_id"]
            == "family.signal_event_rate.expansions.v1"
        )
        prims = {n.matched_primitive for n in result.cdg.nodes if n.matched_primitive}
        # At least one expansion atom should be present
        expansion_atoms = prims & {
            "remove_signal_jumps",
            "assess_signal_quality",
            "reject_outlier_intervals",
        }
        assert len(expansion_atoms) >= 1

    def test_boundary_aware_expansion_integration(self):
        from sciona.principal.expansion_rules.signal_event_rate import (
            SignalEventRateExpansionRuleSet,
        )

        rs = SignalEventRateExpansionRuleSet()
        engine = ExpansionEngine([rs])

        rng = np.random.default_rng(42)
        signal = rng.standard_normal(5000)
        for i in range(10):
            signal[500 * (i + 1) :] += 50.0

        result = engine.expand(
            self._root_boundary_pipeline_cdg(),
            ExpansionContext(signal_data={"signal": signal, "sampling_rate": 500.0}),
        )

        assert result.expanded
        jump = next(
            node
            for node in result.cdg.nodes
            if node.matched_primitive == "remove_signal_jumps"
        )
        root = next(node for node in result.cdg.nodes if node.node_id == "root")
        assert jump.parent_id == "root"
        assert jump.node_id in root.children
        assert any(
            edge.source_id == jump.node_id
            and edge.target_id == "filt"
            and edge.input_name == "signal"
            for edge in result.cdg.edges
        )

    def test_asset_backed_rules_do_not_require_missing_planning_categories(self):
        from sciona.principal.expansion import ExpansionContext
        from sciona.principal.expansion_assets import asset_backed_rule_sets
        from sciona.principal.expansion_rules.signal_event_rate import (
            SignalEventRateExpansionRuleSet,
        )

        wrapped = asset_backed_rule_sets([SignalEventRateExpansionRuleSet()])[0]
        diagnostics = wrapped.diagnose(
            self._root_boundary_pipeline_cdg(),
            ExpansionContext(
                runtime_evidence={
                    "telemetry_summary": {
                        "signal": {
                            "count": 512.0,
                            "discontinuity_count": 12.0,
                        }
                    },
                    "canonical_runtime_context": {
                        "canonical_inputs": {
                            "signal": {
                                "raw_key": "h10_ecg_value",
                            },
                            "sampling_rate": {
                                "raw_key": "ecg_sampling_rate",
                            },
                        }
                    },
                },
                planning_artifact={
                    "family_hint": "signal_detect_measure",
                    "planning_constraints": [
                        {"category": "data_kind"},
                        {"category": "provenance"},
                    ],
                },
            ),
        )
        assert any(
            diagnostic.rule_name == "insert_jump_removal_before_filter"
            for diagnostic in diagnostics
        )
