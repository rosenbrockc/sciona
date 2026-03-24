"""Tests for the Compression expansion rules and runtime atoms."""

import numpy as np
import pytest

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules.compression import CompressionExpansionRuleSet
from sciona.expansion_atoms.runtime_compression import (
    analyze_compression_ratio,
    detect_dictionary_bloat,
    monitor_encoding_throughput,
    validate_lossless_roundtrip,
)


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
    return DependencyEdge(source_id=src, target_id=tgt, output_name="out", input_name="in", source_type="ndarray", target_type="ndarray")


def _cdg(nodes, edges):
    return CDGExport(nodes=nodes, edges=edges, metadata={})


def _compression_cdg():
    return _cdg(
        [
            _node("src", "Source"),
            _node("mdl", "Model Source", ConceptType.COMPRESSION),
            _node("enc", "Encode", ConceptType.COMPRESSION),
            _node("dec", "Decode/Verify", ConceptType.COMPRESSION),
            _node("out", "Output"),
        ],
        [_edge("src", "mdl"), _edge("mdl", "enc"), _edge("enc", "dec"), _edge("dec", "out")],
    )


class TestAnalyzeCompressionRatio:
    def test_efficient(self):
        gap, ok = analyze_compression_ratio(1000.0, 400.0, 0.3)
        assert ok
        assert gap <= 0.2

    def test_inefficient(self):
        gap, ok = analyze_compression_ratio(1000.0, 800.0, 0.3)
        assert not ok
        assert gap > 0.2

    def test_zero_original(self):
        gap, ok = analyze_compression_ratio(0.0, 0.0, 0.0)
        assert ok


class TestValidateLosslessRoundtrip:
    def test_lossless(self):
        mismatch, ok = validate_lossless_roundtrip(np.array([1, 2, 3]), np.array([1, 2, 3]))
        assert ok
        assert mismatch == 0.0

    def test_lossy(self):
        mismatch, ok = validate_lossless_roundtrip(np.array([1, 2, 3]), np.array([1, 9, 3]))
        assert not ok
        assert mismatch > 0.0

    def test_shape_mismatch(self):
        mismatch, ok = validate_lossless_roundtrip(np.array([1, 2]), np.array([1]))
        assert not ok


class TestDetectDictionaryBloat:
    def test_bounded(self):
        growth, ok = detect_dictionary_bloat(np.array([10, 12, 18], dtype=float))
        assert ok
        assert growth <= 2.0

    def test_bloat(self):
        growth, ok = detect_dictionary_bloat(np.array([10, 15, 30], dtype=float))
        assert not ok
        assert growth > 2.0

    def test_short(self):
        growth, ok = detect_dictionary_bloat(np.array([10], dtype=float))
        assert ok


class TestMonitorEncodingThroughput:
    def test_fast(self):
        throughput, ok = monitor_encoding_throughput(np.array([5000, 6000]), np.array([2.0, 3.0]))
        assert ok
        assert throughput >= 1e3

    def test_slow(self):
        throughput, ok = monitor_encoding_throughput(np.array([500, 500]), np.array([10.0, 10.0]))
        assert not ok
        assert throughput < 1e3

    def test_empty(self):
        throughput, ok = monitor_encoding_throughput(np.array([]), np.array([]))
        assert ok


class TestCompressionRules:
    def _get_rules(self):
        return {r.name: r for r in CompressionExpansionRuleSet().rules()}

    def test_ratio_rule_applies(self):
        result = GraphRewriter().apply_rule(
            self._get_rules()["insert_compression_ratio_analysis_before_encode"],
            _compression_cdg(),
        )
        assert not result.is_failure
        assert "analyze_compression_ratio" in {n.matched_primitive for n in result.unwrap().nodes if n.matched_primitive}

    def test_dictionary_bloat_rule_applies(self):
        result = GraphRewriter().apply_rule(
            self._get_rules()["insert_dictionary_bloat_detection_after_encode"],
            _compression_cdg(),
        )
        assert not result.is_failure

    def test_roundtrip_rule_applies(self):
        result = GraphRewriter().apply_rule(
            self._get_rules()["insert_lossless_roundtrip_validation_after_decode"],
            _compression_cdg(),
        )
        assert not result.is_failure

    def test_throughput_rule_applies(self):
        result = GraphRewriter().apply_rule(
            self._get_rules()["insert_encoding_throughput_monitoring_after_decode"],
            _compression_cdg(),
        )
        assert not result.is_failure


class TestCompressionDiagnostics:
    def test_diagnose_ratio(self):
        diags = CompressionExpansionRuleSet().diagnose(
            _compression_cdg(),
            ExpansionContext(intermediates={"ratio_gap": 0.5}),
        )
        assert "insert_compression_ratio_analysis_before_encode" in {d.rule_name for d in diags}

    def test_diagnose_bloat(self):
        diags = CompressionExpansionRuleSet().diagnose(
            _compression_cdg(),
            ExpansionContext(intermediates={"dictionary_growth_rate": 3.0}),
        )
        assert "insert_dictionary_bloat_detection_after_encode" in {d.rule_name for d in diags}

    def test_diagnose_roundtrip(self):
        diags = CompressionExpansionRuleSet().diagnose(
            _compression_cdg(),
            ExpansionContext(intermediates={"mismatch_fraction": 0.2}),
        )
        assert "insert_lossless_roundtrip_validation_after_decode" in {d.rule_name for d in diags}

    def test_diagnose_throughput(self):
        diags = CompressionExpansionRuleSet().diagnose(
            _compression_cdg(),
            ExpansionContext(intermediates={"symbols_per_ms": 100.0}),
        )
        assert "insert_encoding_throughput_monitoring_after_decode" in {d.rule_name for d in diags}

    def test_no_data_returns_nothing(self):
        assert CompressionExpansionRuleSet().diagnose(_compression_cdg(), ExpansionContext()) == []


class TestCompressionIntegration:
    def test_full_expansion(self):
        result = ExpansionEngine([CompressionExpansionRuleSet()]).expand(
            _compression_cdg(),
            ExpansionContext(intermediates={"ratio_gap": 0.5, "mismatch_fraction": 0.2}),
        )
        assert result.expanded
