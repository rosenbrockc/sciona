"""Expansion rules for the Compression family."""

from __future__ import annotations

import logging

from sciona.architect.graph_rewriter import Morphism, RewriteRule
from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.principal.expansion import ExpansionContext, ExpansionDiagnostic

logger = logging.getLogger(__name__)

_DOMAIN = "compression"

_MODEL_SOURCE = "Model Source"
_ENCODE = "Encode"
_DECODE_VERIFY = "Decode/Verify"


def _node(
    node_id: str,
    name: str,
    concept_type: ConceptType,
    *,
    matched_primitive: str | None = None,
    inputs: list[IOSpec] | None = None,
    outputs: list[IOSpec] | None = None,
    description: str = "",
    type_signature: str = "",
) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id,
        name=name,
        description=description or name,
        concept_type=concept_type,
        status=NodeStatus.ATOMIC,
        matched_primitive=matched_primitive,
        inputs=inputs or [],
        outputs=outputs or [],
        type_signature=type_signature or f"{name} -> result",
    )


def _edge(
    source_id: str,
    target_id: str,
    output_name: str = "out",
    input_name: str = "in",
    type_desc: str = "ndarray",
) -> DependencyEdge:
    return DependencyEdge(
        source_id=source_id,
        target_id=target_id,
        output_name=output_name,
        input_name=input_name,
        source_type=type_desc,
        target_type=type_desc,
    )


def _build_insert_compression_ratio_analysis() -> RewriteRule:
    src = _node("src", "source", ConceptType.CUSTOM)
    encode = _node("encode", _ENCODE, ConceptType.COMPRESSION)
    lhs = CDGExport(nodes=[src, encode], edges=[_edge("src", "encode")])
    interface = CDGExport(nodes=[src, encode], edges=[])

    ratio = _node(
        "ratio",
        "Analyze Compression Ratio",
        ConceptType.COMPRESSION,
        matched_primitive="analyze_compression_ratio",
        inputs=[
            IOSpec(name="original_bits", type_desc="float"),
            IOSpec(name="compressed_bits", type_desc="float"),
            IOSpec(name="entropy_bound", type_desc="float"),
        ],
        outputs=[
            IOSpec(name="ratio_gap", type_desc="float"),
            IOSpec(name="is_efficient", type_desc="bool"),
        ],
        type_signature="float, float, float -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[src, ratio, encode],
        edges=[_edge("src", "ratio"), _edge("ratio", "encode")],
    )
    return RewriteRule(
        name="insert_compression_ratio_analysis_before_encode",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "encode": "encode"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "encode": "encode"}, edge_map={}),
        priority=3,
    )


def _build_insert_dictionary_bloat_detection() -> RewriteRule:
    encode = _node("encode", _ENCODE, ConceptType.COMPRESSION)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[encode, sink], edges=[_edge("encode", "sink")])
    interface = CDGExport(nodes=[encode, sink], edges=[])

    bloat = _node(
        "bloat",
        "Detect Dictionary Bloat",
        ConceptType.COMPRESSION,
        matched_primitive="detect_dictionary_bloat",
        inputs=[IOSpec(name="dictionary_sizes", type_desc="ndarray")],
        outputs=[
            IOSpec(name="growth_rate", type_desc="float"),
            IOSpec(name="is_bounded", type_desc="bool"),
        ],
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[encode, bloat, sink],
        edges=[_edge("encode", "bloat"), _edge("bloat", "sink")],
    )
    return RewriteRule(
        name="insert_dictionary_bloat_detection_after_encode",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"encode": "encode", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"encode": "encode", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_lossless_roundtrip_validation() -> RewriteRule:
    decode = _node("decode", _DECODE_VERIFY, ConceptType.COMPRESSION)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[decode, sink], edges=[_edge("decode", "sink")])
    interface = CDGExport(nodes=[decode, sink], edges=[])

    roundtrip = _node(
        "roundtrip",
        "Validate Lossless Roundtrip",
        ConceptType.COMPRESSION,
        matched_primitive="validate_lossless_roundtrip",
        inputs=[
            IOSpec(name="original", type_desc="ndarray"),
            IOSpec(name="decoded", type_desc="ndarray"),
        ],
        outputs=[
            IOSpec(name="mismatch_fraction", type_desc="float"),
            IOSpec(name="is_lossless", type_desc="bool"),
        ],
        type_signature="ndarray, ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[decode, roundtrip, sink],
        edges=[_edge("decode", "roundtrip"), _edge("roundtrip", "sink")],
    )
    return RewriteRule(
        name="insert_lossless_roundtrip_validation_after_decode",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"decode": "decode", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"decode": "decode", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_encoding_throughput_monitoring() -> RewriteRule:
    decode = _node("decode", _DECODE_VERIFY, ConceptType.COMPRESSION)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[decode, sink], edges=[_edge("decode", "sink")])
    interface = CDGExport(nodes=[decode, sink], edges=[])

    throughput = _node(
        "throughput",
        "Monitor Encoding Throughput",
        ConceptType.COMPRESSION,
        matched_primitive="monitor_encoding_throughput",
        inputs=[
            IOSpec(name="symbol_counts", type_desc="ndarray"),
            IOSpec(name="runtimes_ms", type_desc="ndarray"),
        ],
        outputs=[
            IOSpec(name="symbols_per_ms", type_desc="float"),
            IOSpec(name="is_fast_enough", type_desc="bool"),
        ],
        type_signature="ndarray, ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[decode, throughput, sink],
        edges=[_edge("decode", "throughput"), _edge("throughput", "sink")],
    )
    return RewriteRule(
        name="insert_encoding_throughput_monitoring_after_decode",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"decode": "decode", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"decode": "decode", "sink": "sink"}, edge_map={}),
        priority=1,
    )


def _diagnose_compression_ratio(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    value = (context.intermediates or {}).get("ratio_gap")
    if value is None:
        return None
    try:
        gap = float(value)
    except (ValueError, TypeError):
        return None
    if gap > 0.2:
        return ExpansionDiagnostic(
            rule_name="insert_compression_ratio_analysis_before_encode",
            severity=max(0.35, min(1.0, gap)),
            evidence=f"Compression ratio gap {gap:.3f} exceeds 0.2.",
            metric_name="ratio_gap",
            metric_value=gap,
            threshold=0.2,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_dictionary_bloat(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    value = (context.intermediates or {}).get("dictionary_growth_rate")
    if value is None:
        value = (context.intermediates or {}).get("growth_rate")
    if value is None:
        return None
    try:
        growth = float(value)
    except (ValueError, TypeError):
        return None
    if growth > 2.0:
        return ExpansionDiagnostic(
            rule_name="insert_dictionary_bloat_detection_after_encode",
            severity=max(0.35, min(1.0, (growth - 2.0) / 3.0)),
            evidence=f"Dictionary growth rate {growth:.3f} exceeds 2.0.",
            metric_name="dictionary_growth_rate",
            metric_value=growth,
            threshold=2.0,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_lossless_roundtrip(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    value = (context.intermediates or {}).get("mismatch_fraction")
    if value is None:
        return None
    try:
        mismatch = float(value)
    except (ValueError, TypeError):
        return None
    if mismatch > 0.0:
        return ExpansionDiagnostic(
            rule_name="insert_lossless_roundtrip_validation_after_decode",
            severity=max(0.35, min(1.0, mismatch)),
            evidence=f"Mismatch fraction {mismatch:.3f} indicates lossy roundtrip.",
            metric_name="mismatch_fraction",
            metric_value=mismatch,
            threshold=0.0,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_encoding_throughput(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    value = (context.intermediates or {}).get("symbols_per_ms")
    if value is None:
        return None
    try:
        throughput = float(value)
    except (ValueError, TypeError):
        return None
    if throughput < 1e3:
        return ExpansionDiagnostic(
            rule_name="insert_encoding_throughput_monitoring_after_decode",
            severity=max(0.35, min(1.0, (1e3 - throughput) / 1e3)),
            evidence=f"Encoding throughput {throughput:.3f} is below 1e3 symbols/ms.",
            metric_name="symbols_per_ms",
            metric_value=throughput,
            threshold=1e3,
            source_domain=_DOMAIN,
        )
    return None


class CompressionExpansionRuleSet:
    """Expansion rules for compression pipelines."""

    name = "compression"
    domain = "compression"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_compression_ratio_analysis(),
            _build_insert_dictionary_bloat_detection(),
            _build_insert_lossless_roundtrip_validation(),
            _build_insert_encoding_throughput_monitoring(),
        ]

    def diagnose(self, cdg: CDGExport, context: ExpansionContext) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []
        for fn in (
            _diagnose_compression_ratio,
            _diagnose_dictionary_bloat,
            _diagnose_lossless_roundtrip,
            _diagnose_encoding_throughput,
        ):
            diagnostic = fn(cdg, context)
            if diagnostic is not None:
                diagnostics.append(diagnostic)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)
