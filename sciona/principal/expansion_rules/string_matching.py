"""Expansion rules for the String Matching family (KMP, Naive, Rabin-Karp).

Defines DPO rules and diagnostic functions that let the expansion engine
insert alphabet analysis, pattern-text ratio checks, hash collision
detection, and failure function validation into string matching CDGs.

String matching skeleton topology (3 nodes, linear pipeline):

    Preprocess → Scan → Match/Advance

Expansion insertion points:
  - Before Preprocess: alphabet size analysis, pattern-text ratio check
  - After Scan: hash collision detection
  - After Preprocess: failure function validation

All diagnostics are pure functions of string matching intermediates.
"""

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
from sciona.principal.expansion import (
    ExpansionContext,
    ExpansionDiagnostic,
)

logger = logging.getLogger(__name__)

_DOMAIN = "string_matching"

# String matching skeleton node names
_PREPROCESS = "Preprocess"
_SCAN = "Scan"
_MATCH_ADVANCE = "Match/Advance"


# ---------------------------------------------------------------------------
# Node / edge helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# DPO rule builders
# ---------------------------------------------------------------------------


def _build_insert_alphabet_analysis() -> RewriteRule:
    """Interpose ``analyze_alphabet_size`` before Preprocess.

    Alphabet size affects preprocessing table size and hash collision
    probability, guiding algorithm selection.
    """
    src = _node("src", "source", ConceptType.CUSTOM)
    preprocess = _node(
        "preprocess",
        _PREPROCESS,
        ConceptType.STRING_MATCHING,
    )
    lhs = CDGExport(nodes=[src, preprocess], edges=[_edge("src", "preprocess")])
    interface = CDGExport(nodes=[src, preprocess], edges=[])

    alphabet = _node(
        "alphabet",
        "Analyze Alphabet Size",
        ConceptType.STRING_MATCHING,
        matched_primitive="analyze_alphabet_size",
        inputs=[
            IOSpec(name="text", type_desc="ndarray"),
            IOSpec(name="pattern", type_desc="ndarray"),
        ],
        outputs=[
            IOSpec(name="text_alphabet_size", type_desc="int"),
            IOSpec(name="pattern_alphabet_size", type_desc="int"),
            IOSpec(name="overlap_ratio", type_desc="float"),
        ],
        description="Analyze alphabet sizes for algorithm selection guidance.",
        type_signature="ndarray, ndarray -> tuple[int, int, float]",
    )
    rhs = CDGExport(
        nodes=[src, alphabet, preprocess],
        edges=[
            _edge("src", "alphabet"),
            _edge("alphabet", "preprocess"),
        ],
    )

    return RewriteRule(
        name="insert_alphabet_analysis_before_preprocess",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "preprocess": "preprocess"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "preprocess": "preprocess"}, edge_map={}),
        priority=2,
    )


def _build_insert_pattern_text_ratio_check() -> RewriteRule:
    """Interpose ``check_pattern_text_ratio`` before Preprocess.

    Pathological length ratios (pattern > text, or very short pattern
    in very long text) need different algorithmic strategies.
    """
    src = _node("src", "source", ConceptType.CUSTOM)
    preprocess = _node(
        "preprocess",
        _PREPROCESS,
        ConceptType.STRING_MATCHING,
    )
    lhs = CDGExport(nodes=[src, preprocess], edges=[_edge("src", "preprocess")])
    interface = CDGExport(nodes=[src, preprocess], edges=[])

    ratio_check = _node(
        "ratio_check",
        "Check Pattern-Text Ratio",
        ConceptType.STRING_MATCHING,
        matched_primitive="check_pattern_text_ratio",
        inputs=[
            IOSpec(name="pattern_length", type_desc="int"),
            IOSpec(name="text_length", type_desc="int"),
        ],
        outputs=[
            IOSpec(name="ratio", type_desc="float"),
            IOSpec(name="assessment", type_desc="str"),
        ],
        description="Check the ratio of pattern length to text length.",
        type_signature="int, int -> tuple[float, str]",
    )
    rhs = CDGExport(
        nodes=[src, ratio_check, preprocess],
        edges=[
            _edge("src", "ratio_check"),
            _edge("ratio_check", "preprocess"),
        ],
    )

    return RewriteRule(
        name="insert_pattern_text_ratio_check_before_preprocess",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "preprocess": "preprocess"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "preprocess": "preprocess"}, edge_map={}),
        priority=3,
    )


def _build_insert_hash_collision_detection() -> RewriteRule:
    """Interpose ``measure_hash_collision_rate`` after Scan.

    Frequent hash collisions in Rabin-Karp degrade performance to O(nm).
    """
    scan = _node(
        "scan",
        _SCAN,
        ConceptType.STRING_MATCHING,
    )
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[scan, sink], edges=[_edge("scan", "sink")])
    interface = CDGExport(nodes=[scan, sink], edges=[])

    collision = _node(
        "collision",
        "Measure Hash Collision Rate",
        ConceptType.STRING_MATCHING,
        matched_primitive="measure_hash_collision_rate",
        inputs=[
            IOSpec(name="n_hash_matches", type_desc="int"),
            IOSpec(name="n_true_matches", type_desc="int"),
        ],
        outputs=[
            IOSpec(name="collision_rate", type_desc="float"),
            IOSpec(name="is_excessive", type_desc="bool"),
        ],
        description="Measure the spurious match rate for Rabin-Karp style algorithms.",
        type_signature="int, int -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[scan, collision, sink],
        edges=[
            _edge("scan", "collision"),
            _edge("collision", "sink"),
        ],
    )

    return RewriteRule(
        name="insert_hash_collision_detection_after_scan",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"scan": "scan", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"scan": "scan", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_failure_function_validation() -> RewriteRule:
    """Interpose ``validate_failure_function`` after Preprocess.

    A malformed failure table causes KMP to skip valid matches or loop.
    """
    preprocess = _node(
        "preprocess",
        _PREPROCESS,
        ConceptType.STRING_MATCHING,
    )
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[preprocess, sink], edges=[_edge("preprocess", "sink")])
    interface = CDGExport(nodes=[preprocess, sink], edges=[])

    validate = _node(
        "validate",
        "Validate Failure Function",
        ConceptType.STRING_MATCHING,
        matched_primitive="validate_failure_function",
        inputs=[
            IOSpec(name="failure_table", type_desc="ndarray"),
            IOSpec(name="pattern_length", type_desc="int"),
        ],
        outputs=[
            IOSpec(name="n_violations", type_desc="int"),
            IOSpec(name="is_valid", type_desc="bool"),
        ],
        description="Validate basic properties of a KMP failure function table.",
        type_signature="ndarray, int -> tuple[int, bool]",
    )
    rhs = CDGExport(
        nodes=[preprocess, validate, sink],
        edges=[
            _edge("preprocess", "validate"),
            _edge("validate", "sink"),
        ],
    )

    return RewriteRule(
        name="insert_failure_function_validation_after_preprocess",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"preprocess": "preprocess", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"preprocess": "preprocess", "sink": "sink"}, edge_map={}),
        priority=1,
    )


# ---------------------------------------------------------------------------
# Diagnostics (pure, deterministic)
# ---------------------------------------------------------------------------


def _diagnose_alphabet_overlap(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect low alphabet overlap (pattern uses chars not in text)."""
    intermediates = context.intermediates or {}
    overlap_ratio = intermediates.get("alphabet_overlap_ratio")

    if overlap_ratio is None:
        return None

    try:
        ratio = float(overlap_ratio)
    except (ValueError, TypeError):
        return None

    if ratio < 1.0:
        return ExpansionDiagnostic(
            rule_name="insert_alphabet_analysis_before_preprocess",
            severity=min(1.0, 1.0 - ratio),
            evidence=(
                f"Alphabet overlap ratio {ratio:.2f} is below 1.0 "
                f"— pattern contains characters not in text"
            ),
            metric_name="alphabet_overlap_ratio",
            metric_value=ratio,
            threshold=1.0,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_pattern_text_ratio(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect pathological pattern-to-text length ratio."""
    intermediates = context.intermediates or {}
    pattern_length = intermediates.get("pattern_length")
    text_length = intermediates.get("text_length")

    if pattern_length is None or text_length is None:
        return None

    try:
        p = int(pattern_length)
        t = int(text_length)
    except (ValueError, TypeError):
        return None

    if t == 0:
        return None

    ratio = p / t

    if p > t:
        return ExpansionDiagnostic(
            rule_name="insert_pattern_text_ratio_check_before_preprocess",
            severity=1.0,
            evidence=(
                f"Pattern length {p} exceeds text length {t} "
                f"— no match is possible"
            ),
            metric_name="pattern_text_ratio",
            metric_value=ratio,
            threshold=1.0,
            source_domain=_DOMAIN,
        )
    elif ratio < 0.01 and t > 100:
        return ExpansionDiagnostic(
            rule_name="insert_pattern_text_ratio_check_before_preprocess",
            severity=min(1.0, (0.01 - ratio) / 0.01),
            evidence=(
                f"Pattern-text ratio {ratio:.4f} is very low "
                f"— consider multi-pattern algorithm"
            ),
            metric_name="pattern_text_ratio",
            metric_value=ratio,
            threshold=0.01,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_hash_collisions(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect high hash collision rate."""
    intermediates = context.intermediates or {}
    collision_rate = intermediates.get("hash_collision_rate")

    if collision_rate is None:
        return None

    try:
        rate = float(collision_rate)
    except (ValueError, TypeError):
        return None

    if rate > 0.5:
        return ExpansionDiagnostic(
            rule_name="insert_hash_collision_detection_after_scan",
            severity=min(1.0, (rate - 0.5) / 0.5),
            evidence=(
                f"Hash collision rate {rate:.2f} exceeds 0.5 threshold "
                f"— Rabin-Karp is degrading to O(nm)"
            ),
            metric_name="hash_collision_rate",
            metric_value=rate,
            threshold=0.5,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_failure_function(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect malformed failure function table."""
    intermediates = context.intermediates or {}
    n_violations = intermediates.get("failure_function_violations")

    if n_violations is None:
        return None

    try:
        violations = int(n_violations)
    except (ValueError, TypeError):
        return None

    if violations > 0:
        return ExpansionDiagnostic(
            rule_name="insert_failure_function_validation_after_preprocess",
            severity=min(1.0, violations / 5.0),
            evidence=(
                f"{violations} failure function violation(s) detected "
                f"— KMP table may produce incorrect matches"
            ),
            metric_name="failure_function_violations",
            metric_value=float(violations),
            threshold=0.0,
            source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class StringMatchingExpansionRuleSet:
    """Expansion rules for string matching pipelines (KMP, Naive, Rabin-Karp)."""

    name = "string_matching"
    domain = "string_matching"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_alphabet_analysis(),
            _build_insert_pattern_text_ratio_check(),
            _build_insert_hash_collision_detection(),
            _build_insert_failure_function_validation(),
        ]

    def diagnose(
        self,
        cdg: CDGExport,
        context: ExpansionContext,
    ) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []

        alpha = _diagnose_alphabet_overlap(cdg, context)
        if alpha is not None:
            diagnostics.append(alpha)

        ratio = _diagnose_pattern_text_ratio(cdg, context)
        if ratio is not None:
            diagnostics.append(ratio)

        collision = _diagnose_hash_collisions(cdg, context)
        if collision is not None:
            diagnostics.append(collision)

        failure = _diagnose_failure_function(cdg, context)
        if failure is not None:
            diagnostics.append(failure)

        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)
