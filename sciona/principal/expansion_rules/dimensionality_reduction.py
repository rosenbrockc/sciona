"""Expansion rules for the Dimensionality Reduction family (PCA, SVD, t-SNE, UMAP).

Dimensionality Reduction skeleton topology (3 nodes, linear):

    Center/Scale -> Project -> Validate Reconstruction

Expansion insertion points:
  - After Project: explained variance analysis
  - Before Project: crowding detection
  - After Validate Reconstruction: reconstruction error check
  - Before Validate Reconstruction: orthogonality validation
"""

from __future__ import annotations

import logging

import numpy as np

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

_DOMAIN = "dimensionality_reduction"

_CENTER_SCALE = "Center/Scale"
_PROJECT = "Project"
_VALIDATE_RECONSTRUCTION = "Validate Reconstruction"


def _node(
    node_id: str, name: str, concept_type: ConceptType, *,
    matched_primitive: str | None = None, inputs: list[IOSpec] | None = None,
    outputs: list[IOSpec] | None = None, description: str = "",
    type_signature: str = "",
) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id, name=name, description=description or name,
        concept_type=concept_type, status=NodeStatus.ATOMIC,
        matched_primitive=matched_primitive, inputs=inputs or [],
        outputs=outputs or [], type_signature=type_signature or f"{name} -> result",
    )


def _edge(
    source_id: str, target_id: str, output_name: str = "out",
    input_name: str = "in", type_desc: str = "ndarray",
) -> DependencyEdge:
    return DependencyEdge(
        source_id=source_id, target_id=target_id, output_name=output_name,
        input_name=input_name, source_type=type_desc, target_type=type_desc,
    )


# ---------------------------------------------------------------------------
# DPO rule builders
# ---------------------------------------------------------------------------


def _build_insert_explained_variance() -> RewriteRule:
    project = _node("project", _PROJECT, ConceptType.DIMENSIONALITY_REDUCTION)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[project, sink], edges=[_edge("project", "sink")])
    interface = CDGExport(nodes=[project, sink], edges=[])

    variance = _node(
        "variance", "Analyze Explained Variance", ConceptType.DIMENSIONALITY_REDUCTION,
        matched_primitive="analyze_explained_variance",
        inputs=[IOSpec(name="eigenvalues", type_desc="ndarray")],
        outputs=[IOSpec(name="cumulative_ratio", type_desc="float"),
                 IOSpec(name="is_sufficient", type_desc="bool")],
        description="Check cumulative explained variance ratio.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[project, variance, sink],
        edges=[_edge("project", "variance"), _edge("variance", "sink")],
    )

    return RewriteRule(
        name="insert_explained_variance_after_project",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"project": "project", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"project": "project", "sink": "sink"}, edge_map={}),
        priority=3,
    )


def _build_insert_crowding_detection() -> RewriteRule:
    src = _node("src", "source", ConceptType.CUSTOM)
    project = _node("project", _PROJECT, ConceptType.DIMENSIONALITY_REDUCTION)
    lhs = CDGExport(nodes=[src, project], edges=[_edge("src", "project")])
    interface = CDGExport(nodes=[src, project], edges=[])

    crowding = _node(
        "crowding", "Detect Crowding", ConceptType.DIMENSIONALITY_REDUCTION,
        matched_primitive="detect_crowding",
        inputs=[IOSpec(name="neighbor_ranks_original", type_desc="ndarray"),
                IOSpec(name="neighbor_ranks_embedded", type_desc="ndarray")],
        outputs=[IOSpec(name="trustworthiness", type_desc="float"),
                 IOSpec(name="is_trustworthy", type_desc="bool")],
        description="Measure neighbor preservation quality.",
        type_signature="ndarray, ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[src, crowding, project],
        edges=[_edge("src", "crowding"), _edge("crowding", "project")],
    )

    return RewriteRule(
        name="insert_crowding_detection_after_project",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "project": "project"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "project": "project"}, edge_map={}),
        priority=2,
    )


def _build_insert_reconstruction_error() -> RewriteRule:
    validate = _node("validate", _VALIDATE_RECONSTRUCTION, ConceptType.DIMENSIONALITY_REDUCTION)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[validate, sink], edges=[_edge("validate", "sink")])
    interface = CDGExport(nodes=[validate, sink], edges=[])

    recon = _node(
        "recon", "Check Reconstruction Error", ConceptType.DIMENSIONALITY_REDUCTION,
        matched_primitive="check_reconstruction_error",
        inputs=[IOSpec(name="X", type_desc="ndarray"),
                IOSpec(name="X_reconstructed", type_desc="ndarray")],
        outputs=[IOSpec(name="relative_error", type_desc="float"),
                 IOSpec(name="is_acceptable", type_desc="bool")],
        description="Check ||X - X_rec|| / ||X||.",
        type_signature="ndarray, ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[validate, recon, sink],
        edges=[_edge("validate", "recon"), _edge("recon", "sink")],
    )

    return RewriteRule(
        name="insert_reconstruction_error_after_validate",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"validate": "validate", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"validate": "validate", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_orthogonality_validation() -> RewriteRule:
    src = _node("src", "source", ConceptType.CUSTOM)
    validate = _node("validate", _VALIDATE_RECONSTRUCTION, ConceptType.DIMENSIONALITY_REDUCTION)
    lhs = CDGExport(nodes=[src, validate], edges=[_edge("src", "validate")])
    interface = CDGExport(nodes=[src, validate], edges=[])

    ortho = _node(
        "ortho", "Validate Orthogonality", ConceptType.DIMENSIONALITY_REDUCTION,
        matched_primitive="validate_orthogonality",
        inputs=[IOSpec(name="components", type_desc="ndarray")],
        outputs=[IOSpec(name="max_off_diagonal", type_desc="float"),
                 IOSpec(name="is_orthogonal", type_desc="bool")],
        description="Check orthogonality of projection components.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[src, ortho, validate],
        edges=[_edge("src", "ortho"), _edge("ortho", "validate")],
    )

    return RewriteRule(
        name="insert_orthogonality_validation_before_project",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "validate": "validate"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "validate": "validate"}, edge_map={}),
        priority=1,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _diagnose_explained_variance(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    ratio = intermediates.get("cumulative_variance_ratio")
    if ratio is None:
        return None
    try:
        r = float(ratio)
    except (ValueError, TypeError):
        return None
    if r < 0.95:
        return ExpansionDiagnostic(
            rule_name="insert_explained_variance_after_project",
            severity=min(1.0, (0.95 - r) / 0.95),
            evidence=f"Cumulative variance ratio {r:.4f} below 0.95 — insufficient information preserved",
            metric_name="cumulative_variance_ratio", metric_value=r, threshold=0.95,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_crowding(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    trust = intermediates.get("trustworthiness")
    if trust is None:
        return None
    try:
        t = float(trust)
    except (ValueError, TypeError):
        return None
    if t < 0.9:
        return ExpansionDiagnostic(
            rule_name="insert_crowding_detection_after_project",
            severity=min(1.0, (0.9 - t) / 0.9),
            evidence=f"Trustworthiness {t:.4f} below 0.9 — neighbor structure not preserved",
            metric_name="trustworthiness", metric_value=t, threshold=0.9,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_reconstruction_error(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    err = intermediates.get("reconstruction_relative_error")
    if err is None:
        return None
    try:
        e = float(err)
    except (ValueError, TypeError):
        return None
    if e > 0.1:
        return ExpansionDiagnostic(
            rule_name="insert_reconstruction_error_after_validate",
            severity=min(1.0, e),
            evidence=f"Reconstruction relative error {e:.4f} exceeds 0.1 — lossy projection",
            metric_name="reconstruction_relative_error", metric_value=e, threshold=0.1,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_orthogonality(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    off_diag = intermediates.get("max_off_diagonal")
    if off_diag is None:
        return None
    try:
        o = float(off_diag)
    except (ValueError, TypeError):
        return None
    if o > 1e-6:
        return ExpansionDiagnostic(
            rule_name="insert_orthogonality_validation_before_project",
            severity=min(1.0, np.log10(max(o, 1e-30)) / -6.0 + 1.0),
            evidence=f"Max off-diagonal {o:.2e} exceeds 1e-6 — components not orthogonal",
            metric_name="max_off_diagonal", metric_value=o, threshold=1e-6,
            source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class DimensionalityReductionExpansionRuleSet:
    """Expansion rules for dimensionality reduction pipelines (PCA, SVD, t-SNE, UMAP)."""

    name = "dimensionality_reduction"
    domain = "dimensionality_reduction"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_explained_variance(),
            _build_insert_crowding_detection(),
            _build_insert_reconstruction_error(),
            _build_insert_orthogonality_validation(),
        ]

    def diagnose(self, cdg: CDGExport, context: ExpansionContext) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []
        for fn in [_diagnose_explained_variance, _diagnose_crowding,
                    _diagnose_reconstruction_error, _diagnose_orthogonality]:
            d = fn(cdg, context)
            if d is not None:
                diagnostics.append(d)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)
