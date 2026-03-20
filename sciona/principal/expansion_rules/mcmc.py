"""Expansion rules for the MCMC/HMC family.

Defines DPO rules and diagnostic functions that let the expansion engine
insert divergence detection, step size adaptation, mass matrix estimation,
and convergence diagnostics into HMC CDGs.

HMC skeleton topology (6 nodes, leapfrog integrator):

    Init -> Half Step P1 -> Full Step Q -> Oracle Query -> Half Step P2 -> Accept

Expansion insertion points:
  - After Acceptance Criterion: divergence detection, convergence diagnostics
  - Before Half Step Momentum Start: step size adaptation
  - Before Full Step Position: mass matrix estimation

All diagnostics are pure functions of HMC intermediates.
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

_DOMAIN = "mcmc"

# HMC skeleton node names (as they appear in the skeleton topology)
_INIT = "Initialization Subgraph"
_HALF_STEP_P1 = "Half Step Momentum Start"
_FULL_STEP_Q = "Full Step Position"
_ORACLE_QUERY = "Oracle Query"
_HALF_STEP_P2 = "Half Step Momentum End"
_ACCEPTANCE = "Acceptance Criterion"


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


def _build_insert_divergence_detection() -> RewriteRule:
    """Interpose ``detect_divergent_transitions`` after Acceptance Criterion.

    Detects when |H_proposed - H_initial| exceeds a threshold, indicating
    the leapfrog integrator has failed due to too-large step size.
    """
    accept = _node(
        "accept",
        _ACCEPTANCE,
        ConceptType.MCMC_KERNEL,
    )
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[accept, sink], edges=[_edge("accept", "sink")])
    interface = CDGExport(nodes=[accept, sink], edges=[])

    divergence = _node(
        "divergence",
        "Detect Divergent Transitions",
        ConceptType.MCMC_KERNEL,
        matched_primitive="detect_divergent_transitions",
        inputs=[
            IOSpec(name="energies_initial", type_desc="ndarray"),
            IOSpec(name="energies_proposed", type_desc="ndarray"),
            IOSpec(name="threshold", type_desc="float"),
        ],
        outputs=[
            IOSpec(name="energy_errors", type_desc="ndarray"),
            IOSpec(name="divergence_mask", type_desc="ndarray"),
        ],
        description="Detect divergent transitions via energy conservation violation.",
        type_signature="ndarray, ndarray, float -> tuple[ndarray, ndarray]",
    )
    rhs = CDGExport(
        nodes=[accept, divergence, sink],
        edges=[
            _edge("accept", "divergence"),
            _edge("divergence", "sink"),
        ],
    )

    return RewriteRule(
        name="insert_divergence_detection_after_accept",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"accept": "accept", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"accept": "accept", "sink": "sink"}, edge_map={}),
        priority=3,
    )


def _build_insert_step_size_adaptation() -> RewriteRule:
    """Interpose ``compute_dual_averaging_step_size`` before Half Step Momentum Start.

    Adapts epsilon via Nesterov dual averaging when the acceptance rate
    is outside the optimal [0.55, 0.85] range.
    """
    src = _node("src", "source", ConceptType.CUSTOM)
    half_step = _node(
        "half_step",
        _HALF_STEP_P1,
        ConceptType.MCMC_KERNEL,
    )
    lhs = CDGExport(nodes=[src, half_step], edges=[_edge("src", "half_step")])
    interface = CDGExport(nodes=[src, half_step], edges=[])

    adapt_eps = _node(
        "adapt_eps",
        "Adapt Step Size",
        ConceptType.MCMC_KERNEL,
        matched_primitive="compute_dual_averaging_step_size",
        inputs=[
            IOSpec(name="accept_probs", type_desc="ndarray"),
            IOSpec(name="target_accept", type_desc="float"),
            IOSpec(name="epsilon_0", type_desc="float"),
        ],
        outputs=[IOSpec(name="adapted_epsilon", type_desc="float")],
        description="Nesterov dual averaging for HMC step size adaptation.",
        type_signature="ndarray, float, float, float, float, float -> float",
    )
    rhs = CDGExport(
        nodes=[src, adapt_eps, half_step],
        edges=[
            _edge("src", "adapt_eps"),
            _edge("adapt_eps", "half_step"),
        ],
    )

    return RewriteRule(
        name="insert_step_size_adaptation_before_leapfrog",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "half_step": "half_step"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "half_step": "half_step"}, edge_map={}),
        priority=3,
    )


def _build_insert_mass_matrix_estimation() -> RewriteRule:
    """Interpose ``estimate_mass_matrix`` before Full Step Position.

    Estimates the mass matrix from warmup samples when parameter scales
    vary widely (max_std/min_std > 10).
    """
    src = _node("src", "source", ConceptType.CUSTOM)
    full_step = _node(
        "full_step",
        _FULL_STEP_Q,
        ConceptType.MCMC_KERNEL,
    )
    lhs = CDGExport(nodes=[src, full_step], edges=[_edge("src", "full_step")])
    interface = CDGExport(nodes=[src, full_step], edges=[])

    mass_est = _node(
        "mass_est",
        "Estimate Mass Matrix",
        ConceptType.MCMC_KERNEL,
        matched_primitive="estimate_mass_matrix",
        inputs=[
            IOSpec(name="samples", type_desc="ndarray"),
            IOSpec(name="diagonal_only", type_desc="bool"),
        ],
        outputs=[IOSpec(name="M_estimated", type_desc="ndarray")],
        description="Estimate mass matrix M from warmup samples.",
        type_signature="ndarray, bool -> ndarray",
    )
    rhs = CDGExport(
        nodes=[src, mass_est, full_step],
        edges=[
            _edge("src", "mass_est"),
            _edge("mass_est", "full_step"),
        ],
    )

    return RewriteRule(
        name="insert_mass_matrix_estimation_before_leapfrog",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "full_step": "full_step"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "full_step": "full_step"}, edge_map={}),
        priority=2,
    )


def _build_insert_convergence_diagnostics() -> RewriteRule:
    """Interpose ``compute_convergence_diagnostics`` after Acceptance Criterion.

    Computes split R-hat and bulk ESS across chains. Fires when max R-hat > 1.01.
    """
    accept = _node(
        "accept",
        _ACCEPTANCE,
        ConceptType.MCMC_KERNEL,
    )
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[accept, sink], edges=[_edge("accept", "sink")])
    interface = CDGExport(nodes=[accept, sink], edges=[])

    convergence = _node(
        "convergence",
        "Convergence Diagnostics",
        ConceptType.MCMC_KERNEL,
        matched_primitive="compute_convergence_diagnostics",
        inputs=[IOSpec(name="chains", type_desc="ndarray")],
        outputs=[
            IOSpec(name="rhat", type_desc="ndarray"),
            IOSpec(name="ess", type_desc="ndarray"),
        ],
        description="Compute split R-hat and bulk ESS across chains.",
        type_signature="ndarray -> tuple[ndarray, ndarray]",
    )
    rhs = CDGExport(
        nodes=[accept, convergence, sink],
        edges=[
            _edge("accept", "convergence"),
            _edge("convergence", "sink"),
        ],
    )

    return RewriteRule(
        name="insert_convergence_diagnostics_after_accept",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"accept": "accept", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"accept": "accept", "sink": "sink"}, edge_map={}),
        priority=1,
    )


# ---------------------------------------------------------------------------
# Diagnostics (pure, deterministic)
# ---------------------------------------------------------------------------


def _diagnose_divergent_transitions(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect divergent transitions from energy conservation violations."""
    intermediates = context.intermediates or {}
    energies_initial = intermediates.get("energies_initial")
    energies_proposed = intermediates.get("energies_proposed")

    if energies_initial is None or energies_proposed is None:
        return None

    try:
        energies_initial = np.asarray(energies_initial, dtype=np.float64)
        energies_proposed = np.asarray(energies_proposed, dtype=np.float64)
    except (ValueError, TypeError):
        return None

    n = len(energies_initial)
    if n == 0:
        return None

    energy_errors = np.abs(energies_proposed - energies_initial)
    divergent_frac = float(np.mean(energy_errors > 1000.0))

    if divergent_frac > 0.0:
        return ExpansionDiagnostic(
            rule_name="insert_divergence_detection_after_accept",
            severity=min(1.0, divergent_frac / 0.1),
            evidence=(
                f"{divergent_frac:.1%} of transitions divergent "
                f"(|delta_H| > 1000)"
            ),
            metric_name="divergent_transition_fraction",
            metric_value=divergent_frac,
            threshold=0.0,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_acceptance_rate(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect suboptimal acceptance rate requiring step size adaptation."""
    intermediates = context.intermediates or {}
    accept_probs = intermediates.get("accept_probs")

    if accept_probs is None:
        return None

    try:
        accept_probs = np.asarray(accept_probs, dtype=np.float64)
    except (ValueError, TypeError):
        return None

    if len(accept_probs) == 0:
        return None

    mean_accept = float(np.mean(accept_probs))

    if mean_accept < 0.55 or mean_accept > 0.85:
        distance = max(0.55 - mean_accept, mean_accept - 0.85)
        return ExpansionDiagnostic(
            rule_name="insert_step_size_adaptation_before_leapfrog",
            severity=min(1.0, distance / 0.3),
            evidence=(
                f"Mean acceptance probability {mean_accept:.3f} "
                f"outside optimal [0.55, 0.85] range"
            ),
            metric_name="mean_acceptance_probability",
            metric_value=mean_accept,
            threshold=0.55 if mean_accept < 0.55 else 0.85,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_parameter_scale_variance(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect widely varying parameter scales requiring mass matrix adaptation."""
    intermediates = context.intermediates or {}
    samples = intermediates.get("samples")

    if samples is None:
        return None

    try:
        samples = np.asarray(samples, dtype=np.float64)
    except (ValueError, TypeError):
        return None

    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    if samples.shape[0] < 2 or samples.shape[1] < 2:
        return None

    marginal_std = np.std(samples, axis=0, ddof=1)
    min_std = float(np.min(marginal_std))
    max_std = float(np.max(marginal_std))

    if min_std < 1e-15:
        return None

    scale_ratio = max_std / min_std

    if scale_ratio > 10.0:
        return ExpansionDiagnostic(
            rule_name="insert_mass_matrix_estimation_before_leapfrog",
            severity=min(1.0, scale_ratio / 100.0),
            evidence=(
                f"Parameter scale ratio {scale_ratio:.1f} "
                f"(max_std={max_std:.3f}, min_std={min_std:.3f}) exceeds threshold 10"
            ),
            metric_name="parameter_scale_ratio",
            metric_value=scale_ratio,
            threshold=10.0,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_convergence(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect non-convergence via R-hat exceeding threshold."""
    intermediates = context.intermediates or {}
    chains = intermediates.get("chains")

    if chains is None:
        return None

    try:
        chains = np.asarray(chains, dtype=np.float64)
    except (ValueError, TypeError):
        return None

    if chains.ndim < 2:
        return None

    if chains.ndim == 2:
        chains = chains[np.newaxis, :, :]

    n_chains, n_samples, n_params = chains.shape
    if n_samples < 4:
        return None

    # Compute split R-hat inline (lightweight version for diagnostic)
    half = n_samples // 2
    split = np.concatenate(
        [chains[:, :half, :], chains[:, half : 2 * half, :]],
        axis=0,
    )
    m = split.shape[0]
    n = half

    max_rhat = 1.0
    for p in range(n_params):
        chain_means = split[:, :, p].mean(axis=1)
        chain_vars = split[:, :, p].var(axis=1, ddof=1)
        B = n * np.var(chain_means, ddof=1)
        W = np.mean(chain_vars)
        if W < 1e-15:
            continue
        var_hat = ((n - 1) / n) * W + B / n
        rhat = np.sqrt(var_hat / W)
        max_rhat = max(max_rhat, float(rhat))

    if max_rhat > 1.01:
        return ExpansionDiagnostic(
            rule_name="insert_convergence_diagnostics_after_accept",
            severity=min(1.0, (max_rhat - 1.0) / 0.1),
            evidence=(
                f"Max R-hat={max_rhat:.4f} exceeds 1.01 threshold "
                f"across {n_params} parameters"
            ),
            metric_name="max_rhat",
            metric_value=max_rhat,
            threshold=1.01,
            source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class MCMCExpansionRuleSet:
    """Expansion rules for MCMC/HMC sampler pipelines."""

    name = "mcmc"
    domain = "mcmc"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_divergence_detection(),
            _build_insert_step_size_adaptation(),
            _build_insert_mass_matrix_estimation(),
            _build_insert_convergence_diagnostics(),
        ]

    def diagnose(
        self,
        cdg: CDGExport,
        context: ExpansionContext,
    ) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []

        div = _diagnose_divergent_transitions(cdg, context)
        if div is not None:
            diagnostics.append(div)

        accept = _diagnose_acceptance_rate(cdg, context)
        if accept is not None:
            diagnostics.append(accept)

        scale = _diagnose_parameter_scale_variance(cdg, context)
        if scale is not None:
            diagnostics.append(scale)

        conv = _diagnose_convergence(cdg, context)
        if conv is not None:
            diagnostics.append(conv)

        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)
