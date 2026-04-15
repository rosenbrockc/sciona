"""Synthetic family fixture for matcher engine-level tests.

This module provides a minimal, fake algorithmic family used to exercise
family-agnostic machinery (ExpansionEngine, GraphRewriter, PrincipalState,
diagnostic-driven expansion, etc.) without depending on any real provider
repository's atoms, probes, or rule sets.

The synthetic family is intentionally trivial:

    source ──► process ──► sink

Two diagnostics / rules are provided:

* ``insert_quality_check_after_source`` — fires when the runtime reports
  ``synth_quality_score`` below 0.5. Interposes a ``measure_quality`` node
  on the source→process edge.
* ``wrap_sink_with_audit`` — fires when ``synth_error_rate`` exceeds 0.1.
  Interposes an ``audit_sink`` node on the process→sink edge.

Nothing here imports from :mod:`sciona.atoms`, :mod:`sciona.probes`, or
:mod:`sciona.expansion_atoms`. The only matcher symbols used are the core
type system (architect models, graph_rewriter, expansion protocol).
"""

from __future__ import annotations

from tests.fixtures.synthetic_family.cdg import (
    synthetic_cdg,
    synthetic_node,
    synthetic_edge,
)
from tests.fixtures.synthetic_family.rule_set import (
    SYNTH_DOMAIN,
    SyntheticFamilyExpansionRuleSet,
    build_insert_quality_check_rule,
    build_wrap_sink_with_audit_rule,
)

__all__ = [
    "SYNTH_DOMAIN",
    "SyntheticFamilyExpansionRuleSet",
    "build_insert_quality_check_rule",
    "build_wrap_sink_with_audit_rule",
    "synthetic_cdg",
    "synthetic_edge",
    "synthetic_node",
]
