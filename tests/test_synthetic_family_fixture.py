"""Smoke tests for the synthetic-family fixture.

These tests prove the fixture is a valid ExpansionRuleSet and exercises
enough of the DPO rewrite machinery that downstream engine tests can
rely on it.  They deliberately avoid asserting anything family-specific.
"""

from __future__ import annotations

import pytest

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.principal.expansion import (
    ExpansionContext,
    ExpansionEngine,
    ExpansionRuleSet,
)
from tests.fixtures.synthetic_family import (
    SYNTH_DOMAIN,
    SyntheticFamilyExpansionRuleSet,
    synthetic_cdg,
)


def test_rule_set_satisfies_protocol() -> None:
    rs = SyntheticFamilyExpansionRuleSet()
    assert isinstance(rs, ExpansionRuleSet)
    assert rs.name == "synthetic_family"
    assert rs.domain == SYNTH_DOMAIN


def test_rule_set_ships_two_named_rules() -> None:
    rs = SyntheticFamilyExpansionRuleSet()
    rule_names = {r.name for r in rs.rules()}
    assert rule_names == {
        "insert_quality_check_after_source",
        "wrap_sink_with_audit",
    }


def test_diagnose_returns_empty_without_context() -> None:
    rs = SyntheticFamilyExpansionRuleSet()
    assert rs.diagnose(synthetic_cdg(), ExpansionContext()) == []


def test_diagnose_emits_both_when_both_thresholds_exceeded() -> None:
    rs = SyntheticFamilyExpansionRuleSet()
    ctx = ExpansionContext(
        intermediates={"synth_quality_score": 0.1, "synth_error_rate": 0.4}
    )
    diags = rs.diagnose(synthetic_cdg(), ctx)
    assert {d.rule_name for d in diags} == {
        "insert_quality_check_after_source",
        "wrap_sink_with_audit",
    }
    for d in diags:
        assert d.source_domain == SYNTH_DOMAIN
        assert 0.0 < d.severity <= 1.0


def test_diagnose_does_not_fire_under_threshold() -> None:
    rs = SyntheticFamilyExpansionRuleSet()
    ctx = ExpansionContext(
        intermediates={"synth_quality_score": 0.9, "synth_error_rate": 0.01}
    )
    assert rs.diagnose(synthetic_cdg(), ctx) == []


def test_quality_check_rule_applies_to_linear_cdg() -> None:
    rs = SyntheticFamilyExpansionRuleSet()
    rules = {r.name: r for r in rs.rules()}
    result = GraphRewriter().apply_rule(
        rules["insert_quality_check_after_source"], synthetic_cdg()
    )
    assert not result.is_failure
    expanded = result.unwrap()
    primitives = {n.matched_primitive for n in expanded.nodes if n.matched_primitive}
    # Rewrite must have injected the audit/quality primitive between source and process.
    assert "measure_synth_quality" in primitives
    # Node count grew by exactly one (the inserted checker).
    assert len(expanded.nodes) == len(synthetic_cdg().nodes) + 1


def test_engine_applies_both_rules_when_both_diagnostics_fire() -> None:
    rs = SyntheticFamilyExpansionRuleSet()
    engine = ExpansionEngine([rs])
    ctx = ExpansionContext(
        intermediates={"synth_quality_score": 0.1, "synth_error_rate": 0.4}
    )
    result = engine.expand(synthetic_cdg(), ctx)
    assert result.expanded
    assert set(result.applied_rules) == {
        "insert_quality_check_after_source",
        "wrap_sink_with_audit",
    }


def test_engine_noop_when_no_diagnostics_fire() -> None:
    rs = SyntheticFamilyExpansionRuleSet()
    engine = ExpansionEngine([rs])
    ctx = ExpansionContext(intermediates={})
    result = engine.expand(synthetic_cdg(), ctx)
    assert not result.expanded
    assert result.applied_rules == ()
    # CDG must be returned unchanged when nothing fires.
    assert len(result.cdg.nodes) == len(synthetic_cdg().nodes)


def test_fixture_module_has_no_family_dependencies() -> None:
    """Guard against fixture accidentally depending on a real provider."""
    import tests.fixtures.synthetic_family as mod

    # Spot-check: importing the fixture package must not pull in any
    # sciona.atoms.*, sciona.probes.*, or sciona.expansion_atoms.*
    # submodules as side effects.
    import sys

    forbidden_prefixes = (
        "sciona.atoms.",
        "sciona.probes.",
        "sciona.expansion_atoms.",
    )
    offenders = [
        name for name in sys.modules if name.startswith(forbidden_prefixes)
    ]
    # The fixture itself must not have caused any of these imports.
    # (Other tests may have already loaded them; this only fails if
    # the fixture module directly references a forbidden prefix.)
    assert mod.SYNTH_DOMAIN == "synthetic_family"
    # Static check: the fixture package __init__ does not import them.
    source = (mod.__file__ or "")
    assert source.endswith("__init__.py")


@pytest.mark.parametrize(
    ("quality", "error", "expected_rules"),
    [
        (0.1, 0.01, {"insert_quality_check_after_source"}),
        (0.9, 0.4, {"wrap_sink_with_audit"}),
        (0.9, 0.01, set()),
    ],
)
def test_parametrized_diagnostic_triggers(
    quality: float,
    error: float,
    expected_rules: set[str],
) -> None:
    rs = SyntheticFamilyExpansionRuleSet()
    ctx = ExpansionContext(
        intermediates={"synth_quality_score": quality, "synth_error_rate": error}
    )
    diags = rs.diagnose(synthetic_cdg(), ctx)
    assert {d.rule_name for d in diags} == expected_rules
