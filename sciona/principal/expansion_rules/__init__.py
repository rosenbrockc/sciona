"""Built-in expansion rule sets.

Each module in this package implements an :class:`ExpansionRuleSet` for a
specific algorithmic domain.  The :func:`default_rule_sets` factory returns
all shipped rule sets so the :class:`ExpansionEngine` can be constructed
with a single call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sciona.principal.expansion import ExpansionRuleSet


def default_rule_sets() -> list[ExpansionRuleSet]:
    """Return all built-in expansion rule sets."""
    from sciona.principal.expansion_rules.mcmc import (
        MCMCExpansionRuleSet,
    )
    from sciona.principal.expansion_rules.sequential_filter import (
        SequentialFilterExpansionRuleSet,
    )
    from sciona.principal.expansion_rules.signal_event_rate import (
        SignalEventRateExpansionRuleSet,
    )

    return [
        SignalEventRateExpansionRuleSet(),
        SequentialFilterExpansionRuleSet(),
        MCMCExpansionRuleSet(),
    ]
