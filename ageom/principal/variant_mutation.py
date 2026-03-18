"""Variant-family mutation support for Principal trial updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ageom.architect.handoff import CDGExport
from ageom.architect.models import NodeStatus
from ageom.signal_event_rate_registry import (
    SIGNAL_EVENT_RATE_DECLARATIONS,
    next_signal_event_rate_variant,
)


@dataclass(frozen=True)
class VariantMutationResult:
    """Outcome of attempting an in-place family-specific mutation."""

    cdg: CDGExport
    applied: bool
    family: str | None = None
    variant_name: str | None = None
    allow_redecompose: bool = True


class VariantFamily(Protocol):
    """Pluggable family that can detect and mutate compatible CDGs."""

    name: str

    def matches(self, cdg: CDGExport) -> bool:
        """Return whether this family owns the given CDG."""

    def mutate(
        self,
        cdg: CDGExport,
        *,
        bottleneck_name: str | None,
    ) -> VariantMutationResult:
        """Return a mutated CDG when a valid family-specific variant exists."""


class SignalEventRateVariantFamily:
    """Curated variants for signal -> event -> rate pipelines."""

    name = "signal_event_rate"

    def matches(self, cdg: CDGExport) -> bool:
        atomic_nodes = [node for node in cdg.nodes if node.status == NodeStatus.ATOMIC]
        if not atomic_nodes:
            return False
        return all(
            str(node.matched_primitive or "") in SIGNAL_EVENT_RATE_DECLARATIONS
            for node in atomic_nodes
        )

    def mutate(
        self,
        cdg: CDGExport,
        *,
        bottleneck_name: str | None,
    ) -> VariantMutationResult:
        if not bottleneck_name or not self.matches(cdg):
            return VariantMutationResult(cdg=cdg, applied=False)

        updated_nodes = []
        variant_name: str | None = None
        changed = False
        for node in cdg.nodes:
            if node.status != NodeStatus.ATOMIC or node.name != bottleneck_name:
                updated_nodes.append(node)
                continue
            current = str(node.matched_primitive or "")
            candidate = next_signal_event_rate_variant(current)
            if not candidate:
                updated_nodes.append(node)
                continue
            updated_nodes.append(
                node.model_copy(update={"matched_primitive": candidate})
            )
            changed = True
            variant_name = candidate

        if not changed:
            return VariantMutationResult(
                cdg=cdg,
                applied=False,
                family=self.name,
                allow_redecompose=False,
            )

        return VariantMutationResult(
            cdg=cdg.model_copy(update={"nodes": updated_nodes}),
            applied=True,
            family=self.name,
            variant_name=variant_name,
            allow_redecompose=False,
        )


VARIANT_FAMILIES: tuple[VariantFamily, ...] = (
    SignalEventRateVariantFamily(),
)


def maybe_apply_bottleneck_variant(
    cdg: CDGExport,
    *,
    bottleneck_name: str | None,
) -> VariantMutationResult:
    """Apply the first matching family-specific mutation for the bottleneck node."""
    if not bottleneck_name:
        return VariantMutationResult(cdg=cdg, applied=False)

    for family in VARIANT_FAMILIES:
        if not family.matches(cdg):
            continue
        result = family.mutate(cdg, bottleneck_name=bottleneck_name)
        if result.applied or result.family is not None:
            return result
    return VariantMutationResult(cdg=cdg, applied=False)
