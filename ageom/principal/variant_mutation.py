"""Variant-family mutation support for Principal trial updates."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ageom.architect.handoff import CDGExport
from ageom.architect.models import NodeStatus
from ageom.signal_event_rate_registry import (
    SIGNAL_EVENT_RATE_DECLARATIONS,
    next_signal_event_rate_variant,
)

if TYPE_CHECKING:
    from ageom.architect.catalog import PrimitiveCatalog
    from ageom.principal.atom_ledger import AtomLedger

logger = logging.getLogger(__name__)


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


class LedgerVariantFamily:
    """Universal fallback that uses UCB1 bandit rankings to select atom variants."""

    name = "ledger_bandit"

    def __init__(self, ledger: AtomLedger, catalog: PrimitiveCatalog) -> None:
        self._ledger = ledger
        self._catalog = catalog

    def matches(self, cdg: CDGExport) -> bool:
        return True

    def mutate(
        self,
        cdg: CDGExport,
        *,
        bottleneck_name: str | None,
    ) -> VariantMutationResult:
        from ageom.principal.atom_ledger import compute_slot_signature

        if not bottleneck_name:
            return VariantMutationResult(cdg=cdg, applied=False)

        node_map = {n.node_id: n for n in cdg.nodes}
        target = None
        for node in cdg.nodes:
            if node.status == NodeStatus.ATOMIC and node.name == bottleneck_name:
                target = node
                break

        if target is None or not target.matched_primitive:
            return VariantMutationResult(cdg=cdg, applied=False)

        parent = node_map.get(target.parent_id) if target.parent_id else None
        slot = compute_slot_signature(target, parent)

        candidates = [
            p.name for p in self._catalog.search_by_category(target.concept_type)
        ]
        if not candidates:
            return VariantMutationResult(cdg=cdg, applied=False)
        if target.matched_primitive not in candidates:
            candidates.append(target.matched_primitive)

        ranked = self._ledger.rank_candidates(slot, candidates)
        if not ranked:
            return VariantMutationResult(cdg=cdg, applied=False)

        best_name, _best_score = ranked[0]
        if best_name == target.matched_primitive:
            return VariantMutationResult(
                cdg=cdg, applied=False, family=self.name, allow_redecompose=True
            )

        updated_nodes = []
        for node in cdg.nodes:
            if node.node_id == target.node_id:
                updated_nodes.append(
                    node.model_copy(update={"matched_primitive": best_name})
                )
            else:
                updated_nodes.append(node)

        logger.info(
            "Ledger bandit: swapping '%s' -> '%s' for '%s'",
            target.matched_primitive,
            best_name,
            bottleneck_name,
        )
        return VariantMutationResult(
            cdg=cdg.model_copy(update={"nodes": updated_nodes}),
            applied=True,
            family=self.name,
            variant_name=best_name,
            allow_redecompose=True,
        )


def maybe_apply_bottleneck_variant(
    cdg: CDGExport,
    *,
    bottleneck_name: str | None,
    atom_ledger: AtomLedger | None = None,
    catalog: PrimitiveCatalog | None = None,
) -> VariantMutationResult:
    """Apply the first matching family-specific mutation for the bottleneck node.

    Families are tried in order.  A successful mutation (``applied=True``)
    short-circuits immediately.  An exhausted family (``applied=False``)
    is recorded but does **not** block later families — this lets the
    ledger bandit try after a curated family runs out of variants.
    """
    if not bottleneck_name:
        return VariantMutationResult(cdg=cdg, applied=False)

    families: list[VariantFamily] = list(VARIANT_FAMILIES)
    if atom_ledger is not None and catalog is not None:
        families.append(LedgerVariantFamily(atom_ledger, catalog))

    last_exhausted: VariantMutationResult | None = None
    for family in families:
        if not family.matches(cdg):
            continue
        result = family.mutate(cdg, bottleneck_name=bottleneck_name)
        if result.applied:
            return result
        # Family matched but had nothing to apply — remember it but keep
        # trying remaining families.  Later families override earlier ones
        # so that e.g. the ledger's allow_redecompose=True supersedes a
        # curated family's allow_redecompose=False.
        if result.family is not None:
            last_exhausted = result

    if last_exhausted is not None:
        return last_exhausted
    return VariantMutationResult(cdg=cdg, applied=False)
