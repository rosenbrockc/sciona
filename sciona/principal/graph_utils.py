"""Small shared helpers for the Principal graph."""

from __future__ import annotations

from sciona.architect.catalog import PrimitiveCatalog
from sciona.architect.handoff import CDGExport
from sciona.architect.models import NodeStatus
from sciona.principal.structure_summary import summarize_trial_structure


def _param_signature(cdg: CDGExport) -> str:
    """Return the scoped study signature for the current structure + primitives."""
    summary = summarize_trial_structure(cdg)
    return f"{summary.get('topo_hash', '')}:{summary.get('primitive_signature', '')}"


def _structure_has_tunables(cdg: CDGExport, catalog: PrimitiveCatalog | None) -> bool:
    """Return whether any atomic node in *cdg* exposes approved tunables."""
    if catalog is None:
        return False
    for node in cdg.nodes:
        if node.status != NodeStatus.ATOMIC:
            continue
        primitive_name = str(node.matched_primitive or "").strip()
        if not primitive_name:
            continue
        primitive = catalog.get(primitive_name)
        if primitive is not None and primitive.tunable_params:
            return True
    return False
