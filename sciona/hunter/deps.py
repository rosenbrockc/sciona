"""Hunter agent dependencies injected into the graph."""

from __future__ import annotations

from dataclasses import dataclass

from sciona.architect.catalog import PrimitiveCatalog
from sciona.hunter.llm import LLMClient
from sciona.protocols import SemanticIndex, VerificationOracle
from sciona.shared_context import SharedContextMetrics, SharedContextStore


@dataclass
class HunterDeps:
    """External dependencies for the Hunter graph nodes."""

    index: SemanticIndex
    oracle: VerificationOracle
    llm: LLMClient
    live_catalog: PrimitiveCatalog | None = None
    shared_context: SharedContextStore | None = None
    shared_context_metrics: SharedContextMetrics | None = None
