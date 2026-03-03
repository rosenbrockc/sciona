"""Hunter agent dependencies injected into the graph."""

from __future__ import annotations

from dataclasses import dataclass

from ageom.hunter.llm import LLMClient
from ageom.protocols import SemanticIndex, VerificationOracle
from ageom.shared_context import SharedContextStore


@dataclass
class HunterDeps:
    """External dependencies for the Hunter graph nodes."""

    index: SemanticIndex
    oracle: VerificationOracle
    llm: LLMClient
    shared_context: SharedContextStore | None = None
