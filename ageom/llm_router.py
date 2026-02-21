"""Per-prompt LLM routing.

Wraps a default ``LLMClient`` with optional per-prompt overrides so that
different prompts in the pipeline can be dispatched to different providers
(e.g. local Qwen for ranking, Anthropic for decomposition).
"""

from __future__ import annotations

from typing import Any

from ageom.hunter.llm import LLMClient

# ---------------------------------------------------------------------------
# Prompt key constants — one per LLM call site
# ---------------------------------------------------------------------------

# Architect
ARCHITECT_STRATEGY = "architect_strategy"
ARCHITECT_DECOMPOSE = "architect_decompose"
ARCHITECT_CRITIQUE = "architect_critique"

# Hunter
HUNTER_SCORE = "hunter_score"
HUNTER_REFORMULATE = "hunter_reformulate"
HUNTER_ANALYZE_FAILURE = "hunter_analyze_failure"

# Synthesizer
SYNTHESIZER_REPAIR = "synthesizer_repair"
SYNTHESIZER_TACTIC = "synthesizer_tactic"

# Ingester
INGESTER_CHUNK = "ingester_chunk"
INGESTER_HOIST_STATE = "ingester_hoist_state"
INGESTER_ABSTRACT = "ingester_abstract"
INGESTER_FIX_TYPE = "ingester_fix_type"
INGESTER_FIX_GHOST = "ingester_fix_ghost"
INGESTER_OPAQUE_WITNESS = "ingester_opaque_witness"

# Orchestrator
ORCHESTRATOR_REFINE = "orchestrator_refine"

ALL_PROMPT_KEYS = [
    ARCHITECT_STRATEGY,
    ARCHITECT_DECOMPOSE,
    ARCHITECT_CRITIQUE,
    HUNTER_SCORE,
    HUNTER_REFORMULATE,
    HUNTER_ANALYZE_FAILURE,
    SYNTHESIZER_REPAIR,
    SYNTHESIZER_TACTIC,
    INGESTER_CHUNK,
    INGESTER_HOIST_STATE,
    INGESTER_ABSTRACT,
    INGESTER_FIX_TYPE,
    INGESTER_FIX_GHOST,
    INGESTER_OPAQUE_WITNESS,
    ORCHESTRATOR_REFINE,
]


# ---------------------------------------------------------------------------
# LLMRouter
# ---------------------------------------------------------------------------


class LLMRouter:
    """Routes LLM calls to per-prompt override clients or a shared default.

    Satisfies the ``LLMClient`` protocol so it can be used as a drop-in
    replacement everywhere the pipeline expects an ``LLMClient``.
    """

    def __init__(
        self,
        default: LLMClient,
        overrides: dict[str, LLMClient] | None = None,
    ) -> None:
        self._default = default
        self._overrides: dict[str, LLMClient] = overrides or {}

    def for_prompt(self, key: str) -> LLMClient:
        """Return the override client for *key*, or the default."""
        return self._overrides.get(key, self._default)

    # -- LLMClient protocol methods (delegate to default) ------------------

    async def complete(self, system: str, user: str) -> str:
        return await self._default.complete(system, user)

    async def complete_with_grammar(
        self, system: str, user: str, grammar: str
    ) -> str:
        return await self._default.complete_with_grammar(system, user, grammar)


# ---------------------------------------------------------------------------
# select_llm helper
# ---------------------------------------------------------------------------


def select_llm(llm: Any, key: str) -> LLMClient:
    """Pick the right client for *key*.

    If *llm* is an ``LLMRouter``, delegates to ``for_prompt(key)``.
    Otherwise returns *llm* unchanged (plain ``LLMClient``).
    """
    if isinstance(llm, LLMRouter):
        return llm.for_prompt(key)
    return llm
