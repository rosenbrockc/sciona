"""Per-prompt LLM routing.

Wraps a default ``LLMClient`` with optional per-prompt overrides so that
different prompts in the pipeline can be dispatched to different providers
(e.g. local Qwen for ranking, Anthropic for decomposition).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ageom.hunter.llm import LLMClient
from ageom.telemetry import (
    finish_prompt_dispatch,
    get_current_run_id,
    get_current_stage,
    log_event,
    start_prompt_dispatch,
    update_stage,
)

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
INGESTER_FIX_MESSAGE_CYCLE = "ingester_fix_message_cycle"
INGESTER_DECOMPOSE = "ingester_decompose"

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
    INGESTER_FIX_MESSAGE_CYCLE,
    INGESTER_DECOMPOSE,
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

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self._default.complete_with_grammar(system, user, grammar)


# ---------------------------------------------------------------------------
# Prompt-key wrapper for telemetry
# ---------------------------------------------------------------------------


class PromptKeyLLMClient:
    """LLMClient wrapper that attributes dispatches to one prompt key."""

    def __init__(self, base: LLMClient, prompt_key: str) -> None:
        self._base = base
        self._prompt_key = prompt_key
        self._heartbeat_interval_sec = 5.0

    async def _heartbeat_loop(self, dispatch_id: str, started_at: float) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_interval_sec)
            elapsed = max(0.0, time.time() - started_at)
            stage = get_current_stage()
            if stage:
                update_stage(
                    stage=stage,
                    status="running",
                    message=f"waiting on {self._prompt_key} ({elapsed:.0f}s)",
                )
            log_event(
                "llm",
                phase=stage or "prompt_dispatch",
                event_type="PROMPT_DISPATCH_HEARTBEAT",
                stage=stage,
                prompt_key=self._prompt_key,
                dispatch_id=dispatch_id,
                payload={"elapsed_sec": round(elapsed, 2)},
            )

    async def _stop_heartbeat(
        self, task: asyncio.Task[None] | None
    ) -> None:
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def complete(self, system: str, user: str) -> str:
        dispatch_id = start_prompt_dispatch(self._prompt_key, client=self._base)
        started_at = time.time()
        heartbeat_task = (
            asyncio.create_task(self._heartbeat_loop(dispatch_id, started_at))
            if dispatch_id
            else None
        )
        try:
            output = await self._base.complete(system, user)
        except Exception as exc:
            await self._stop_heartbeat(heartbeat_task)
            finish_prompt_dispatch(dispatch_id, ok=False, error=str(exc))
            raise
        await self._stop_heartbeat(heartbeat_task)
        finish_prompt_dispatch(dispatch_id, ok=True)
        return output

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        dispatch_id = start_prompt_dispatch(self._prompt_key, client=self._base)
        started_at = time.time()
        heartbeat_task = (
            asyncio.create_task(self._heartbeat_loop(dispatch_id, started_at))
            if dispatch_id
            else None
        )
        try:
            output = await self._base.complete_with_grammar(system, user, grammar)
        except Exception as exc:
            await self._stop_heartbeat(heartbeat_task)
            finish_prompt_dispatch(dispatch_id, ok=False, error=str(exc))
            raise
        await self._stop_heartbeat(heartbeat_task)
        finish_prompt_dispatch(dispatch_id, ok=True)
        return output


# ---------------------------------------------------------------------------
# select_llm helper
# ---------------------------------------------------------------------------


def select_llm(llm: Any, key: str) -> LLMClient:
    """Pick the right client for *key*.

    If *llm* is an ``LLMRouter``, delegates to ``for_prompt(key)``.
    Otherwise returns *llm* unchanged (plain ``LLMClient``).
    """
    base = llm.for_prompt(key) if isinstance(llm, LLMRouter) else llm

    # Preserve historical behavior outside telemetry run scope.
    # This keeps identity semantics used across tests/callers.
    if not get_current_run_id():
        return base

    if isinstance(base, PromptKeyLLMClient):
        if base._prompt_key == key:
            return base
        # Avoid nested wrappers on prompt-key switches.
        return PromptKeyLLMClient(base._base, key)
    return PromptKeyLLMClient(base, key)
