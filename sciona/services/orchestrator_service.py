"""Service wrapper around the full orchestration loop."""

from __future__ import annotations

from sciona.services.models import OrchestrationRequest


class OrchestratorService:
    """Stable service entrypoint for orchestration escalation."""

    def __init__(self, hunter_agent: object, orchestrate: object) -> None:
        self._hunter_agent = hunter_agent
        self._orchestrate = orchestrate

    async def run(self, request: OrchestrationRequest):
        return await self._orchestrate(
            request.cdg,
            hunter_agent=self._hunter_agent,
            llm=request.llm,
            prover=request.prover,
            max_rounds=request.max_rounds,
            hunter_concurrency=request.hunter_concurrency,
        )
