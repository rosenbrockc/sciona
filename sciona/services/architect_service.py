"""Service wrapper around the Architect decomposition agent."""

from __future__ import annotations

from sciona.services.models import ArchitectDecomposeRequest, ArchitectDecomposeResult


class ArchitectService:
    """Stable service entrypoint for decomposition operations."""

    def __init__(self, agent: object) -> None:
        self._agent = agent

    async def decompose(
        self,
        request: ArchitectDecomposeRequest,
    ) -> ArchitectDecomposeResult:
        cdg = await self._agent.decompose(
            request.goal,
            thread_id=request.thread_id,
        )
        return ArchitectDecomposeResult(goal=request.goal, cdg=cdg)
