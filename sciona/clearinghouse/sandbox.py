"""Sandbox executor protocol and base infrastructure."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sciona.clearinghouse.models import SandboxPayload, SandboxResult


@runtime_checkable
class SandboxExecutor(Protocol):
    """Protocol for sandbox execution backends."""

    async def execute(self, payload: SandboxPayload) -> SandboxResult: ...


class LocalSandboxExecutor:
    """Local sandbox executor for development and testing.

    Mirrors the pattern from ``sciona.principal.evaluator.ExecutionSandbox``
    but operates on CDG payloads rather than export bundles.
    """

    def __init__(self, *, timeout_s: float = 120.0) -> None:
        self._timeout_s = timeout_s

    async def execute(self, payload: SandboxPayload) -> SandboxResult:
        """Execute a CDG payload locally (stub for Phase C).

        In production this is replaced by Lambda/SageMaker executors.
        """
        return SandboxResult(
            error="LocalSandboxExecutor is a development stub",
        )
