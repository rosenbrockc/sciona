"""Data models for the verification oracle."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CompilerFeedback:
    """Structured feedback from a proof compiler invocation."""

    raw_output: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    goals_remaining: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True if no errors and no remaining goals."""
        return len(self.errors) == 0 and len(self.goals_remaining) == 0
