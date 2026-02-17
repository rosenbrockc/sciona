"""Lightweight pipeline telemetry for structured event logging.

Provides a ``PipelineEvent`` dataclass and an ``EventLog`` singleton for
append-only event recording.  Events are exported as JSONL for post-hoc
analysis or debugging.

Usage::

    from ageom.telemetry import log_event

    log_event("architect", "decompose", "DECOMPOSITION_START",
              node_id="root", payload={"goal": goal})
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineEvent:
    """A single timestamped event from the pipeline."""

    timestamp: float
    round: str  # "architect", "hunter", "synthesizer"
    phase: str  # e.g. "decompose", "search", "compile"
    event_type: str  # e.g. "DECOMPOSITION_START", "MATCH_FOUND"
    node_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None


class EventLog:
    """Append-only event log with JSONL export."""

    def __init__(self) -> None:
        self._events: list[PipelineEvent] = []

    def append(self, event: PipelineEvent) -> None:
        self._events.append(event)

    @property
    def events(self) -> list[PipelineEvent]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def to_jsonl(self) -> str:
        """Serialize all events to a JSONL string."""
        lines: list[str] = []
        for ev in self._events:
            lines.append(json.dumps(asdict(ev), default=str))
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Write the event log to a JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_jsonl())

    def clear(self) -> None:
        self._events.clear()


# Module-level singleton
_global_log = EventLog()


def get_event_log() -> EventLog:
    """Return the global event log singleton."""
    return _global_log


def log_event(
    round_name: str,
    phase: str,
    event_type: str,
    *,
    node_id: str = "",
    payload: dict[str, Any] | None = None,
    duration_ms: float | None = None,
) -> PipelineEvent:
    """Append an event to the global log.

    Args:
        round_name: Pipeline round ("architect", "hunter", "synthesizer").
        phase: Sub-phase within the round.
        event_type: Event type identifier.
        node_id: Optional node identifier.
        payload: Optional dict of additional data.
        duration_ms: Optional timing measurement.

    Returns:
        The created PipelineEvent.
    """
    event = PipelineEvent(
        timestamp=time.time(),
        round=round_name,
        phase=phase,
        event_type=event_type,
        node_id=node_id,
        payload=payload or {},
        duration_ms=duration_ms,
    )
    _global_log.append(event)
    return event
