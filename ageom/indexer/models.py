"""Data models for the semantic index."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from ageom.types import Declaration, Prover


@dataclass
class IndexEntry:
    """A declaration paired with its embedding vector."""

    declaration: Declaration
    embedding: np.ndarray  # shape (768,), L2-normalized
    source_text: str = ""  # the text that was embedded

    def __post_init__(self) -> None:
        if self.embedding.ndim != 1:
            raise ValueError(
                f"Expected 1-d embedding, got shape {self.embedding.shape}"
            )


@dataclass
class IndexMetadata:
    """Metadata about a built index."""

    num_entries: int
    prover: Prover
    source_lib: str
    embedding_model: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
