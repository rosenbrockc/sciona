"""State model for the BioSPPy online filter v2 wrappers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class FilterState:
    """Serializable state for chunked OnlineFilter execution."""

    b: np.ndarray
    a: np.ndarray
    zi: np.ndarray | None = None
