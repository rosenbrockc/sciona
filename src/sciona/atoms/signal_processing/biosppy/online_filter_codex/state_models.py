"""Pydantic state model for the BioSPPy online filter wrappers."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class FilterState(BaseModel):
    """Serializable state for chunked OnlineFilter execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    b: np.ndarray = Field(...)
    a: np.ndarray = Field(...)
    zi: np.ndarray | None = Field(default=None)
