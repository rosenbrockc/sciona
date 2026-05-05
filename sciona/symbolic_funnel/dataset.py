"""Empirical dataset model for the heuristic funnel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sciona.ghost.dimensions import DimensionalSignature


@dataclass(frozen=True)
class ColumnMetadata:
    """Metadata for a single dataset column."""

    name: str
    dim_signature: DimensionalSignature | None = None
    is_positive: bool = False


@dataclass
class EmpiricalDataset:
    """Wraps an (N, D) NumPy array with column metadata for funnel stages.

    Parameters
    ----------
    data : NDArray
        Shape ``(N, D)`` array of observations.
    columns : list[ColumnMetadata]
        One entry per column, in the same order as ``data`` columns.
    """

    data: NDArray[np.floating[Any]]
    columns: list[ColumnMetadata]

    # Lazily computed column stats.
    _col_index: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)
        if self.data.shape[1] != len(self.columns):
            msg = (
                f"Data has {self.data.shape[1]} columns but "
                f"{len(self.columns)} column metadata entries were provided"
            )
            raise ValueError(msg)
        self._col_index = {col.name: i for i, col in enumerate(self.columns)}
        # Auto-detect positivity.
        for i, col in enumerate(self.columns):
            if not col.is_positive:
                col_data = self.data[:, i]
                finite = col_data[np.isfinite(col_data)]
                if len(finite) > 0 and np.all(finite > 0):
                    object.__setattr__(col, "is_positive", True)

    @property
    def n_rows(self) -> int:
        return self.data.shape[0]

    @property
    def n_cols(self) -> int:
        return self.data.shape[1]

    @property
    def column_names(self) -> list[str]:
        return [col.name for col in self.columns]

    def column_by_name(self, name: str) -> NDArray[np.floating[Any]]:
        """Return a 1-D array for the named column."""
        idx = self._col_index.get(name)
        if idx is None:
            msg = f"Column {name!r} not found. Available: {self.column_names}"
            raise KeyError(msg)
        return self.data[:, idx]

    def column_min(self, name: str) -> float:
        return float(np.nanmin(self.column_by_name(name)))

    def column_max(self, name: str) -> float:
        return float(np.nanmax(self.column_by_name(name)))

    def log_transform(self) -> tuple[EmpiricalDataset, list[str]]:
        """Return a new dataset with log-transformed positive columns.

        Returns the new dataset and the list of column names that were
        successfully log-transformed.  Columns with non-positive values
        are dropped.
        """
        log_cols: list[ColumnMetadata] = []
        log_data_cols: list[NDArray[np.floating[Any]]] = []
        included_names: list[str] = []
        for i, col in enumerate(self.columns):
            col_data = self.data[:, i]
            finite = col_data[np.isfinite(col_data)]
            if len(finite) > 0 and np.all(finite > 0):
                log_cols.append(
                    ColumnMetadata(
                        name=col.name,
                        dim_signature=col.dim_signature,
                        is_positive=True,
                    )
                )
                log_data_cols.append(np.log(col_data))
                included_names.append(col.name)
        if not log_data_cols:
            return (
                EmpiricalDataset(
                    data=np.empty((self.n_rows, 0)),
                    columns=[],
                ),
                [],
            )
        return (
            EmpiricalDataset(
                data=np.column_stack(log_data_cols),
                columns=log_cols,
            ),
            included_names,
        )
