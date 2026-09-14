"""CHAMPS final per-coupling trimmed ensemble.

Selection and reduction derived from Bosch's MIT-licensed predictor.py at
4a42e18b5b88043fb40ec15289216a1d88789698; see docs/licenses/CHAMPS-MIT.txt.
Record alignment and strict validation are adapter additions. Inputs are
unscaled scalar predictions, not normalized training targets.
"""
from collections.abc import Mapping

import numpy as np

MODEL_ORDER = ("model_H", "model_I", "model_L", "model_A", "model_F",
               "model_B", "model_C", "model_M", "model_G", "model_E",
               "model_D", "model_J", "model_K")
COUPLING_TYPES = ("1JHC", "1JHN", "2JHC", "2JHH", "2JHN", "3JHC", "3JHH", "3JHN")
MODEL_MASK = (
    (1, 1, 1, 1, 1, 1, 1, 1),
    (1, 1, 1, 1, 1, 1, 1, 1),
    (1, 0, 1, 0, 0, 1, 1, 0),
    (1, 1, 1, 1, 1, 1, 1, 1),
    (0, 1, 0, 1, 1, 0, 1, 1),
    (1, 1, 1, 1, 1, 0, 0, 1),
    (1, 1, 1, 1, 1, 1, 1, 1),
    (0, 0, 0, 0, 1, 0, 0, 0),
    (0, 0, 0, 1, 1, 0, 0, 1),
    (0, 0, 0, 0, 0, 1, 0, 0),
    (1, 1, 1, 0, 0, 1, 1, 0),
    (1, 1, 1, 1, 0, 1, 1, 1),
    (1, 1, 1, 1, 1, 1, 1, 1),
)


def blend_predictions(predictions: Mapping[str, Mapping[str, float]],
                      coupling_types: Mapping[str, str]) -> dict[str, float]:
    """Align all thirteen models by opaque record key and average central five.

Returns keys in coupling_types iteration order. Missing/extra models or records,
unknown coupling types, boolean predictions and nonfinite values are rejected.
Empty input is accepted only when every model supplies an empty mapping.
"""
    if not isinstance(predictions, Mapping) or set(predictions) != set(MODEL_ORDER):
        raise ValueError("Exactly thirteen configured model variants are required")
    if not isinstance(coupling_types, Mapping):
        raise ValueError("Coupling types must be a mapping")
    keys = tuple(coupling_types)
    if any(not isinstance(k, str) or not k for k in keys):
        raise ValueError("Record keys must be nonempty strings")
    if any(not isinstance(t, str) or t not in COUPLING_TYPES for t in coupling_types.values()):
        raise ValueError("Unknown coupling type")
    rows = []
    for model in MODEL_ORDER:
        values = predictions[model]
        if not isinstance(values, Mapping) or set(values) != set(keys):
            raise ValueError("Each model must predict exactly the requested records")
        row = [values[k] for k in keys]
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating))
               for v in row):
            raise ValueError("Predictions must be real numbers")
        try:
            array = np.asarray(row, dtype=np.float64)
        except (OverflowError, ValueError) as exc:
            raise ValueError("Predictions must fit finite float64") from exc
        if not np.isfinite(array).all():
            raise ValueError("Predictions must be finite")
        rows.append(array)
    matrix = np.stack(rows)
    mask = np.asarray(MODEL_MASK, dtype=bool)
    output = np.empty(len(keys), dtype=np.float64)
    for index, coupling_type in enumerate(COUPLING_TYPES):
        columns = np.asarray([coupling_types[k] == coupling_type for k in keys])
        if not columns.any():
            continue
        selected = np.sort(matrix[mask[:, index]][:, columns], axis=0)
        start = (selected.shape[0] - 5) // 2
        with np.errstate(over="ignore", invalid="ignore"):
            output[columns] = selected[start:start + 5].mean(axis=0)
    if not np.isfinite(output).all():
        raise ValueError("Ensemble reduction overflowed")
    return {k: float(v) for k, v in zip(keys, output)}
