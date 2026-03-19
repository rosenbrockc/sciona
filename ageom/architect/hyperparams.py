"""Hyperparameter manifest loader and built-in runtime param definitions."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ageom.architect.models import PrimitiveParamSpec

logger = logging.getLogger(__name__)


def load_hyperparams_manifest(
    manifest_path: Path,
) -> dict[str, list[PrimitiveParamSpec]]:
    """Load ageo-atoms manifest.json and return atom_name -> tunable_params mapping.

    Only returns params from atoms with status="approved" and safe_to_optimize=True.
    """
    path = Path(manifest_path)
    if not path.exists():
        logger.warning("Hyperparams manifest not found: %s", path)
        return {}

    raw = json.loads(path.read_text())
    atoms: list[dict] = raw.get("atoms", [])
    result: dict[str, list[PrimitiveParamSpec]] = {}

    for atom in atoms:
        status = atom.get("status", "")
        if status != "approved":
            continue
        atom_name = atom.get("name", "")
        if not atom_name:
            continue

        params: list[PrimitiveParamSpec] = []
        for p in atom.get("tunable_params", []):
            try:
                spec = PrimitiveParamSpec(**p)
            except Exception:
                logger.warning(
                    "Skipping invalid param %s on atom %s",
                    p.get("name", "?"),
                    atom_name,
                )
                continue
            if spec.safe_to_optimize:
                params.append(spec)

        if params:
            result[atom_name] = params

    return result


def get_runtime_signal_event_rate_params() -> dict[str, list[PrimitiveParamSpec]]:
    """Return hand-audited tunable params for the built-in signal_event_rate functions."""
    return {
        "filter_signal_for_detection": [
            PrimitiveParamSpec(
                name="filter_order",
                kind="int",
                default=4,
                min_value=2,
                max_value=8,
                step=2,
                semantic_role="Butterworth filter order",
                range_source="signal processing convention",
                source_confidence="high",
            ),
            PrimitiveParamSpec(
                name="clipping_scale",
                kind="float",
                default=8.0,
                min_value=3.0,
                max_value=15.0,
                semantic_role="Outlier clipping threshold in MAD units",
                range_source="empirical",
                source_confidence="medium",
            ),
            PrimitiveParamSpec(
                name="low_cutoff_hz",
                kind="float",
                default=3.0,
                min_value=0.5,
                max_value=10.0,
                semantic_role="Bandpass low cutoff frequency",
                range_source="physiological signal range",
                source_confidence="high",
            ),
            PrimitiveParamSpec(
                name="high_cutoff_hz",
                kind="float",
                default=25.0,
                min_value=10.0,
                max_value=50.0,
                semantic_role="Bandpass high cutoff frequency",
                range_source="physiological signal range",
                source_confidence="high",
            ),
        ],
        "detect_peaks_in_signal": [
            PrimitiveParamSpec(
                name="prominence_scale",
                kind="float",
                default=1.5,
                min_value=0.5,
                max_value=5.0,
                semantic_role="Peak prominence threshold in MAD units",
                range_source="empirical",
                source_confidence="medium",
            ),
            PrimitiveParamSpec(
                name="refractory_scale",
                kind="float",
                default=0.45,
                min_value=0.2,
                max_value=0.8,
                semantic_role="Refractory period as fraction of sampling rate",
                range_source="physiological minimum IBI",
                source_confidence="high",
            ),
        ],
        "compute_event_rate_smoothed": [
            PrimitiveParamSpec(
                name="smoothing_window",
                kind="int",
                default=5,
                min_value=1,
                max_value=15,
                step=2,
                semantic_role="Moving average window size for rate smoothing",
                range_source="empirical",
                source_confidence="medium",
            ),
        ],
    }
