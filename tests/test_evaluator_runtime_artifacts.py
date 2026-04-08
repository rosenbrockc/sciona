from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sciona.principal.evaluator import (
    _build_runtime_artifacts,
    _collect_runtime_inputs_from_frames,
)


def test_build_runtime_artifacts_persists_canonical_runtime_evidence(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    signal = np.sin(np.linspace(0.0, 20.0, 2000))

    artifacts = _build_runtime_artifacts(
        trace_path=trace_path,
        stdout_payload={
            "intermediates": {"rpeaks": [100.0, 350.0, 600.0]},
            "outputs": {"heart_rate": [70.0, 71.0, 69.5]},
        },
        runtime_inputs={
            "signal": list(np.linspace(0.0, 1.0, 30)),
            "sampling_rate": 21.0,
            "capnostream_value": list(np.linspace(0.0, 1.0, 30)),
            "capnostream_sampling_rate": 21.0,
            "h10_ecg_value": signal,
            "ecg_sampling_rate": 100.0,
            "h10_ecg_t": list(np.linspace(0.0, 20.0, 2000)),
        },
    )

    assert artifacts["runtime_inputs"]["h10_ecg_value"] is signal
    assert artifacts["signal_data"]["h10_ecg_value"] is signal
    assert artifacts["runtime_inputs"]["sampling_rate"] == 100.0
    assert artifacts["signal_data"]["sampling_rate"] == 100.0
    assert artifacts["intermediates"]["events"] == [100.0, 350.0, 600.0]
    assert (
        artifacts["canonical_runtime_context"]["canonical_inputs"]["sampling_rate"]["stream_id"]
        == "ecg"
    )
    assert artifacts["telemetry_summary"]["events"]["source_key"] == "rpeaks"
    assert artifacts["telemetry_summary"]["outputs"]["heart_rate"]["mean"] > 0.0

    evidence_path = tmp_path / "runtime_evidence.json"
    assert evidence_path.exists()
    persisted = json.loads(evidence_path.read_text())
    assert persisted["runtime_context"]["canonical_inputs"]["signal"] == "h10_ecg_value"
    assert persisted["telemetry_summary"]["events"]["source_key"] == "rpeaks"


def test_collect_runtime_inputs_from_frames_prefers_primary_waveform_stream() -> None:
    class _Series:
        def __init__(self, values):
            self._values = np.asarray(values)

        def to_numpy(self):
            return self._values

    class _Frame:
        def __init__(self, columns: dict[str, np.ndarray]) -> None:
            self._columns = {name: _Series(values) for name, values in columns.items()}
            self.columns = list(columns.keys())

        def __getitem__(self, key: str):
            return self._columns[key]

    frames = {
        "capnostream": _Frame(
            {
                "capnostream_t": np.linspace(0.0, 10.0, 210),
                "capnostream_value": np.linspace(0.0, 1.0, 210),
            }
        ),
        "h10_ecg": _Frame(
            {
                "h10_ecg_t": np.linspace(0.0, 10.0, 1300),
                "h10_ecg_value": np.sin(np.linspace(0.0, 30.0, 1300)),
            }
        ),
    }

    runtime_inputs = _collect_runtime_inputs_from_frames(frames)

    assert "signal" in runtime_inputs
    assert "sampling_rate" in runtime_inputs
    assert runtime_inputs["signal"].shape == runtime_inputs["h10_ecg_value"].shape
    assert runtime_inputs["sampling_rate"] > 100.0
