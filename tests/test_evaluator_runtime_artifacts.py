from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sciona.principal.evaluator import _build_runtime_artifacts


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
        signal_data={
            "signal": list(np.linspace(0.0, 1.0, 30)),
            "sampling_rate": 21.0,
            "capnostream_value": list(np.linspace(0.0, 1.0, 30)),
            "capnostream_sampling_rate": 21.0,
            "h10_ecg_value": signal,
            "ecg_sampling_rate": 100.0,
            "h10_ecg_t": list(np.linspace(0.0, 20.0, 2000)),
        },
    )

    assert artifacts["signal_data"]["signal"] is signal
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
