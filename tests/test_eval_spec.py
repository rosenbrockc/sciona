from __future__ import annotations

import json

import numpy as np

from sciona.principal.eval_spec import (
    compute_evaluation_payload,
    load_evaluation_spec,
)


def test_load_evaluation_spec_inline_json():
    spec = load_evaluation_spec('{"loss":"rmse"}')
    assert spec == {"loss": "rmse"}


def test_compute_evaluation_payload_time_aligned_index_predictions():
    flat_inputs = {
        "signal_t": np.array([10.0, 11.0, 12.0, 13.0]),
        "ref_t": np.array([10.0, 11.0, 12.0, 13.0]),
        "ref_value": np.array([60.0, 70.0, 80.0, 90.0]),
    }
    result = (
        np.array([1, 3]),
        np.array([72.0, 86.0]),
    )
    spec = {
        "loss": "rmse",
        "prediction": {
            "value_output": 1,
            "time_output": 0,
            "time_kind": "index",
            "time_source": "signal_t",
        },
        "reference": {
            "value_source": "ref_value",
            "time_source": "ref_t",
        },
    }

    payload = compute_evaluation_payload(result, flat_inputs, spec)

    assert payload["n_eval_samples"] == 2.0
    assert payload["rmse"] == np.sqrt(10.0)
    assert payload["loss"] == payload["rmse"]
    assert payload["mse"] == 10.0


def test_compute_evaluation_payload_sequence_aligned_without_times():
    flat_inputs = {"ref_value": np.array([1.0, 3.0, 5.0])}
    result = np.array([2.0, 4.0, 6.0])
    spec = {
        "loss": "mae",
        "prediction": {},
        "reference": {"value_source": "ref_value"},
    }

    payload = compute_evaluation_payload(result, flat_inputs, spec)

    assert payload["mae"] == 1.0
    assert payload["loss"] == 1.0
