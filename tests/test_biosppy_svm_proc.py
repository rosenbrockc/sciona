from __future__ import annotations

import importlib

import numpy as np
import biosppy.biometrics as biometrics

from sciona.atoms.signal_processing.biosppy.svm_proc import cross_validation, get_subject_results
from sciona.probes.signal_processing.biosppy_svm_proc import probe_records


def test_biosppy_svm_proc_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.signal_processing.biosppy.svm_proc")
    probes = importlib.import_module("sciona.probes.signal_processing.biosppy_svm_proc")
    assert hasattr(atoms, "get_auth_rates")
    assert hasattr(atoms, "cross_validation")
    assert hasattr(probes, "SVM_PROC_PROBE_TARGETS")


def test_probe_records_resolve_to_live_symbols() -> None:
    for record in probe_records():
        module = importlib.import_module(str(record["module_import_path"]))
        assert hasattr(module, str(record["wrapper_symbol"]))


def test_cross_validation_matches_upstream_first_split() -> None:
    labels = np.array([0, 0, 1, 1, 1, 0], dtype=int)

    wrapped_split = next(cross_validation(labels, n_iter=2, test_size=0.33, train_size=None, random_state=7))
    upstream_split = next(
        biometrics.cross_validation(
            labels=labels,
            n_iter=2,
            test_size=0.33,
            train_size=None,
            random_state=7,
        )[0]
    )

    assert np.array_equal(wrapped_split[0], upstream_split[0])
    assert np.array_equal(wrapped_split[1], upstream_split[1])


def test_get_subject_results_accepts_hashable_subject_labels() -> None:
    result = get_subject_results(
        results={"authentication": {"foo": 1}, "identification": {"bar": 2}},
        subject="subject-1",
        thresholds=np.array([0.2, 0.5], dtype=float),
        subjects=["subject-1"],
        subject_dict={"subject-1": 0},
        subject_idx=[0],
    )

    assert result["subject"] == "subject-1"
    assert set(result.keys()) == {"authentication", "identification", "subject", "thresholds"}
