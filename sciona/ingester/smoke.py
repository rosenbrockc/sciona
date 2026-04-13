"""Deterministic ingest-time smoke validation.

This module keeps probe coverage intentionally narrow. The goal is to catch
obviously bad generated outputs for a small allowlisted subset, not to replay
the full external audit stack inside the matcher.
"""

from __future__ import annotations

import importlib
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

SMOKE_STATUS_PASS = "pass"
SMOKE_STATUS_FAIL = "fail"
SMOKE_STATUS_NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class SmokeResult:
    status: str
    target_symbol: str
    probe_id: str
    message: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "target_symbol": self.target_symbol,
            "probe_id": self.probe_id,
            "message": self.message,
            "details": dict(self.details),
        }


ProbeRunner = Callable[[Callable[..., Any]], dict[str, Any]]


@dataclass(frozen=True)
class SmokeProbe:
    probe_id: str
    target_symbol: str
    runner: ProbeRunner
    package_basenames: tuple[str, ...] = ()

    def matches(self, *, package_basename: str, target_symbol: str) -> bool:
        if self.target_symbol != target_symbol:
            return False
        if self.package_basenames and package_basename not in self.package_basenames:
            return False
        return True


def _detail_case(
    case_id: str,
    *,
    status: str,
    message: str,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "case_id": case_id,
        "status": status,
        "message": message,
    }
    payload.update(extra)
    return payload


def _repr_value(value: Any) -> str:
    return repr(value)


def _run_probe_case(
    case_id: str,
    fn: Callable[..., Any],
    *,
    args: Iterable[Any] = (),
    kwargs: dict[str, Any] | None = None,
    validator: Callable[[Any], tuple[bool, str, dict[str, Any]]] | None = None,
    expect_exception: bool = False,
) -> dict[str, Any]:
    kwargs = kwargs or {}
    try:
        result = fn(*tuple(args), **kwargs)
    except Exception as exc:
        if expect_exception:
            return _detail_case(
                case_id,
                status=SMOKE_STATUS_PASS,
                message="probe raised on the negative path as expected",
                exception=repr(exc),
            )
        return _detail_case(
            case_id,
            status=SMOKE_STATUS_FAIL,
            message="probe raised unexpectedly",
            exception=repr(exc),
        )

    if expect_exception:
        return _detail_case(
            case_id,
            status=SMOKE_STATUS_FAIL,
            message="negative-path probe did not raise",
            observed=_repr_value(result),
        )

    if validator is None:
        return _detail_case(
            case_id,
            status=SMOKE_STATUS_PASS,
            message="positive-path probe completed",
            observed=_repr_value(result),
        )

    ok, message, extra = validator(result)
    return _detail_case(
        case_id,
        status=SMOKE_STATUS_PASS if ok else SMOKE_STATUS_FAIL,
        message=message,
        **extra,
    )


def _compile_probe_result(
    probe_id: str,
    target_symbol: str,
    *,
    positive_case: dict[str, Any],
    negative_case: dict[str, Any],
) -> dict[str, Any]:
    status = SMOKE_STATUS_PASS
    if positive_case["status"] == SMOKE_STATUS_FAIL:
        status = SMOKE_STATUS_FAIL
    if negative_case["status"] == SMOKE_STATUS_FAIL:
        status = SMOKE_STATUS_FAIL
    message = "allowlisted smoke probe passed"
    if status == SMOKE_STATUS_FAIL:
        failing_case = positive_case if positive_case["status"] == SMOKE_STATUS_FAIL else negative_case
        message = failing_case["message"]
    return {
        "status": status,
        "probe_id": probe_id,
        "target_symbol": target_symbol,
        "message": message,
        "details": {
            "positive_case": positive_case,
            "negative_case": negative_case,
        },
    }


def _validate_patch_array(result: Any) -> tuple[bool, str, dict[str, Any]]:
    import numpy as np

    array = np.asarray(result)
    ok = array.ndim >= 3 and tuple(array.shape[-2:]) == (2, 2) and array.shape[0] > 0
    return (
        ok,
        "positive-path image patches look structurally valid"
        if ok
        else "expected a non-empty patch tensor with 2x2 patches",
        {
            "observed_shape": list(array.shape),
            "observed_dtype": str(array.dtype),
        },
    )


def _validate_image_shape(expected_shape: tuple[int, ...]) -> Callable[[Any], tuple[bool, str, dict[str, Any]]]:
    def _validator(result: Any) -> tuple[bool, str, dict[str, Any]]:
        import numpy as np

        array = np.asarray(result)
        observed_shape = tuple(int(dim) for dim in array.shape)
        ok = observed_shape == expected_shape
        return (
            ok,
            f"positive-path reconstruction returned shape {expected_shape}"
            if ok
            else f"expected reconstructed shape {expected_shape}, got {observed_shape}",
            {
                "observed_shape": list(observed_shape),
                "observed_dtype": str(array.dtype),
            },
        )

    return _validator


def _validate_square_shape(expected_nodes: int) -> Callable[[Any], tuple[bool, str, dict[str, Any]]]:
    def _validator(result: Any) -> tuple[bool, str, dict[str, Any]]:
        observed_shape = tuple(int(dim) for dim in getattr(result, "shape", ()))
        ok = observed_shape == (expected_nodes, expected_nodes)
        return (
            ok,
            f"positive-path graph shape is {expected_nodes}x{expected_nodes}"
            if ok
            else f"expected graph shape {(expected_nodes, expected_nodes)}, got {observed_shape}",
            {
                "observed_shape": list(observed_shape),
                "observed_type": type(result).__name__,
            },
        )

    return _validator


def _validate_fft_output(result: Any) -> tuple[bool, str, dict[str, Any]]:
    import numpy as np

    array = np.asarray(result)
    ok = array.shape == (4,) and np.iscomplexobj(array)
    return (
        ok,
        "positive-path FFT output has the expected shape and complex dtype"
        if ok
        else "expected a length-4 complex FFT result",
        {
            "observed_shape": list(array.shape),
            "observed_dtype": str(array.dtype),
        },
    )


def _validate_array_shape(expected_shape: tuple[int, ...]) -> Callable[[Any], tuple[bool, str, dict[str, Any]]]:
    def _validator(result: Any) -> tuple[bool, str, dict[str, Any]]:
        import numpy as np

        array = np.asarray(result)
        observed_shape = tuple(int(dim) for dim in array.shape)
        ok = observed_shape == expected_shape
        return (
            ok,
            f"positive-path result shape is {expected_shape}"
            if ok
            else f"expected result shape {expected_shape}, got {observed_shape}",
            {
                "observed_shape": list(observed_shape),
                "observed_dtype": str(array.dtype),
            },
        )

    return _validator


def _validate_scalar_result(result: Any) -> tuple[bool, str, dict[str, Any]]:
    import numpy as np

    array = np.asarray(result)
    ok = array.shape == ()
    return (
        ok,
        "positive-path scalar result looks structurally valid"
        if ok
        else f"expected a scalar result, got shape {array.shape}",
        {"observed_shape": list(array.shape), "observed_dtype": str(array.dtype)},
    )


def _validate_tuple_result(expected_length: int) -> Callable[[Any], tuple[bool, str, dict[str, Any]]]:
    def _validator(result: Any) -> tuple[bool, str, dict[str, Any]]:
        ok = isinstance(result, tuple) and len(result) == expected_length
        return (
            ok,
            f"positive-path tuple result has length {expected_length}"
            if ok
            else f"expected a tuple of length {expected_length}",
            {
                "observed_type": type(result).__name__,
                "observed_length": len(result) if isinstance(result, tuple) else None,
            },
        )

    return _validator


def _validate_optimize_vector(expected_length: int) -> Callable[[Any], tuple[bool, str, dict[str, Any]]]:
    def _validator(result: Any) -> tuple[bool, str, dict[str, Any]]:
        import numpy as np

        x = getattr(result, "x", None)
        if x is None:
            return (
                False,
                "expected an optimize result with x",
                {"observed_type": type(result).__name__},
            )
        array = np.asarray(x)
        observed_shape = tuple(int(dim) for dim in array.shape)
        ok = observed_shape == (expected_length,)
        return (
            ok,
            f"positive-path optimize result vector has length {expected_length}"
            if ok
            else f"expected optimize result vector length {expected_length}, got {observed_shape}",
            {
                "observed_shape": list(observed_shape),
                "observed_dtype": str(array.dtype),
                "observed_success": getattr(result, "success", None),
            },
        )

    return _validator


def _validate_curve_fit_result(expected_n_params: int) -> Callable[[Any], tuple[bool, str, dict[str, Any]]]:
    def _validator(result: Any) -> tuple[bool, str, dict[str, Any]]:
        import numpy as np

        if not isinstance(result, tuple) or len(result) != 2:
            return (
                False,
                "expected curve_fit to return (popt, pcov)",
                {"observed_type": type(result).__name__},
            )
        popt, pcov = result
        popt_arr = np.asarray(popt)
        pcov_arr = np.asarray(pcov)
        ok = popt_arr.shape == (expected_n_params,) and pcov_arr.shape == (expected_n_params, expected_n_params)
        return (
            ok,
            f"positive-path curve_fit result has {expected_n_params} fitted parameters"
            if ok
            else f"expected curve_fit result shapes {(expected_n_params,)}, {(expected_n_params, expected_n_params)}",
            {
                "observed_popt_shape": list(popt_arr.shape),
                "observed_pcov_shape": list(pcov_arr.shape),
                "observed_popt_dtype": str(popt_arr.dtype),
                "observed_pcov_dtype": str(pcov_arr.dtype),
            },
        )

    return _validator


def _validate_numeric_offset(result: Any) -> tuple[bool, str, dict[str, Any]]:
    try:
        value = float(result)
    except Exception:
        return (
            False,
            "expected a numeric leap-second offset result",
            {"observed_type": type(result).__name__},
        )

    ok = value > 32.0
    return (
        ok,
        "positive-path offset result looks numerically valid"
        if ok
        else "expected a plausible UTC/TAI offset greater than 32 seconds",
        {"observed_value": value},
    )


def _validate_monotonic_index_array(
    *,
    allow_empty: bool = False,
    max_value: int | None = None,
) -> Callable[[Any], tuple[bool, str, dict[str, Any]]]:
    def _validator(result: Any) -> tuple[bool, str, dict[str, Any]]:
        import numpy as np

        array = np.asarray(result)
        if array.ndim != 1:
            return (
                False,
                "expected a one-dimensional index array",
                {
                    "observed_shape": list(array.shape),
                    "observed_dtype": str(array.dtype),
                },
            )

        length = int(array.shape[0])
        monotonic = length <= 1 or bool(np.all(np.diff(array) >= 0))
        nonempty = allow_empty or length > 0
        within_bounds = True
        if max_value is not None and length > 0:
            within_bounds = bool(np.min(array) >= 0 and np.max(array) <= max_value)

        ok = monotonic and nonempty and within_bounds
        if ok:
            if allow_empty:
                message = "positive-path onset indices look structurally valid"
            else:
                message = "positive-path peak indices look structurally valid"
        elif not nonempty:
            message = "expected a non-empty monotonic index array"
        elif not monotonic:
            message = "expected monotonic index output"
        else:
            message = f"expected index output within [0, {max_value}]"

        return (
            ok,
            message,
            {
                "observed_shape": list(array.shape),
                "observed_dtype": str(array.dtype),
                "observed_count": length,
            },
        )

    return _validator


def _synthetic_ecg_signal() -> tuple[Any, float]:
    import numpy as np

    sampling_rate = 1000.0
    time = np.linspace(0.0, 2.0, int(2.0 * sampling_rate), endpoint=False)
    signal = 0.02 * np.sin(2 * np.pi * 5 * time)
    for center in (0.3, 0.8, 1.3, 1.8):
        signal += np.exp(-((time - center) ** 2) / (2 * (0.01 ** 2)))
    return signal, sampling_rate


def _synthetic_ppg_signal() -> tuple[Any, float]:
    import numpy as np

    sampling_rate = 100.0
    time = np.linspace(0.0, 10.0, int(10.0 * sampling_rate), endpoint=False)
    signal = np.zeros_like(time)
    for center in np.arange(0.5, 10.0, 1.0):
        signal += np.exp(-((time - center) ** 2) / (2 * (0.03 ** 2)))
    return signal, sampling_rate


def _synthetic_emg_signal() -> tuple[Any, Any, float]:
    import numpy as np

    sampling_rate = 1000.0
    time = np.linspace(0.0, 2.0, int(2.0 * sampling_rate), endpoint=False)
    rest = 0.01 * np.sin(
        2 * np.pi * 10 * np.linspace(0.0, 0.4, int(0.4 * sampling_rate), endpoint=False)
    )
    signal = 0.01 * np.sin(2 * np.pi * 10 * time)
    signal[700:1100] += 0.5 * np.sin(np.linspace(0.0, np.pi, 400))
    signal[1300:1600] += 0.7 * np.sin(np.linspace(0.0, np.pi, 300))
    return signal, rest, sampling_rate


def _run_extract_patches_2d_probe(fn: Callable[..., Any]) -> dict[str, Any]:
    import numpy as np

    positive_case = _run_probe_case(
        "positive",
        fn,
        args=(np.arange(16).reshape(4, 4), (2, 2)),
        validator=_validate_patch_array,
    )
    negative_case = _run_probe_case(
        "negative",
        fn,
        args=(None, (2, 2)),
        expect_exception=True,
    )
    return _compile_probe_result(
        "sklearn.images.extract_patches_2d.basic",
        "extract_patches_2d",
        positive_case=positive_case,
        negative_case=negative_case,
    )


def _run_reconstruct_from_patches_2d_probe(fn: Callable[..., Any]) -> dict[str, Any]:
    import numpy as np

    positive_case = _run_probe_case(
        "positive",
        fn,
        args=(np.arange(16).reshape(4, 2, 2), (3, 3)),
        validator=_validate_image_shape((3, 3)),
    )
    negative_case = _run_probe_case(
        "negative",
        fn,
        args=(None, (3, 3)),
        expect_exception=True,
    )
    return _compile_probe_result(
        "sklearn.images.reconstruct_from_patches_2d.basic",
        "reconstruct_from_patches_2d",
        positive_case=positive_case,
        negative_case=negative_case,
    )


def _run_img_to_graph_probe(fn: Callable[..., Any]) -> dict[str, Any]:
    import numpy as np

    positive_case = _run_probe_case(
        "positive",
        fn,
        args=(np.arange(8).reshape(2, 2, 2),),
        validator=_validate_square_shape(8),
    )
    negative_case = _run_probe_case(
        "negative",
        fn,
        args=(None,),
        expect_exception=True,
    )
    return _compile_probe_result(
        "sklearn.images.img_to_graph.basic",
        "img_to_graph",
        positive_case=positive_case,
        negative_case=negative_case,
    )


def _run_grid_to_graph_probe(fn: Callable[..., Any]) -> dict[str, Any]:
    positive_case = _run_probe_case(
        "positive",
        fn,
        args=(2, 2),
        validator=_validate_square_shape(4),
    )
    negative_case = _run_probe_case(
        "negative",
        fn,
        args=(None, 2),
        expect_exception=True,
    )
    return _compile_probe_result(
        "sklearn.images.grid_to_graph.basic",
        "grid_to_graph",
        positive_case=positive_case,
        negative_case=negative_case,
    )


def _run_fft_probe(fn: Callable[..., Any]) -> dict[str, Any]:
    import numpy as np

    positive_case = _run_probe_case(
        "positive",
        fn,
        args=(np.array([0.0, 1.0, 0.0, 0.0]),),
        validator=_validate_fft_output,
    )
    negative_case = _run_probe_case(
        "negative",
        fn,
        args=(None,),
        expect_exception=True,
    )
    return _compile_probe_result(
        "numerical.fft.basic",
        "fft",
        positive_case=positive_case,
        negative_case=negative_case,
    )


def _run_utc_to_tai_leap_second_kernel_probe(fn: Callable[..., Any]) -> dict[str, Any]:
    positive_case = _run_probe_case(
        "positive",
        fn,
        args=(100.0,),
        kwargs={"leap_seconds": 37.0},
        validator=_validate_numeric_offset,
    )
    negative_case = _run_probe_case(
        "negative",
        fn,
        args=(None,),
        kwargs={"leap_seconds": 37.0},
        expect_exception=True,
    )
    return _compile_probe_result(
        "tempo_jl.offsets.utc_to_tai_leap_second_kernel.basic",
        "utc_to_tai_leap_second_kernel",
        positive_case=positive_case,
        negative_case=negative_case,
    )


def _run_hamilton_segmentation_probe(fn: Callable[..., Any]) -> dict[str, Any]:
    signal, sampling_rate = _synthetic_ecg_signal()

    positive_case = _run_probe_case(
        "positive",
        fn,
        args=(signal, sampling_rate),
        validator=_validate_monotonic_index_array(max_value=len(signal) - 1),
    )
    negative_case = _run_probe_case(
        "negative",
        fn,
        args=(None, sampling_rate),
        expect_exception=True,
    )
    return _compile_probe_result(
        "biosppy.ecg.hamilton_segmentation.basic",
        "hamilton_segmentation",
        positive_case=positive_case,
        negative_case=negative_case,
    )


def _run_hamilton_segmenter_probe(fn: Callable[..., Any]) -> dict[str, Any]:
    signal, sampling_rate = _synthetic_ecg_signal()

    positive_case = _run_probe_case(
        "positive",
        fn,
        args=(signal, sampling_rate),
        validator=_validate_monotonic_index_array(max_value=len(signal) - 1),
    )
    negative_case = _run_probe_case(
        "negative",
        fn,
        args=(signal, "bad"),
        expect_exception=True,
    )
    return _compile_probe_result(
        "biosppy.ecg.hamilton_segmenter.basic",
        "hamilton_segmenter",
        positive_case=positive_case,
        negative_case=negative_case,
    )


def _run_detect_signal_onsets_elgendi2013_probe(fn: Callable[..., Any]) -> dict[str, Any]:
    signal, sampling_rate = _synthetic_ppg_signal()

    positive_case = _run_probe_case(
        "positive",
        fn,
        args=(signal, sampling_rate, 0.111, 0.667, 0.02, 0.3),
        validator=_validate_monotonic_index_array(max_value=len(signal) - 1),
    )
    negative_case = _run_probe_case(
        "negative",
        fn,
        args=(signal, "bad", 0.111, 0.667, 0.02, 0.3),
        expect_exception=True,
    )
    return _compile_probe_result(
        "biosppy.ppg.detect_signal_onsets_elgendi2013.basic",
        "detect_signal_onsets_elgendi2013",
        positive_case=positive_case,
        negative_case=negative_case,
    )


def _run_detectonsetevents_probe(fn: Callable[..., Any]) -> dict[str, Any]:
    signal, sampling_rate = _synthetic_ppg_signal()

    positive_case = _run_probe_case(
        "positive",
        fn,
        args=(signal, sampling_rate, 0.2, 4, 60.0, 0.3, 180.0),
        validator=_validate_monotonic_index_array(max_value=len(signal) - 1),
    )
    negative_case = _run_probe_case(
        "negative",
        fn,
        args=(None, sampling_rate, 0.2, 4, 60.0, 0.3, 180.0),
        expect_exception=True,
    )
    return _compile_probe_result(
        "biosppy.ppg.detectonsetevents.basic",
        "detectonsetevents",
        positive_case=positive_case,
        negative_case=negative_case,
    )


def _run_detect_onsets_with_rest_aware_thresholds_probe(fn: Callable[..., Any]) -> dict[str, Any]:
    signal, rest, sampling_rate = _synthetic_emg_signal()

    positive_case = _run_probe_case(
        "positive",
        fn,
        args=(signal, rest, sampling_rate, 20, 10, 1.0, 0.5),
        validator=_validate_monotonic_index_array(
            allow_empty=True,
            max_value=len(signal) - 1,
        ),
    )
    negative_case = _run_probe_case(
        "negative",
        fn,
        args=(None, rest, sampling_rate, 20, 10, 1.0, 0.5),
        expect_exception=True,
    )
    return _compile_probe_result(
        "biosppy.emg.detect_onsets_with_rest_aware_thresholds.basic",
        "detect_onsets_with_rest_aware_thresholds",
        positive_case=positive_case,
        negative_case=negative_case,
    )


def _run_threshold_based_onset_detection_probe(fn: Callable[..., Any]) -> dict[str, Any]:
    signal, rest, sampling_rate = _synthetic_emg_signal()

    positive_case = _run_probe_case(
        "positive",
        fn,
        args=(signal, rest, sampling_rate, 1.0, 0.05),
        validator=_validate_monotonic_index_array(
            allow_empty=True,
            max_value=len(signal) - 1,
        ),
    )
    negative_case = _run_probe_case(
        "negative",
        fn,
        args=(None, rest, sampling_rate, 1.0, 0.05),
        expect_exception=True,
    )
    return _compile_probe_result(
        "biosppy.emg.threshold_based_onset_detection.basic",
        "threshold_based_onset_detection",
        positive_case=positive_case,
        negative_case=negative_case,
    )


def _run_linalg_solve_probe(fn: Callable[..., Any]) -> dict[str, Any]:
    import numpy as np

    a = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([9.0, 8.0])
    positive_case = _run_probe_case(
        "positive",
        fn,
        args=(a, b),
        validator=_validate_array_shape((2,)),
    )
    negative_case = _run_probe_case(
        "negative",
        fn,
        args=(None, b),
        expect_exception=True,
    )
    return _compile_probe_result(
        "scipy.linalg.solve.basic",
        "solve",
        positive_case=positive_case,
        negative_case=negative_case,
    )


def _run_linalg_inv_probe(fn: Callable[..., Any]) -> dict[str, Any]:
    import numpy as np

    a = np.array([[2.0, 0.0], [0.0, 4.0]])
    positive_case = _run_probe_case(
        "positive",
        fn,
        args=(a,),
        validator=_validate_array_shape((2, 2)),
    )
    negative_case = _run_probe_case(
        "negative",
        fn,
        args=(None,),
        expect_exception=True,
    )
    return _compile_probe_result(
        "scipy.linalg.inv.basic",
        "inv",
        positive_case=positive_case,
        negative_case=negative_case,
    )


def _run_linalg_det_probe(fn: Callable[..., Any]) -> dict[str, Any]:
    import numpy as np

    a = np.array([[2.0, 0.0], [0.0, 4.0]])
    positive_case = _run_probe_case(
        "positive",
        fn,
        args=(a,),
        validator=_validate_scalar_result,
    )
    negative_case = _run_probe_case(
        "negative",
        fn,
        args=(None,),
        expect_exception=True,
    )
    return _compile_probe_result(
        "scipy.linalg.det.basic",
        "det",
        positive_case=positive_case,
        negative_case=negative_case,
    )


def _run_linalg_lu_factor_probe(fn: Callable[..., Any]) -> dict[str, Any]:
    import numpy as np

    a = np.array([[3.0, 1.0], [1.0, 2.0]])
    positive_case = _run_probe_case(
        "positive",
        fn,
        args=(a,),
        validator=_validate_tuple_result(2),
    )
    negative_case = _run_probe_case(
        "negative",
        fn,
        args=(None,),
        expect_exception=True,
    )
    return _compile_probe_result(
        "scipy.linalg.lu_factor.basic",
        "lu_factor",
        positive_case=positive_case,
        negative_case=negative_case,
    )


def _run_linalg_lu_solve_probe(fn: Callable[..., Any]) -> dict[str, Any]:
    import numpy as np
    from scipy.linalg import lu_factor

    a = np.array([[3.0, 1.0], [1.0, 2.0]])
    lu_and_piv = lu_factor(a)
    b = np.array([9.0, 8.0])
    positive_case = _run_probe_case(
        "positive",
        fn,
        args=(lu_and_piv, b),
        validator=_validate_array_shape((2,)),
    )
    negative_case = _run_probe_case(
        "negative",
        fn,
        args=((lu_and_piv[0],), b),
        expect_exception=True,
    )
    return _compile_probe_result(
        "scipy.linalg.lu_solve.basic",
        "lu_solve",
        positive_case=positive_case,
        negative_case=negative_case,
    )


def _run_optimize_minimize_probe(fn: Callable[..., Any]) -> dict[str, Any]:
    import numpy as np

    def objective(x: np.ndarray) -> float:
        return float(np.sum((x - 1.0) ** 2))

    x0 = np.array([0.0, 0.0])
    positive_case = _run_probe_case(
        "positive",
        fn,
        args=(objective, x0),
        validator=_validate_optimize_vector(2),
    )
    negative_case = _run_probe_case(
        "negative",
        fn,
        args=(None, x0),
        expect_exception=True,
    )
    return _compile_probe_result(
        "scipy.optimize.minimize.basic",
        "minimize",
        positive_case=positive_case,
        negative_case=negative_case,
    )


def _run_optimize_root_probe(fn: Callable[..., Any]) -> dict[str, Any]:
    import numpy as np

    def objective(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] - 1.0, x[1] - 2.0])

    x0 = np.array([0.0, 0.0])
    positive_case = _run_probe_case(
        "positive",
        fn,
        args=(objective, x0),
        validator=_validate_optimize_vector(2),
    )
    negative_case = _run_probe_case(
        "negative",
        fn,
        args=(None, x0),
        expect_exception=True,
    )
    return _compile_probe_result(
        "scipy.optimize.root.basic",
        "root",
        positive_case=positive_case,
        negative_case=negative_case,
    )


def _run_optimize_linprog_probe(fn: Callable[..., Any]) -> dict[str, Any]:
    import numpy as np

    c = np.array([1.0, 1.0])
    A_ub = np.array([[-1.0, 0.0], [0.0, -1.0]])
    b_ub = np.array([0.0, 0.0])
    positive_case = _run_probe_case(
        "positive",
        fn,
        args=(c,),
        kwargs={"A_ub": A_ub, "b_ub": b_ub},
        validator=_validate_optimize_vector(2),
    )
    negative_case = _run_probe_case(
        "negative",
        fn,
        args=(None,),
        kwargs={"A_ub": A_ub, "b_ub": b_ub},
        expect_exception=True,
    )
    return _compile_probe_result(
        "scipy.optimize.linprog.basic",
        "linprog",
        positive_case=positive_case,
        negative_case=negative_case,
    )


def _run_optimize_curve_fit_probe(fn: Callable[..., Any]) -> dict[str, Any]:
    import numpy as np

    def model(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * x + b

    xdata = np.array([0.0, 1.0, 2.0, 3.0])
    ydata = np.array([1.0, 3.0, 5.0, 7.0])
    positive_case = _run_probe_case(
        "positive",
        fn,
        args=(model, xdata, ydata),
        validator=_validate_curve_fit_result(2),
    )
    negative_case = _run_probe_case(
        "negative",
        fn,
        args=(model, None, ydata),
        expect_exception=True,
    )
    return _compile_probe_result(
        "scipy.optimize.curve_fit.basic",
        "curve_fit",
        positive_case=positive_case,
        negative_case=negative_case,
    )


def _run_optimize_shgo_probe(fn: Callable[..., Any]) -> dict[str, Any]:
    def objective(x: Any) -> float:
        return float((x[0] - 1.0) ** 2)

    bounds = [(-2.0, 2.0)]
    positive_case = _run_probe_case(
        "positive",
        fn,
        args=(objective, bounds),
        validator=_validate_optimize_vector(1),
    )
    negative_case = _run_probe_case(
        "negative",
        fn,
        args=(None, bounds),
        expect_exception=True,
    )
    return _compile_probe_result(
        "scipy.optimize.shgo.basic",
        "shgo",
        positive_case=positive_case,
        negative_case=negative_case,
    )


def _run_optimize_differential_evolution_probe(fn: Callable[..., Any]) -> dict[str, Any]:
    def objective(x: Any) -> float:
        return float((x[0] - 1.0) ** 2)

    bounds = [(-2.0, 2.0)]
    positive_case = _run_probe_case(
        "positive",
        fn,
        args=(objective, bounds),
        validator=_validate_optimize_vector(1),
    )
    negative_case = _run_probe_case(
        "negative",
        fn,
        args=(None, bounds),
        expect_exception=True,
    )
    return _compile_probe_result(
        "scipy.optimize.differential_evolution.basic",
        "differential_evolution",
        positive_case=positive_case,
        negative_case=negative_case,
    )


ALLOWLISTED_SMOKE_PROBES: tuple[SmokeProbe, ...] = (
    SmokeProbe(
        probe_id="sklearn.images.extract_patches_2d.basic",
        target_symbol="extract_patches_2d",
        package_basenames=("images",),
        runner=_run_extract_patches_2d_probe,
    ),
    SmokeProbe(
        probe_id="sklearn.images.reconstruct_from_patches_2d.basic",
        target_symbol="reconstruct_from_patches_2d",
        package_basenames=("images",),
        runner=_run_reconstruct_from_patches_2d_probe,
    ),
    SmokeProbe(
        probe_id="sklearn.images.img_to_graph.basic",
        target_symbol="img_to_graph",
        package_basenames=("images",),
        runner=_run_img_to_graph_probe,
    ),
    SmokeProbe(
        probe_id="sklearn.images.grid_to_graph.basic",
        target_symbol="grid_to_graph",
        package_basenames=("images",),
        runner=_run_grid_to_graph_probe,
    ),
    SmokeProbe(
        probe_id="numerical.fft.basic",
        target_symbol="fft",
        runner=_run_fft_probe,
    ),
    SmokeProbe(
        probe_id="tempo_jl.offsets.utc_to_tai_leap_second_kernel.basic",
        target_symbol="utc_to_tai_leap_second_kernel",
        package_basenames=("offsets",),
        runner=_run_utc_to_tai_leap_second_kernel_probe,
    ),
    SmokeProbe(
        probe_id="biosppy.ecg.hamilton_segmentation.basic",
        target_symbol="hamilton_segmentation",
        package_basenames=("ecg_detectors",),
        runner=_run_hamilton_segmentation_probe,
    ),
    SmokeProbe(
        probe_id="biosppy.ecg.hamilton_segmenter.basic",
        target_symbol="hamilton_segmenter",
        package_basenames=("ecg_detectors",),
        runner=_run_hamilton_segmenter_probe,
    ),
    SmokeProbe(
        probe_id="biosppy.ppg.detect_signal_onsets_elgendi2013.basic",
        target_symbol="detect_signal_onsets_elgendi2013",
        package_basenames=("ppg_detectors",),
        runner=_run_detect_signal_onsets_elgendi2013_probe,
    ),
    SmokeProbe(
        probe_id="biosppy.ppg.detectonsetevents.basic",
        target_symbol="detectonsetevents",
        package_basenames=("ppg_detectors",),
        runner=_run_detectonsetevents_probe,
    ),
    SmokeProbe(
        probe_id="biosppy.emg.detect_onsets_with_rest_aware_thresholds.basic",
        target_symbol="detect_onsets_with_rest_aware_thresholds",
        package_basenames=("emg_detectors",),
        runner=_run_detect_onsets_with_rest_aware_thresholds_probe,
    ),
    SmokeProbe(
        probe_id="biosppy.emg.threshold_based_onset_detection.basic",
        target_symbol="threshold_based_onset_detection",
        package_basenames=("emg_detectors",),
        runner=_run_threshold_based_onset_detection_probe,
    ),
    SmokeProbe(
        probe_id="scipy.linalg.solve.basic",
        target_symbol="solve",
        package_basenames=("linalg",),
        runner=_run_linalg_solve_probe,
    ),
    SmokeProbe(
        probe_id="scipy.linalg.inv.basic",
        target_symbol="inv",
        package_basenames=("linalg",),
        runner=_run_linalg_inv_probe,
    ),
    SmokeProbe(
        probe_id="scipy.linalg.det.basic",
        target_symbol="det",
        package_basenames=("linalg",),
        runner=_run_linalg_det_probe,
    ),
    SmokeProbe(
        probe_id="scipy.linalg.lu_factor.basic",
        target_symbol="lu_factor",
        package_basenames=("linalg",),
        runner=_run_linalg_lu_factor_probe,
    ),
    SmokeProbe(
        probe_id="scipy.linalg.lu_solve.basic",
        target_symbol="lu_solve",
        package_basenames=("linalg",),
        runner=_run_linalg_lu_solve_probe,
    ),
    SmokeProbe(
        probe_id="scipy.optimize.minimize.basic",
        target_symbol="minimize",
        package_basenames=("optimize",),
        runner=_run_optimize_minimize_probe,
    ),
    SmokeProbe(
        probe_id="scipy.optimize.root.basic",
        target_symbol="root",
        package_basenames=("optimize",),
        runner=_run_optimize_root_probe,
    ),
    SmokeProbe(
        probe_id="scipy.optimize.linprog.basic",
        target_symbol="linprog",
        package_basenames=("optimize",),
        runner=_run_optimize_linprog_probe,
    ),
    SmokeProbe(
        probe_id="scipy.optimize.curve_fit.basic",
        target_symbol="curve_fit",
        package_basenames=("optimize",),
        runner=_run_optimize_curve_fit_probe,
    ),
    SmokeProbe(
        probe_id="scipy.optimize.shgo.basic",
        target_symbol="shgo",
        package_basenames=("optimize",),
        runner=_run_optimize_shgo_probe,
    ),
    SmokeProbe(
        probe_id="scipy.optimize.differential_evolution.basic",
        target_symbol="differential_evolution",
        package_basenames=("optimize",),
        runner=_run_optimize_differential_evolution_probe,
    ),
)


@contextmanager
def _module_import_path(path: Path):
    path_str = str(path)
    original = list(sys.path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    try:
        yield
    finally:
        sys.path[:] = original


def _clear_module(module_name: str) -> None:
    doomed = [
        name
        for name in sys.modules
        if name == module_name or name.startswith(module_name + ".")
    ]
    for name in doomed:
        sys.modules.pop(name, None)


def _import_atoms_module(output_dir: Path):
    package_name = output_dir.name
    module_name = f"{package_name}.atoms"
    _clear_module(package_name)
    with _module_import_path(output_dir.parent):
        return importlib.import_module(module_name)


def _select_probe(
    *,
    package_basename: str,
    target_symbol: str,
) -> SmokeProbe | None:
    for probe in ALLOWLISTED_SMOKE_PROBES:
        if probe.matches(
            package_basename=package_basename,
            target_symbol=target_symbol,
        ):
            return probe
    return None


def run_smoke_validation(
    staged_dir: str | Path,
    *,
    package_basename: str,
    target_symbol: str,
) -> dict[str, Any]:
    staged_path = Path(staged_dir)
    probe = _select_probe(
        package_basename=package_basename,
        target_symbol=target_symbol,
    )
    if probe is None:
        return SmokeResult(
            status=SMOKE_STATUS_NOT_APPLICABLE,
            target_symbol=target_symbol,
            probe_id="",
            message="no allowlisted smoke probe for target",
            details={"package_basename": package_basename},
        ).to_dict()

    try:
        with tempfile.TemporaryDirectory(prefix="sciona_ingest_smoke_") as tmp_root:
            package_dir = Path(tmp_root) / package_basename
            package_dir.mkdir(parents=True, exist_ok=True)
            for path in sorted(staged_path.iterdir()):
                if path.is_file():
                    shutil.copy2(path, package_dir / path.name)
            module = _import_atoms_module(package_dir)
    except Exception as exc:
        return SmokeResult(
            status=SMOKE_STATUS_FAIL,
            target_symbol=target_symbol,
            probe_id=probe.probe_id,
            message="failed to import generated atoms module",
            details={"exception": repr(exc)},
        ).to_dict()

    fn = getattr(module, probe.target_symbol, None)
    if not callable(fn):
        return SmokeResult(
            status=SMOKE_STATUS_FAIL,
            target_symbol=target_symbol,
            probe_id=probe.probe_id,
            message="allowlisted smoke target is missing or not callable",
            details={"callable_name": probe.target_symbol},
        ).to_dict()

    try:
        return probe.runner(fn)
    except Exception as exc:
        return SmokeResult(
            status=SMOKE_STATUS_FAIL,
            target_symbol=target_symbol,
            probe_id=probe.probe_id,
            message="allowlisted smoke probe failed",
            details={"exception": repr(exc)},
        ).to_dict()
