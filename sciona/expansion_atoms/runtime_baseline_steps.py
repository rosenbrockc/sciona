"""Standalone processing atoms for baseline analysis skeleton phases.

Each atom is a pure function operating on numpy arrays. No external baseline
package dependency.
Returns ``tuple[np.ndarray, bool]`` where the bool indicates successful
operation / non-degenerate output.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import gaussian as gaussian_window

try:
    from scipy.signal import cwt, ricker
except ImportError:
    def ricker(points: int, a: float) -> np.ndarray:
        """Compatibility fallback for SciPy builds without top-level ricker."""
        points = max(1, int(points))
        a = max(float(a), 1e-12)
        x = np.arange(points, dtype=np.float64) - (points - 1.0) / 2.0
        x = x / a
        norm = 2.0 / (np.sqrt(3.0 * a) * np.pi ** 0.25)
        return norm * (1.0 - x**2) * np.exp(-(x**2) / 2.0)

    def cwt(data: np.ndarray, wavelet, widths: np.ndarray) -> np.ndarray:
        """Compatibility fallback for SciPy builds without top-level cwt."""
        data_arr = np.asarray(data, dtype=np.float64).reshape(-1)
        widths_arr = np.asarray(widths, dtype=np.float64).reshape(-1)
        output = np.zeros((widths_arr.size, data_arr.size), dtype=np.float64)
        for i, width in enumerate(widths_arr):
            points = max(1, int(round(10.0 * float(width))))
            kernel = np.asarray(wavelet(points, float(width)), dtype=np.float64).reshape(-1)
            if kernel.size == 0:
                continue
            output[i] = np.convolve(data_arr, kernel[::-1], mode="same")
        return output


def _as_float_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def baseline_mask(
    signal: np.ndarray,
    t: np.ndarray,
    mask: np.ndarray,
    *,
    zero: bool = True,
) -> tuple[np.ndarray, bool]:
    """Apply a boolean mask to signal data."""
    _ = t
    signal_arr = _as_float_array(signal)
    mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
    if signal_arr.size == 0:
        return signal_arr, False
    if zero:
        result = signal_arr.copy()
        result[~mask_arr] = 0.0
    else:
        result = signal_arr[mask_arr]
    return result, bool(np.any(result != 0.0))


def baseline_resample(
    signal: np.ndarray,
    t: np.ndarray,
    anchor: np.ndarray,
    *,
    method: str = "linear",
) -> tuple[np.ndarray, bool]:
    """Resample signal to an anchor grid."""
    signal_arr = _as_float_array(signal)
    t_arr = _as_float_array(t)
    anchor_arr = _as_float_array(anchor)
    if signal_arr.size == 0 or t_arr.size == 0 or anchor_arr.size == 0:
        return np.array([], dtype=np.float64), False
    if signal_arr.size != t_arr.size:
        raise ValueError("signal and t must have the same length")

    if method == "nearest":
        indices = np.searchsorted(t_arr, anchor_arr, side="left")
        indices = np.clip(indices, 0, signal_arr.size - 1)
        left = np.clip(indices - 1, 0, signal_arr.size - 1)
        right = indices
        choose_right = np.abs(anchor_arr - t_arr[right]) <= np.abs(anchor_arr - t_arr[left])
        nearest = np.where(choose_right, right, left)
        result = signal_arr[nearest]
    else:
        if signal_arr.size == 1:
            result = np.full(anchor_arr.shape, signal_arr[0], dtype=np.float64)
        else:
            result = np.interp(anchor_arr, t_arr, signal_arr)
    return np.asarray(result, dtype=np.float64), result.size > 0


def baseline_scale_constant(
    signal: np.ndarray,
    *,
    floor: float = 0.0,
    ceil: float = 1.0,
) -> tuple[np.ndarray, bool]:
    """Normalize signal magnitude by constant ratio."""
    signal_arr = _as_float_array(signal)
    if signal_arr.size == 0:
        return signal_arr, False

    wrng = min(float(np.nanmax(signal_arr)), ceil)
    denom = ceil - floor
    if denom <= 0 or wrng < floor:
        return signal_arr.copy(), True
    ratio = wrng / denom
    if ratio <= 0:
        return signal_arr.copy(), False
    return signal_arr / ratio, True


def baseline_output_nonzero(
    values: np.ndarray,
    t: np.ndarray,
    *,
    maxscale: float | None = None,
    discretize: bool = False,
) -> tuple[np.ndarray, bool]:
    """Extract non-zero onset values from a signal."""
    _ = t
    values_arr = _as_float_array(values)
    if values_arr.size == 0:
        return np.array([], dtype=np.float64), False
    if maxscale is not None and maxscale != 0:
        values_arr = values_arr / float(maxscale)
    nz = np.nonzero(values_arr)[0]
    if nz.size == 0:
        return np.array([], dtype=np.float64), False
    if discretize:
        return np.ones(nz.size, dtype=np.float64), True
    return values_arr[nz], True


def baseline_output_clipshift(
    values: np.ndarray,
    t: np.ndarray,
    *,
    threshold: float = 0.0,
    ceil: float = 1.0,
    qscale: float | None = None,
) -> tuple[np.ndarray, bool]:
    """Threshold-shift and clip onset values."""
    _ = t
    values_arr = _as_float_array(values)
    if values_arr.size == 0:
        return np.array([], dtype=np.float64), False
    if qscale is not None:
        positive = values_arr[values_arr > 0.0]
        qval = float(np.nanquantile(positive, qscale)) if positive.size > 0 else 1.0
        if qval > 0.0:
            values_arr = np.clip(values_arr / qval, 0.0, 1.0)
    shifted = np.clip(values_arr - threshold, 0.0, ceil)
    nz = np.nonzero(shifted)[0]
    if nz.size == 0:
        return np.array([], dtype=np.float64), False
    return shifted[nz], True


def baseline_output_copy(
    values: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Pass through onset values unchanged."""
    _ = t
    values_arr = _as_float_array(values)
    return values_arr.copy(), values_arr.size > 0


def _pad_core(
    onsets: np.ndarray,
    t: np.ndarray,
    anchor: np.ndarray,
    width: float,
    decay_fn,
) -> tuple[np.ndarray, bool]:
    """Shared padding loop: accumulate onset values over the anchor grid."""
    onsets_arr = _as_float_array(onsets)
    t_arr = _as_float_array(t)
    anchor_arr = _as_float_array(anchor)
    result = np.zeros(anchor_arr.size, dtype=np.float64)
    if onsets_arr.size == 0 or t_arr.size == 0 or anchor_arr.size == 0 or width == 0:
        return result, False

    limit = min(onsets_arr.size, t_arr.size)
    for i in range(limit):
        onset_t = t_arr[i]
        if width < 0:
            start_t, end_t = onset_t + width, onset_t
        else:
            start_t, end_t = onset_t, onset_t + width
        start = int(np.searchsorted(anchor_arr, start_t, side="left"))
        stop = int(np.searchsorted(anchor_arr, end_t, side="right"))
        n = stop - start
        if n <= 0:
            continue
        curve = np.asarray(decay_fn(n), dtype=np.float64).reshape(-1)
        if curve.size != n:
            curve = np.resize(curve, n)
        if width < 0:
            curve = curve[::-1]
        result[start:stop] += onsets_arr[i] * curve

    return result, bool(np.any(result > 0.0))


def baseline_pad_constant(
    onsets: np.ndarray,
    t: np.ndarray,
    anchor: np.ndarray,
    *,
    width: float = 1.0,
) -> tuple[np.ndarray, bool]:
    """Rectangular padding around onsets."""
    return _pad_core(onsets, t, anchor, width, lambda n: np.ones(n, dtype=np.float64))


def baseline_pad_linear(
    onsets: np.ndarray,
    t: np.ndarray,
    anchor: np.ndarray,
    *,
    width: float = 1.0,
) -> tuple[np.ndarray, bool]:
    """Linearly decaying padding around onsets."""
    return _pad_core(
        onsets,
        t,
        anchor,
        width,
        lambda n: np.linspace(1.0, 0.0, n) if n > 1 else np.ones(n, dtype=np.float64),
    )


def baseline_normalize_max(
    signal: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Normalize signal by its maximum value."""
    signal_arr = _as_float_array(signal)
    if signal_arr.size == 0:
        return signal_arr, False
    mx = float(np.nanmax(signal_arr))
    if mx <= 0.0:
        return np.zeros_like(signal_arr), False
    return signal_arr / mx, True


def baseline_normalize_constant(
    signal: np.ndarray,
    *,
    value: float = 1.0,
) -> tuple[np.ndarray, bool]:
    """Normalize signal by a fixed constant."""
    signal_arr = _as_float_array(signal)
    if value == 0:
        return np.zeros_like(signal_arr), False
    return np.clip(signal_arr / float(value), 0.0, 1.0), True


def baseline_normalize_quantile(
    signal: np.ndarray,
    *,
    q: float = 0.95,
) -> tuple[np.ndarray, bool]:
    """Normalize signal by a quantile value."""
    signal_arr = _as_float_array(signal)
    if signal_arr.size == 0:
        return signal_arr, False
    qval = float(np.nanquantile(signal_arr, q))
    if qval <= 0.0:
        return np.zeros_like(signal_arr), False
    return np.clip(signal_arr / qval, 0.0, 1.0), True


def baseline_regionize(
    signal: np.ndarray,
    *,
    threshold: float = 0.5,
    min_length: int = 1,
) -> tuple[np.ndarray, bool]:
    """Threshold signal into discrete event regions."""
    signal_arr = _as_float_array(signal)
    if signal_arr.size == 0:
        return np.array([], dtype=np.int64), False

    binary = signal_arr >= threshold
    labels = np.zeros(signal_arr.size, dtype=np.int64)
    padded = np.concatenate(([False], binary, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    stops = np.where(diffs == -1)[0]

    region_id = 0
    for start, stop in zip(starts, stops):
        if stop - start < min_length:
            continue
        region_id += 1
        labels[start:stop] = region_id

    return labels, region_id > 0


def baseline_scale_wavelet(
    signal: np.ndarray,
    *,
    widths: np.ndarray | None = None,
    floor: float = 0.0,
    ceil: float = 1.0,
) -> tuple[np.ndarray, bool]:
    """Normalize signal magnitude via wavelet prominence."""
    signal_arr = _as_float_array(signal)
    if signal_arr.size < 4:
        return signal_arr.copy(), False

    if widths is None:
        widths_arr = np.array([1, 2, 4, 8, 16], dtype=np.float64)
    else:
        widths_arr = np.asarray(widths, dtype=np.float64).reshape(-1)
        widths_arr = widths_arr[widths_arr > 0]
        if widths_arr.size == 0:
            widths_arr = np.array([1, 2, 4, 8, 16], dtype=np.float64)

    try:
        from scipy.ndimage import median_filter

        smooth_size = max(3, signal_arr.size // 50 or 1)
        if smooth_size % 2 == 0:
            smooth_size += 1
        smoothed = median_filter(signal_arr, size=smooth_size, mode="nearest")
    except Exception:
        smoothed = signal_arr.copy()

    coeffs = cwt(smoothed, ricker, widths_arr)
    prominence = np.max(np.abs(coeffs), axis=0)
    mx = float(np.nanmax(prominence)) if prominence.size > 0 else 0.0
    if not np.isfinite(mx) or mx <= floor:
        return np.zeros_like(signal_arr), False

    normed = np.clip(prominence / mx, 0.0, ceil)
    result = np.log1p(normed)
    return result, bool(np.any(result > 0.0))


def _segment_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    stops = np.where(diffs == -1)[0]
    return list(zip(starts, stops))


def _exp_model(x: np.ndarray, A: float, a: float, x0: float, y0: float) -> np.ndarray:
    return A * np.exp(-a * (x - x0)) + y0


def _sinh_model(x: np.ndarray, A: float, a: float, x0: float, y0: float) -> np.ndarray:
    arg = np.clip(a * (x - x0), -50.0, 50.0)
    return A * np.sinh(arg) + y0


def _fit_segments(
    signal: np.ndarray,
    t: np.ndarray,
    model_fn,
    rising: bool,
    *,
    min_width: int = 3,
    floor: float = 0.0,
) -> tuple[np.ndarray, bool]:
    """Detect contiguous derivative runs and fit a simple parametric model."""
    signal_arr = _as_float_array(signal)
    t_arr = _as_float_array(t)
    result = np.zeros_like(signal_arr, dtype=np.float64)
    if signal_arr.size < max(2, min_width + 1) or t_arr.size != signal_arr.size:
        return result, False

    deriv = np.diff(signal_arr)
    candidates = deriv > floor if rising else deriv < -floor
    found = False

    for start, stop in _segment_runs(candidates):
        seg_stop = stop + 1
        if seg_stop - start < min_width:
            continue
        seg_s = signal_arr[start:seg_stop]
        seg_t = t_arr[start:seg_stop]
        if seg_s.size < 2 or seg_t.size < 2:
            continue

        span = max(float(seg_t[-1] - seg_t[0]), 1e-12)
        slope = float(seg_s[-1] - seg_s[0])
        A0 = slope if abs(slope) > 1e-6 else (1.0 if rising else -1.0)
        a0 = 1.0 / span
        x0_0 = float(seg_t[0])
        y0_0 = float(seg_s[-1])

        try:
            fit = least_squares(
                lambda p: model_fn(seg_t, *p) - seg_s,
                x0=np.array([A0, a0, x0_0, y0_0], dtype=np.float64),
                bounds=(
                    np.array([-np.inf, 1e-8, -np.inf, -np.inf], dtype=np.float64),
                    np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float64),
                ),
                max_nfev=200,
            )
        except (ValueError, RuntimeError):
            continue

        if not fit.success:
            continue
        fit_slope = abs(float(fit.x[0]) * float(fit.x[1]))
        result[start:seg_stop] = np.log1p(fit_slope)
        found = True

    return result, found


def baseline_fit_exp_rise(
    signal: np.ndarray,
    t: np.ndarray,
    *,
    min_width: int = 3,
    floor: float = 0.0,
) -> tuple[np.ndarray, bool]:
    return _fit_segments(
        signal,
        t,
        _exp_model,
        True,
        min_width=min_width,
        floor=floor,
    )


def baseline_fit_exp_fall(
    signal: np.ndarray,
    t: np.ndarray,
    *,
    min_width: int = 3,
    floor: float = 0.0,
) -> tuple[np.ndarray, bool]:
    return _fit_segments(
        signal,
        t,
        _exp_model,
        False,
        min_width=min_width,
        floor=floor,
    )


def baseline_fit_sinh_rise(
    signal: np.ndarray,
    t: np.ndarray,
    *,
    min_width: int = 3,
    floor: float = 0.0,
) -> tuple[np.ndarray, bool]:
    return _fit_segments(
        signal,
        t,
        _sinh_model,
        True,
        min_width=min_width,
        floor=floor,
    )


def baseline_fit_sinh_fall(
    signal: np.ndarray,
    t: np.ndarray,
    *,
    min_width: int = 3,
    floor: float = 0.0,
) -> tuple[np.ndarray, bool]:
    return _fit_segments(
        signal,
        t,
        _sinh_model,
        False,
        min_width=min_width,
        floor=floor,
    )


def baseline_pad_exponential(
    onsets: np.ndarray,
    t: np.ndarray,
    anchor: np.ndarray,
    *,
    width: float = 1.0,
) -> tuple[np.ndarray, bool]:
    """Exponentially decaying padding around onsets."""
    alpha = -np.log(0.01) / abs(width) if width != 0 else 1.0

    def decay(n: int) -> np.ndarray:
        if n <= 0:
            return np.array([], dtype=np.float64)
        return np.exp(-alpha * np.linspace(0.0, abs(width), n))

    return _pad_core(onsets, t, anchor, width, decay)


def baseline_pad_gaussian(
    onsets: np.ndarray,
    t: np.ndarray,
    anchor: np.ndarray,
    *,
    width: float = 1.0,
) -> tuple[np.ndarray, bool]:
    """Gaussian-shaped padding around onsets."""

    def decay(n: int) -> np.ndarray:
        if n <= 0:
            return np.array([], dtype=np.float64)
        if n == 1:
            return np.ones(1, dtype=np.float64)
        std = 2.0 * np.sqrt(abs(width)) * n / 60.0
        std = max(float(std), 1.0)
        full = gaussian_window(max(2, n * 2), std)
        return np.asarray(full[n:], dtype=np.float64)

    return _pad_core(onsets, t, anchor, width, decay)


def baseline_combine_product(
    components: list[np.ndarray],
    *,
    plus_one: bool = False,
) -> tuple[np.ndarray, bool]:
    """Element-wise product of component vectors."""
    if not components:
        return np.array([], dtype=np.float64), False
    arrays = [np.asarray(component, dtype=np.float64).reshape(-1) for component in components]
    result = np.stack(arrays)
    if plus_one:
        return np.prod(1.0 + result, axis=0) - 1.0, True
    return np.prod(result, axis=0), True


def baseline_combine_convolve(
    components: list[np.ndarray],
) -> tuple[np.ndarray, bool]:
    """Sequential convolution of component vectors."""
    if not components:
        return np.array([], dtype=np.float64), False
    arrays = [np.asarray(component, dtype=np.float64).reshape(-1) for component in components]
    result = arrays[0].copy()
    for component in arrays[1:]:
        result = np.convolve(result, component, mode="same")
        total = float(np.sum(result))
        if total > 0.0:
            result = result / total
    return result, True


def baseline_combine_weighted(
    components: list[np.ndarray],
    *,
    weights: np.ndarray | None = None,
    plus_one: bool = False,
) -> tuple[np.ndarray, bool]:
    """Weighted element-wise product of component vectors."""
    if not components:
        return np.array([], dtype=np.float64), False
    arrays = [np.asarray(component, dtype=np.float64).reshape(-1) for component in components]
    if weights is None:
        weight_arr = np.ones(len(arrays), dtype=np.float64)
    else:
        weight_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weight_arr.size != len(arrays):
            raise ValueError("weights must match the number of components")
    stacked = np.stack(arrays)
    weighted = stacked * weight_arr[:, np.newaxis]
    if plus_one:
        return np.prod(1.0 + weighted, axis=0) - 1.0, True
    return np.prod(weighted, axis=0), True


def baseline_combine_coherence(
    components: list[np.ndarray],
    *,
    dt: float = 1.0,
    cutoff: float = 0.05,
    quantile: float = 0.975,
) -> tuple[np.ndarray, bool]:
    """Morphological-coherence weighted combination of component vectors."""
    if not components:
        return np.array([], dtype=np.float64), False

    def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if a.size == b.size:
            return a, b
        target_len = max(a.size, b.size)
        x_target = np.linspace(0.0, 1.0, target_len)
        if a.size == 1:
            a_aligned = np.full(target_len, a[0], dtype=np.float64)
        else:
            a_interp = interp1d(
                np.linspace(0.0, 1.0, a.size),
                a,
                kind="linear",
                fill_value="extrapolate",
                assume_sorted=True,
            )
            a_aligned = np.asarray(a_interp(x_target), dtype=np.float64)
        if b.size == 1:
            b_aligned = np.full(target_len, b[0], dtype=np.float64)
        else:
            b_interp = interp1d(
                np.linspace(0.0, 1.0, b.size),
                b,
                kind="linear",
                fill_value="extrapolate",
                assume_sorted=True,
            )
            b_aligned = np.asarray(b_interp(x_target), dtype=np.float64)
        return a_aligned, b_aligned

    def _pairwise_coherence(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_arr = np.asarray(a, dtype=np.float64).reshape(-1)
        b_arr = np.asarray(b, dtype=np.float64).reshape(-1)
        a_arr, b_arr = _align(a_arr, b_arr)
        diff_signal = a_arr - b_arr
        vs = np.abs(np.diff(diff_signal))
        if vs.size == 0:
            vs = np.zeros(1, dtype=np.float64)
        elif vs.size == 1:
            vs = np.pad(vs, (0, 1), mode="edge")
        else:
            vs = np.pad(vs, (0, 1), mode="reflect")

        nyq = 0.5 / max(dt, 1e-12)
        norm_cutoff = min(max(cutoff / nyq, 0.0), 0.99)
        filtered = vs.copy()
        if norm_cutoff > 0.0 and vs.size > 12:
            try:
                sos = butter(2, norm_cutoff, btype="low", output="sos")
                filtered = sosfiltfilt(sos, vs)
            except (ValueError, np.linalg.LinAlgError):
                filtered = vs.copy()

        scale = float(np.nanquantile(filtered, quantile)) if filtered.size > 0 else 0.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        norm_filt = np.clip(filtered, 0.0, scale) / scale
        norm_raw = np.clip(vs, 0.0, scale) / scale
        closeness = norm_filt * norm_raw
        return (1.0 - closeness) * (a_arr + b_arr) / 2.0

    result = np.asarray(components[0], dtype=np.float64).reshape(-1)
    if result.size == 0:
        return result, False
    if len(components) == 1:
        return result.copy(), True

    for component in components[1:]:
        result = _pairwise_coherence(result, component)
    return result, True
