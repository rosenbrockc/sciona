from __future__ import annotations
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal

from typing import Tuple

from sciona.ghost.abstract import (
    AbstractSignal,
    AbstractArray,
    AbstractScalar,
    AbstractBeatPool,
    AbstractDistribution,
    AbstractMCMCTrace,
    AbstractRNGState,
    AbstractMatrix,
    CONJUGATE_PAIRS,
)
from sciona.ghost.registry import register_atom


# ---------------------------------------------------------------------------
# FFT family
# ---------------------------------------------------------------------------

def witness_fft(sig: AbstractSignal) -> AbstractSignal:
    """Ghost witness for numpy.fft.fft.

    Preconditions:
        - Input must be in time domain.
    Postconditions:
        - Output shape matches input shape.
        - Output dtype is complex128.
        - Output domain is 'freq'.
    """
    sig.assert_domain("time")
    return AbstractSignal(
        shape=sig.shape,
        dtype="complex128",
        sampling_rate=sig.sampling_rate,
        domain="freq",
        units=sig.units,
    )


def witness_ifft(sig: AbstractSignal) -> AbstractSignal:
    """Ghost witness for numpy.fft.ifft.

    Preconditions:
        - Input must be in frequency domain.
    Postconditions:
        - Output shape matches input shape.
        - Output dtype is complex128.
        - Output domain is 'time'.
    """
    sig.assert_domain("freq")
    return AbstractSignal(
        shape=sig.shape,
        dtype="complex128",
        sampling_rate=sig.sampling_rate,
        domain="time",
        units=sig.units,
    )


def witness_rfft(sig: AbstractSignal) -> AbstractSignal:
    """Ghost witness for numpy.fft.rfft.

    Preconditions:
        - Input must be in time domain.
        - Input dtype must be real.
    Postconditions:
        - Output length is n//2 + 1.
        - Output dtype is complex128.
        - Output domain is 'freq'.
    """
    sig.assert_domain("time")
    if "complex" in sig.dtype:
        raise ValueError("rfft requires real-valued input, got dtype={sig.dtype}")
    n = sig.shape[0] if len(sig.shape) > 0 else 0
    out_len = n // 2 + 1
    return AbstractSignal(
        shape=(out_len,) + sig.shape[1:],
        dtype="complex128",
        sampling_rate=sig.sampling_rate,
        domain="freq",
        units=sig.units,
    )


def witness_irfft(sig: AbstractSignal) -> AbstractSignal:
    """Ghost witness for numpy.fft.irfft.

    Preconditions:
        - Input must be in frequency domain.
    Postconditions:
        - Output length is 2 * (input_length - 1).
        - Output dtype is float64.
        - Output domain is 'time'.
    """
    sig.assert_domain("freq")
    n = sig.shape[0] if len(sig.shape) > 0 else 0
    out_len = 2 * (n - 1)
    return AbstractSignal(
        shape=(out_len,) + sig.shape[1:],
        dtype="float64",
        sampling_rate=sig.sampling_rate,
        domain="time",
        units=sig.units,
    )


# ---------------------------------------------------------------------------
# DCT family
# ---------------------------------------------------------------------------

def witness_dct(sig: AbstractSignal) -> AbstractSignal:
    """Ghost witness for scipy.fft.dct.

    Preconditions:
        - Input must be in time domain.
        - Input dtype must be real.
    Postconditions:
        - Output shape matches input shape.
        - Output dtype is float64 (DCT is real-to-real).
        - Output domain is 'freq'.
    """
    sig.assert_domain("time")
    if "complex" in sig.dtype:
        raise ValueError(f"DCT requires real-valued input, got dtype={sig.dtype}")
    return AbstractSignal(
        shape=sig.shape,
        dtype="float64",
        sampling_rate=sig.sampling_rate,
        domain="freq",
        units=sig.units,
    )


def witness_idct(sig: AbstractSignal) -> AbstractSignal:
    """Ghost witness for scipy.fft.idct.

    Preconditions:
        - Input must be in frequency domain.
    Postconditions:
        - Output shape matches input shape.
        - Output dtype is float64.
        - Output domain is 'time'.
    """
    sig.assert_domain("freq")
    return AbstractSignal(
        shape=sig.shape,
        dtype="float64",
        sampling_rate=sig.sampling_rate,
        domain="time",
        units=sig.units,
    )


# ---------------------------------------------------------------------------
# Filter design witnesses
# ---------------------------------------------------------------------------

class AbstractFilterCoefficients:
    """Lightweight metadata for filter design output (b, a) or (sos)."""

    def __init__(
        self,
        order: int,
        btype: str,
        format: str = "ba",
        is_stable: bool = True,
    ) -> None:
        self.order = order
        self.btype = btype
        self.format = format
        self.is_stable = is_stable

    def assert_stable(self) -> None:
        if not self.is_stable:
            raise ValueError("Filter is unstable (poles outside unit circle)")


def witness_butter(
    order: int,
    wn: float,
    fs: float,
    btype: str = "low",
) -> AbstractFilterCoefficients:
    """Propagates metadata for scipy.signal.butter — designs a Butterworth digital filter, which has a maximally flat frequency response in the passband. Returns filter coefficients metadata for the specified order and cutoff frequency.

    Preconditions:
        - Order must be positive.
        - Critical frequency must be below Nyquist.
    Postconditions:
        - Returns filter coefficients metadata.
        - Butterworth filters are always stable.
    """
    if order <= 0:
        raise ValueError(f"Filter order must be positive, got {order}")
    nyquist = fs / 2.0
    if wn <= 0 or wn >= nyquist:
        raise ValueError(
            f"Critical frequency {wn} must be in (0, {nyquist}) for fs={fs}"
        )
    return AbstractFilterCoefficients(
        order=order, btype=btype, format="ba", is_stable=True,
    )


def witness_cheby1(
    order: int,
    rp: float,
    wn: float,
    fs: float,
    btype: str = "low",
) -> AbstractFilterCoefficients:
    """Propagates metadata for scipy.signal.cheby1 — designs a Chebyshev Type I digital filter, which trades passband ripple (small oscillations in the pass region) for a steeper roll-off than Butterworth.

    Preconditions:
        - Order must be positive.
        - Ripple must be positive.
        - Critical frequency must be below Nyquist.
    """
    if order <= 0:
        raise ValueError(f"Filter order must be positive, got {order}")
    if rp <= 0:
        raise ValueError(f"Passband ripple must be positive, got {rp}")
    nyquist = fs / 2.0
    if wn <= 0 or wn >= nyquist:
        raise ValueError(
            f"Critical frequency {wn} must be in (0, {nyquist}) for fs={fs}"
        )
    return AbstractFilterCoefficients(
        order=order, btype=btype, format="ba", is_stable=True,
    )


def witness_cheby2(
    order: int,
    rs: float,
    wn: float,
    fs: float,
    btype: str = "low",
) -> AbstractFilterCoefficients:
    """Propagates metadata for a digital filter with a flat pass region and a sharp cutoff into the rejected frequency region.

    Preconditions:
        - Order must be positive.
        - Rejected-band reduction must be positive.
        - Critical frequency must be below Nyquist.
    """
    if order <= 0:
        raise ValueError(f"Filter order must be positive, got {order}")
    if rs <= 0:
        raise ValueError(f"Stopband attenuation must be positive, got {rs}")
    nyquist = fs / 2.0
    if wn <= 0 or wn >= nyquist:
        raise ValueError(
            f"Critical frequency {wn} must be in (0, {nyquist}) for fs={fs}"
        )
    return AbstractFilterCoefficients(
        order=order, btype=btype, format="ba", is_stable=True,
    )


def witness_firwin(numtaps: int, fs: float) -> AbstractFilterCoefficients:
    """Propagates metadata for scipy.signal.firwin — designs a Finite Impulse Response (FIR) filter with the given number of taps. FIR filters use only past input values (no feedback), so they are always stable.

    Preconditions:
        - numtaps must be positive.
    Postconditions:
        - FIR filters are trivially stable (no feedback).
    """
    if numtaps <= 0:
        raise ValueError(f"numtaps must be positive, got {numtaps}")
    return AbstractFilterCoefficients(
        order=numtaps - 1, btype="low", format="fir", is_stable=True,
    )


# ---------------------------------------------------------------------------
# Filter application witnesses
# ---------------------------------------------------------------------------

def witness_lfilter(
    coefficients: AbstractFilterCoefficients,
    sig: AbstractSignal,
) -> AbstractSignal:
    """Propagates metadata for scipy.signal.lfilter — applies a linear digital filter to a time-domain signal using transfer-function (b, a) coefficients. The output signal has the same shape and sampling rate as the input.

    Preconditions:
        - Filter must be stable.
        - Signal must be in time domain.
    Postconditions:
        - Output shape matches input shape.
        - Output dtype matches input dtype.
        - Output domain is 'time'.
    """
    coefficients.assert_stable()
    sig.assert_domain("time")
    return AbstractSignal(
        shape=sig.shape,
        dtype=sig.dtype,
        sampling_rate=sig.sampling_rate,
        domain="time",
        units=sig.units,
    )


def witness_sosfilt(
    coefficients: AbstractFilterCoefficients,
    sig: AbstractSignal,
) -> AbstractSignal:
    """Propagates metadata for scipy.signal.sosfilt — applies a digital filter in second-order sections (SOS) form, which is more numerically stable than direct transfer-function form. Output shape and sampling rate match the input.

    Preconditions:
        - Signal must be in time domain.
    Postconditions:
        - Output shape matches input shape.
        - SOS format is stable by construction.
    """
    sig.assert_domain("time")
    return AbstractSignal(
        shape=sig.shape,
        dtype=sig.dtype,
        sampling_rate=sig.sampling_rate,
        domain="time",
        units=sig.units,
    )


# ---------------------------------------------------------------------------
# Analysis witnesses
# ---------------------------------------------------------------------------

def witness_peak_detect(sig: AbstractSignal) -> AbstractSignal:
    """Ghost witness for peak detection.

    Preconditions:
        - Input must be in time domain.
    Postconditions:
        - Output is a list of integer indices.
        - Shape is (0,) - dynamic length, unknown until runtime.
    """
    sig.assert_domain("time")
    return AbstractSignal(
        shape=(0,),
        dtype="int64",
        sampling_rate=sig.sampling_rate,
        domain="index",
        units="index",
    )


def witness_freqz(
    coefficients: AbstractFilterCoefficients,
    n_freqs: int = 512,
) -> AbstractSignal:
    """Propagates metadata for scipy.signal.freqz — computes the frequency response of a digital filter, returning complex gain values at n_freqs evenly spaced frequency points.

    Postconditions:
        - Output is a frequency response of length n_freqs.
        - Domain is 'freq'.
    """
    if n_freqs <= 0:
        raise ValueError(f"n_freqs must be positive, got {n_freqs}")
    return AbstractSignal(
        shape=(n_freqs,),
        dtype="complex128",
        sampling_rate=1.0,  # freqz returns normalized frequency
        domain="freq",
        units="magnitude",
    )


# ---------------------------------------------------------------------------
# Accumulator / state witnesses
# ---------------------------------------------------------------------------

def witness_sqi_update(
    pool: AbstractBeatPool,
    new_beats: AbstractSignal,
) -> AbstractBeatPool:
    """Ghost witness for Signal Quality Index (SQI) accumulation.

Simulates beat accumulation without processing waveforms.  Uses a
heuristic estimate of ~10 beats per window.

Preconditions:
    - new_beats must be in time domain.
Postconditions:
    - Pool size increases.
    - Calibration flag is set once threshold is reached."""
    new_beats.assert_domain("time")
    return pool.accumulate(new_beat_count=10)


# ---------------------------------------------------------------------------
# Graph Signal Processing witnesses
# ---------------------------------------------------------------------------

class AbstractGraphMeta:
    """Lightweight metadata for a graph (Laplacian / adjacency)."""

    def __init__(self, n_nodes: int, is_symmetric: bool = True) -> None:
        self.n_nodes = n_nodes
        self.is_symmetric = is_symmetric

    def assert_square(self) -> None:
        pass  # always square by construction

    def assert_symmetric(self) -> None:
        if not self.is_symmetric:
            raise ValueError("Graph matrix must be symmetric")


def witness_graph_laplacian(graph: AbstractGraphMeta) -> AbstractGraphMeta:
    """Ghost witness for graph_laplacian.

Preconditions:
    - Input must be symmetric.
Postconditions:
    - Output is a symmetric positive semi-definite (PSD) matrix of same size."""
    graph.assert_symmetric()
    return AbstractGraphMeta(n_nodes=graph.n_nodes, is_symmetric=True)


def witness_graph_fourier_transform(
    graph: AbstractGraphMeta,
    sig: AbstractSignal,
) -> AbstractSignal:
    """Ghost witness for graph_fourier_transform.

Preconditions:
    - Signal length must equal graph node count.
Postconditions:
    - Output is Graph Fourier Transform (GFT) coefficients of same length.
    - Domain switches to 'freq'."""
    if len(sig.shape) == 0 or sig.shape[0] != graph.n_nodes:
        raise ValueError(
            f"Signal length {sig.shape[0] if sig.shape else 0} "
            f"must equal graph size {graph.n_nodes}"
        )
    return AbstractSignal(
        shape=sig.shape,
        dtype="float64",
        sampling_rate=sig.sampling_rate,
        domain="freq",
        units="coefficient",
    )


def witness_inverse_graph_fourier_transform(
    sig: AbstractSignal,
    graph: AbstractGraphMeta,
) -> AbstractSignal:
    """Ghost witness for inverse_graph_fourier_transform.

    Preconditions:
        - Coefficient count must match number of eigenvectors.
    Postconditions:
        - Output length equals graph node count.
        - Domain switches to 'time'.
    """
    return AbstractSignal(
        shape=(graph.n_nodes,),
        dtype="float64",
        sampling_rate=sig.sampling_rate,
        domain="time",
        units=sig.units,
    )


def witness_heat_kernel_diffusion(
    graph: AbstractGraphMeta,
    sig: AbstractSignal,
    t: float,
) -> AbstractSignal:
    """Ghost witness for heat_kernel_diffusion.

    Preconditions:
        - t must be >= 0.
        - Signal length must equal graph node count.
    Postconditions:
        - Output shape matches input.
        - Output stays in the same domain.
        - Total variation is reduced (smoothing).
    """
    if t < 0:
        raise ValueError(f"Diffusion time must be >= 0, got {t}")
    if len(sig.shape) == 0 or sig.shape[0] != graph.n_nodes:
        raise ValueError(
            f"Signal length {sig.shape[0] if sig.shape else 0} "
            f"must equal graph size {graph.n_nodes}"
        )
    return AbstractSignal(
        shape=sig.shape,
        dtype=sig.dtype,
        sampling_rate=sig.sampling_rate,
        domain=sig.domain,
        units=sig.units,
    )


# ---------------------------------------------------------------------------
# Generic witnesses for non-DSP atoms (sorting, search, matrix ops)
# ---------------------------------------------------------------------------


def witness_sort(x: "AbstractArray") -> "AbstractArray":
    """Ghost witness for sorting atoms.

    Postconditions:
        - Output has same shape, dtype, value range.
        - Output is marked as sorted.
    """
    return AbstractArray(
        shape=x.shape,
        dtype=x.dtype,
        is_sorted=True,
        min_val=x.min_val,
        max_val=x.max_val,
    )


def witness_search(arr: "AbstractArray", key: "AbstractScalar") -> "AbstractScalar":
    """Ghost witness for search atoms (binary search, linear search).

    Postconditions:
        - Output is an index type.
        - Output range is [-1, len(arr)-1].
    """
    return AbstractScalar(
        dtype="int64",
        min_val=-1,
        max_val=float(arr.shape[0] - 1) if arr.shape else 0,
        is_index=True,
    )


def witness_matrix_op(A: "AbstractArray", B: "AbstractArray") -> "AbstractArray":
    """Ghost witness for matrix operations (multiply, solve, etc.).

    Postconditions:
        - Validates dimension compatibility.
        - Output shape is derived from input shapes.
    """
    if len(A.shape) < 2 or len(B.shape) < 2:
        raise ValueError(
            f"Matrix ops require 2D inputs, got shapes {A.shape} and {B.shape}"
        )
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"Incompatible matrix dimensions: {A.shape} @ {B.shape}"
        )
    return AbstractArray(
        shape=(A.shape[0], B.shape[1]),
        dtype=A.dtype,
    )


def witness_identity(x: AbstractArray) -> AbstractArray:
    """Pass-through witness for atoms with no structural constraints."""
    return x


# ---------------------------------------------------------------------------
# Bayesian / probabilistic inference witnesses
# ---------------------------------------------------------------------------


def witness_prior_init(
    event_shape: tuple[int, ...],
    family: str = "normal",
    rng: AbstractRNGState | None = None,
) -> AbstractDistribution:
    """Ghost witness for prior distribution initialization.

    Validates that the distribution family is known and event shape is
    non-empty, then returns an AbstractDistribution.

    Preconditions:
        - family must be a recognized distribution family.
        - event_shape must be non-empty.
    Postconditions:
        - Returns an AbstractDistribution with correct family and shape.
    """
    from sciona.ghost.abstract import DISTRIBUTION_FAMILIES

    if family not in DISTRIBUTION_FAMILIES:
        raise ValueError(
            f"Unknown distribution family '{family}'. "
            f"Known families: {sorted(DISTRIBUTION_FAMILIES)}"
        )
    if not event_shape or any(d <= 0 for d in event_shape):
        raise ValueError(
            f"event_shape must be non-empty with positive dims, got {event_shape}"
        )

    is_discrete = family in {"categorical", "bernoulli", "poisson"}
    support_lower = 0.0 if family in {
        "gamma", "exponential", "poisson", "log_normal",
    } else None
    support_upper = 1.0 if family in {"beta", "bernoulli"} else None
    if family == "beta":
        support_lower = 0.0

    return AbstractDistribution(
        family=family,
        event_shape=event_shape,
        is_discrete=is_discrete,
        support_lower=support_lower,
        support_upper=support_upper,
    )


def witness_log_prob(
    dist: AbstractDistribution,
    samples: AbstractArray,
) -> AbstractScalar:
    """Ghost witness for log-probability evaluation.

    Given a distribution and a batch of samples, validates that sample
    dimensions match the distribution's event shape and returns an
    AbstractScalar representing the log-probability.

    Preconditions:
        - Trailing dimensions of samples must match dist.event_shape.
    Postconditions:
        - Returns a scalar with dtype float64, range (-inf, 0].
    """
    # Last len(event_shape) dims of samples must match event_shape
    n_event = len(dist.event_shape)
    if n_event > 0:
        sample_tail = samples.shape[-n_event:]
        if sample_tail != dist.event_shape:
            raise ValueError(
                f"Sample trailing dims {sample_tail} don't match "
                f"distribution event_shape {dist.event_shape}"
            )

    return AbstractScalar(
        dtype="float64",
        max_val=0.0,  # log-prob is <= 0
    )


def witness_mcmc_step(
    trace: AbstractMCMCTrace,
    log_prob_fn: AbstractDistribution,
    rng: AbstractRNGState,
    step_size: float = 0.01,
) -> tuple[AbstractMCMCTrace, AbstractRNGState]:
    """Ghost witness for a generic MCMC step (Metropolis-Hastings / HMC).

Validates dimensionality compatibility between the trace's parameter
space and the target distribution, checks that RNG state is available,
and returns an updated trace and advanced RNG state.

Preconditions:
    - trace.param_dims must match log_prob_fn.event_shape.
    - step_size must be positive.
Postconditions:
    - Trace n_samples incremented by 1.
    - Warmup status updated.
    - RNG state advanced (1 draw consumed per step)."""
    # Dimensionality check: param space must match target distribution
    if trace.param_dims != log_prob_fn.event_shape:
        raise ValueError(
            f"MCMC parameter dims {trace.param_dims} don't match "
            f"target distribution event_shape {log_prob_fn.event_shape}"
        )

    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")

    # Advance trace by one step (assume accepted for ghost purposes)
    new_trace = trace.step(accepted=True)

    # Advance RNG state (one draw per MCMC step)
    new_rng = rng.advance(n_draws=1)

    return new_trace, new_rng


def witness_posterior_update(
    prior: AbstractDistribution,
    likelihood: AbstractDistribution,
    data_shape: tuple[int, ...],
) -> AbstractDistribution:
    """Ghost witness for a conjugate Bayesian posterior update.

    Validates that the prior-likelihood pair is conjugate and that data
    dimensions are compatible, then returns an AbstractDistribution
    representing the posterior (same family as the prior for conjugate
    updates).

    Preconditions:
        - prior must be conjugate to the likelihood.
        - data trailing dims must match likelihood.event_shape.
    Postconditions:
        - Posterior family matches prior family (conjugate closure).
        - Posterior event_shape matches prior event_shape.
    """
    # Validate conjugacy
    prior.assert_conjugate_to(likelihood)

    # Validate data dimensions match likelihood
    n_event = len(likelihood.event_shape)
    if n_event > 0 and len(data_shape) >= n_event:
        data_tail = data_shape[-n_event:]
        if data_tail != likelihood.event_shape:
            raise ValueError(
                f"Data trailing dims {data_tail} don't match "
                f"likelihood event_shape {likelihood.event_shape}"
            )

    # Conjugate update: posterior is same family as prior
    return AbstractDistribution(
        family=prior.family,
        event_shape=prior.event_shape,
        batch_shape=prior.batch_shape,
        support_lower=prior.support_lower,
        support_upper=prior.support_upper,
        is_discrete=prior.is_discrete,
    )


def witness_vi_elbo(
    q_dist: AbstractDistribution | tuple[AbstractDistribution, Any],
    p_dist: AbstractDistribution | tuple[AbstractDistribution, Any],
    n_samples: int = 1,
) -> AbstractScalar:
    """Ghost witness for Evidence Lower Bound (ELBO) computation. Validates that q and p have compatible shapes and that n_samples is positive.

Postconditions:
    - Returns a scalar (the ELBO estimate) with dtype float64."""
    # Extract distribution if passed as (dist, jacobian) tuple
    q = q_dist[0] if isinstance(q_dist, tuple) else q_dist
    p = p_dist[0] if isinstance(p_dist, tuple) else p_dist

    if q.event_shape != p.event_shape:
        raise ValueError(
            f"Variational q event_shape {q.event_shape} doesn't match "
            f"target p event_shape {p.event_shape}"
        )

    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    return AbstractScalar(dtype="float64")


# ---------------------------------------------------------------------------
# Advanced Bayesian / Filter witnesses
# ---------------------------------------------------------------------------

def witness_kalman_gain(
    P: AbstractMatrix,
    H: AbstractMatrix,
) -> AbstractMatrix:
    """Ghost witness for the Kalman Gain calculation.

    Simulates the matrix product K = P H^T (H P H^T + R)^{-1}.
    Preconditions:
        - P must be square (N, N).
        - H must be (M, N) to align with P's inner dimension.
    Postconditions:
        - Output gain K is (N, M).
    """
    if P.shape[0] != P.shape[1]:
        raise ValueError(f"Covariance P must be square, got {P.shape}")
    if H.shape[1] != P.shape[0]:
        raise ValueError(
            f"Inner dimension mismatch: H {H.shape} cannot multiply P {P.shape}"
        )
    return AbstractMatrix(shape=(P.shape[0], H.shape[0]), dtype=P.dtype)


def witness_bijector_transform(
    dist: AbstractDistribution,
) -> tuple[AbstractDistribution, AbstractSignal]:
    """Ghost witness for a bijector transformation (e.g., Logit/Exp).

    Transforms a constrained distribution into an unconstrained one
    while tracking the log-determinant Jacobian.

    Postconditions:
        - Returns unconstrained AbstractDistribution.
        - Returns AbstractSignal for the Jacobian log-determinant.
    """
    unconstrained = AbstractDistribution(
        family=dist.family,
        event_shape=dist.event_shape,
        batch_shape=dist.batch_shape,
        support="unconstrained",
        is_discrete=dist.is_discrete,
    )
    jacobian = AbstractSignal(
        shape=dist.event_shape,
        dtype="float64",
        sampling_rate=1.0,
        domain="index",
        units="log-determinant",
    )
    return unconstrained, jacobian


@register_atom(witness_kalman_gain)
def kalman_gain(P: AbstractMatrix, H: AbstractMatrix) -> AbstractMatrix:
    """Computes the Kalman gain — a weighting matrix that determines how much to trust new measurements versus the current state prediction. Multiplies the predicted error covariance P by the observation matrix H, scaled by the innovation covariance, to produce an optimal correction factor."""


@register_atom(witness_bijector_transform)
def bijector_transform(dist: AbstractDistribution) -> tuple[AbstractDistribution, AbstractSignal]:
    """Applies a bijector — an invertible, differentiable transformation — to a probability distribution. This remaps a simple base distribution (e.g., Gaussian) into a more expressive one while tracking the change-of-variables Jacobian needed for correct density evaluation."""


@register_atom(witness_vi_elbo)
def vi_elbo(q_dist: AbstractDistribution, p_dist: AbstractDistribution, n_samples: int = 1) -> AbstractScalar:
    """Estimates how well an approximate distribution matches the true target. Uses random draws to score the fit between q_dist and p_dist."""
