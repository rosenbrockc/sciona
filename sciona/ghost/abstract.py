"""Abstract value types for Ghost Witness metadata propagation.

These models carry *only* metadata about signals and state - shape, dtype,
sampling rate, domain - never the actual sample data.  Witnesses consume
and produce these types to simulate a computation graph at near-zero cost.
"""

from __future__ import annotations

from typing import Tuple

from pydantic import BaseModel, Field


class AbstractSignal(BaseModel):
    """The 'Ghost' representation of a signal array.

    Carries the structural envelope of an ndarray without any sample data.
    Witnesses read and transform these to verify that a graph is wired
    correctly before any heavy computation runs.
    """

    shape: Tuple[int, ...] = Field(..., description="Array shape, e.g. (1024,) or (128, 2)")
    dtype: str = Field(..., description="NumPy dtype string, e.g. 'float64', 'complex128'")
    sampling_rate: float = Field(..., gt=0, description="Sampling frequency in Hz")
    domain: str = Field(default="time", description="Signal domain: 'time', 'freq', 'quefrency', 'index'")
    units: str = Field(default="volts", description="Physical units of the signal")

    @property
    def duration(self) -> float:
        """Signal duration in seconds (only meaningful for time-domain signals)."""
        if self.domain == "time" and self.sampling_rate > 0 and len(self.shape) > 0:
            return self.shape[0] / self.sampling_rate
        return 0.0

    @property
    def nyquist(self) -> float:
        """Nyquist frequency in Hz."""
        return self.sampling_rate / 2.0

    def assert_compatible(self, other: AbstractSignal) -> None:
        """Assert that two signals are compatible for element-wise operations.

        Raises:
            ValueError: If sampling rates or shapes don't match.
        """
        if self.sampling_rate != other.sampling_rate:
            raise ValueError(
                f"Sampling rate mismatch: {self.sampling_rate} vs {other.sampling_rate}"
            )
        if self.shape != other.shape:
            raise ValueError(
                f"Shape mismatch: {self.shape} vs {other.shape}"
            )

    def assert_domain(self, expected: str) -> None:
        """Assert that the signal is in the expected domain.

        Raises:
            ValueError: If the signal domain doesn't match.
        """
        if self.domain != expected:
            raise ValueError(
                f"Domain mismatch: expected '{expected}', got '{self.domain}'"
            )


class AbstractArray(BaseModel):
    """Generic abstract array type for non-DSP atoms.

    Propagates shape, dtype, and value-range constraints without domain-specific
    semantics (unlike AbstractSignal which carries sampling_rate and domain).
    """

    shape: Tuple[int, ...] = Field(..., description="Array shape")
    dtype: str = Field(default="float64", description="NumPy dtype string")
    is_sorted: bool = Field(default=False, description="Whether elements are sorted")
    min_val: float | None = Field(default=None, description="Minimum value constraint")
    max_val: float | None = Field(default=None, description="Maximum value constraint")

    def assert_shape_compatible(self, other: "AbstractArray") -> None:
        """Assert shapes are compatible for element-wise operations."""
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

    def assert_sorted(self) -> None:
        """Assert that the array is sorted."""
        if not self.is_sorted:
            raise ValueError("Array is not sorted")


class AbstractScalar(BaseModel):
    """Generic abstract scalar type for non-DSP atoms."""

    dtype: str = Field(default="int64", description="Scalar dtype")
    min_val: float | None = Field(default=None, description="Minimum value")
    max_val: float | None = Field(default=None, description="Maximum value")
    is_index: bool = Field(default=False, description="Whether this is an array index")


class AbstractMatrix(BaseModel):
    """Abstract representation of a matrix with symbolic dimensions.

    Handles generic dimensions like "N", "M" for shape-checking in atoms
    like Kalman filters where exact sizes are unknown until runtime but
    inner dimensions must align.
    """

    shape: Tuple[str, str] = Field(..., description="Symbolic shape, e.g. ('N', 'M')")
    dtype: str = Field(default="float64", description="NumPy dtype string")


class AbstractBeatPool(BaseModel):
    """Abstract state for accumulative beat detection / SQI pipelines.

    Models the evolving confidence state of a beat accumulator without
    storing any actual waveform data.
    """

    size: int = Field(default=0, ge=0, description="Number of beats accumulated so far")
    is_calibrated: bool = Field(default=False, description="Whether the pool has enough beats to be reliable")
    calibration_threshold: int = Field(default=50, description="Minimum beats required for calibration")

    def accumulate(self, new_beat_count: int) -> "AbstractBeatPool":
        """Return a new pool with additional beats accumulated.

        Args:
            new_beat_count: Number of new beats to add.

        Returns:
            Updated AbstractBeatPool with new size and calibration status.
        """
        new_size = self.size + new_beat_count
        return AbstractBeatPool(
            size=new_size,
            is_calibrated=new_size >= self.calibration_threshold,
            calibration_threshold=self.calibration_threshold,
        )


# ---------------------------------------------------------------------------
# Bayesian / probabilistic inference types
# ---------------------------------------------------------------------------

# Supported distribution families for AbstractDistribution.family
DISTRIBUTION_FAMILIES = frozenset({
    "normal", "multivariate_normal", "categorical", "dirichlet",
    "beta", "gamma", "exponential", "poisson", "bernoulli",
    "uniform", "log_normal", "student_t", "wishart", "inverse_wishart",
})

# Conjugate prior pairs: (likelihood_family, prior_family)
CONJUGATE_PAIRS = frozenset({
    ("normal", "normal"),               # Normal-Normal (known variance)
    ("normal", "inverse_wishart"),       # Normal-InverseWishart (unknown variance)
    ("bernoulli", "beta"),              # Bernoulli-Beta
    ("categorical", "dirichlet"),       # Categorical-Dirichlet
    ("poisson", "gamma"),               # Poisson-Gamma
    ("exponential", "gamma"),           # Exponential-Gamma
    ("normal", "gamma"),                # Normal-Gamma (precision)
})


class AbstractDistribution(BaseModel):
    """Abstract representation of a probability distribution.

    Carries family, shape, and support metadata without storing actual
    parameter values.  Witnesses use this to verify that Bayesian pipelines
    wire distributions of compatible shapes and families.
    """

    family: str = Field(
        ..., description="Distribution family, e.g. 'normal', 'dirichlet'"
    )
    event_shape: Tuple[int, ...] = Field(
        ..., description="Shape of a single draw, e.g. (3,) for 3D normal"
    )
    batch_shape: Tuple[int, ...] = Field(
        default=(), description="Shape of independent distributions, e.g. (100,)"
    )
    support: str = Field(
        default="unconstrained", description="Distribution support: 'positive', 'simplex', 'unconstrained', etc."
    )
    support_lower: float | None = Field(
        default=None, description="Lower bound of support (None = unbounded)"
    )
    support_upper: float | None = Field(
        default=None, description="Upper bound of support (None = unbounded)"
    )
    is_discrete: bool = Field(
        default=False, description="Whether the distribution is discrete"
    )

    def assert_family(self, expected: str) -> None:
        """Assert the distribution belongs to the expected family."""
        if self.family != expected:
            raise ValueError(
                f"Distribution family mismatch: expected '{expected}', "
                f"got '{self.family}'"
            )

    def assert_event_shape(self, expected: Tuple[int, ...]) -> None:
        """Assert the event shape matches."""
        if self.event_shape != expected:
            raise ValueError(
                f"Event shape mismatch: expected {expected}, "
                f"got {self.event_shape}"
            )

    def assert_conjugate_to(self, likelihood: "AbstractDistribution") -> None:
        """Assert this distribution is a valid conjugate prior for the likelihood."""
        pair = (likelihood.family, self.family)
        if pair not in CONJUGATE_PAIRS:
            raise ValueError(
                f"'{self.family}' is not a conjugate prior for "
                f"'{likelihood.family}' likelihood. "
                f"Known conjugate pairs: {sorted(CONJUGATE_PAIRS)}"
            )


class AbstractRNGState(BaseModel):
    """Abstract representation of a random number generator state.

    Tracks seed lineage and consumption count so that witnesses can
    verify that RNG states are properly split/forked and never reused.
    """

    seed: int = Field(..., description="Initial seed or key")
    consumed: int = Field(
        default=0, ge=0, description="Number of draws consumed from this state"
    )
    is_split: bool = Field(
        default=False,
        description="Whether this state was produced by a split/fork operation",
    )

    def advance(self, n_draws: int) -> "AbstractRNGState":
        """Return a new state after consuming n_draws."""
        if n_draws <= 0:
            raise ValueError(f"n_draws must be positive, got {n_draws}")
        return AbstractRNGState(
            seed=self.seed,
            consumed=self.consumed + n_draws,
            is_split=self.is_split,
        )

    def split(self) -> Tuple["AbstractRNGState", "AbstractRNGState"]:
        """Return two independent child states (JAX-style key splitting)."""
        return (
            AbstractRNGState(seed=self.seed, consumed=self.consumed, is_split=True),
            AbstractRNGState(seed=self.seed + 1, consumed=self.consumed, is_split=True),
        )


class AbstractMCMCTrace(BaseModel):
    """Abstract representation of an MCMC chain trace.

    Carries the structural metadata of accumulated samples without
    storing actual parameter values.  Witnesses verify dimensionality
    and chain health.
    """

    n_samples: int = Field(default=0, ge=0, description="Number of samples drawn so far")
    n_chains: int = Field(default=1, ge=1, description="Number of parallel chains")
    param_dims: Tuple[int, ...] = Field(
        ..., description="Shape of each parameter sample, e.g. (3,) for 3D"
    )
    warmup_steps: int = Field(default=0, ge=0, description="Number of warmup/burn-in steps")
    is_warmed_up: bool = Field(
        default=False, description="Whether warmup phase is complete"
    )
    accept_rate: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Running acceptance rate (0 before first sample)",
    )

    def assert_warmed_up(self) -> None:
        """Assert that the chain has completed warmup."""
        if not self.is_warmed_up:
            raise ValueError(
                f"Chain not warmed up: {self.n_samples} samples drawn, "
                f"{self.warmup_steps} warmup steps required"
            )

    def assert_param_dims(self, expected: Tuple[int, ...]) -> None:
        """Assert parameter dimensions match."""
        if self.param_dims != expected:
            raise ValueError(
                f"Parameter dimension mismatch: expected {expected}, "
                f"got {self.param_dims}"
            )

    def step(self, accepted: bool = True) -> "AbstractMCMCTrace":
        """Return a new trace after one MCMC step.

        Updates sample count, warmup status, and running acceptance rate
        using an exponential moving average.
        """
        new_n = self.n_samples + 1
        alpha = 2.0 / (new_n + 1)
        new_rate = alpha * (1.0 if accepted else 0.0) + (1 - alpha) * self.accept_rate
        return AbstractMCMCTrace(
            n_samples=new_n,
            n_chains=self.n_chains,
            param_dims=self.param_dims,
            warmup_steps=self.warmup_steps,
            is_warmed_up=new_n >= self.warmup_steps,
            accept_rate=round(new_rate, 6),
        )

# Concrete sentinel for Ghost Witness propagation.
ANYTHING = "ANYTHING"
