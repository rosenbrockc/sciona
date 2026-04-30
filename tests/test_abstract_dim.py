"""Tests for dimensional analysis integration in ghost abstract types."""

from __future__ import annotations

import pytest

from sciona.ghost.abstract import (
    AbstractArray,
    AbstractMatrix,
    AbstractScalar,
    AbstractSignal,
)
from sciona.ghost.dimensions import (
    DIMENSIONLESS,
    HERTZ,
    METER,
    VOLT,
    WATT,
)


# ---------------------------------------------------------------------------
# AbstractSignal dim field
# ---------------------------------------------------------------------------


class TestAbstractSignalDim:
    def test_default_dim_is_none(self):
        sig = AbstractSignal(shape=(1024,), dtype="float64", sampling_rate=256.0)
        assert sig.dim is None

    def test_dim_can_be_set(self):
        sig = AbstractSignal(
            shape=(1024,), dtype="float64", sampling_rate=256.0, dim=VOLT,
        )
        assert sig.dim == VOLT

    def test_assert_compatible_ignores_when_dim_none(self):
        a = AbstractSignal(shape=(100,), dtype="float64", sampling_rate=100.0, dim=VOLT)
        b = AbstractSignal(shape=(100,), dtype="float64", sampling_rate=100.0, dim=None)
        # Should not raise — one side is None
        a.assert_compatible(b)

    def test_assert_compatible_passes_same_dim(self):
        a = AbstractSignal(shape=(100,), dtype="float64", sampling_rate=100.0, dim=WATT)
        b = AbstractSignal(shape=(100,), dtype="float64", sampling_rate=100.0, dim=WATT)
        a.assert_compatible(b)

    def test_assert_compatible_raises_dim_mismatch(self):
        a = AbstractSignal(shape=(100,), dtype="float64", sampling_rate=100.0, dim=WATT)
        b = AbstractSignal(shape=(100,), dtype="float64", sampling_rate=100.0, dim=VOLT)
        with pytest.raises(ValueError, match="Dimensional mismatch"):
            a.assert_compatible(b)

    def test_assert_compatible_still_checks_shape(self):
        a = AbstractSignal(shape=(100,), dtype="float64", sampling_rate=100.0, dim=WATT)
        b = AbstractSignal(shape=(200,), dtype="float64", sampling_rate=100.0, dim=WATT)
        with pytest.raises(ValueError, match="Shape mismatch"):
            a.assert_compatible(b)

    def test_assert_compatible_still_checks_sampling_rate(self):
        a = AbstractSignal(shape=(100,), dtype="float64", sampling_rate=100.0, dim=WATT)
        b = AbstractSignal(shape=(100,), dtype="float64", sampling_rate=200.0, dim=WATT)
        with pytest.raises(ValueError, match="Sampling rate mismatch"):
            a.assert_compatible(b)


# ---------------------------------------------------------------------------
# AbstractArray dim field
# ---------------------------------------------------------------------------


class TestAbstractArrayDim:
    def test_default_none(self):
        arr = AbstractArray(shape=(10,))
        assert arr.dim is None

    def test_can_set_dim(self):
        arr = AbstractArray(shape=(10,), dim=METER)
        assert arr.dim == METER


# ---------------------------------------------------------------------------
# AbstractScalar dim field
# ---------------------------------------------------------------------------


class TestAbstractScalarDim:
    def test_default_none(self):
        s = AbstractScalar()
        assert s.dim is None

    def test_can_set_dim(self):
        s = AbstractScalar(dtype="float64", dim=HERTZ)
        assert s.dim == HERTZ


# ---------------------------------------------------------------------------
# AbstractMatrix dim field
# ---------------------------------------------------------------------------


class TestAbstractMatrixDim:
    def test_default_none(self):
        m = AbstractMatrix(shape=("N", "M"))
        assert m.dim is None

    def test_can_set_dim(self):
        m = AbstractMatrix(shape=("N", "M"), dim=DIMENSIONLESS)
        assert m.dim == DIMENSIONLESS


# ---------------------------------------------------------------------------
# Backward compatibility: existing code that doesn't pass dim still works
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_signal_without_dim(self):
        sig = AbstractSignal(
            shape=(512,), dtype="float64", sampling_rate=44100.0,
            domain="time", units="volts",
        )
        assert sig.dim is None
        assert sig.units == "volts"

    def test_two_signals_without_dim_compatible(self):
        a = AbstractSignal(shape=(100,), dtype="float64", sampling_rate=100.0)
        b = AbstractSignal(shape=(100,), dtype="float64", sampling_rate=100.0)
        a.assert_compatible(b)
