"""Tests for sciona.ghost.dimensions – DimensionalSignature type system."""

from __future__ import annotations

from fractions import Fraction

import pytest

from sciona.ghost.dimensions import (
    ACCELERATION,
    AMPERE,
    COULOMB,
    DIMENSIONLESS,
    FARAD,
    HERTZ,
    JOULE,
    KELVIN,
    KILOGRAM,
    METER,
    NEWTON,
    OHM,
    PASCAL,
    POWER,
    SECOND,
    UNKNOWN_DIMENSION,
    VELOCITY,
    VOLT,
    WATT,
    DimensionalSignature,
    parse_units_string,
)


# ---------------------------------------------------------------------------
# Construction & identity
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_dimensionless_default(self):
        d = DimensionalSignature()
        assert d == DIMENSIONLESS
        assert d.is_dimensionless

    def test_meter(self):
        assert METER == DimensionalSignature(L=1)
        assert not METER.is_dimensionless

    def test_frozen(self):
        with pytest.raises(Exception):
            METER.L = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


class TestArithmetic:
    def test_multiply(self):
        # velocity = length / time, but via multiply: m * (1/s) = m/s
        inv_s = SECOND.power(-1)
        assert METER.multiply(inv_s) == VELOCITY

    def test_divide(self):
        # velocity / time = acceleration
        assert VELOCITY.divide(SECOND) == ACCELERATION

    def test_power(self):
        assert METER.power(2) == DimensionalSignature(L=2)
        assert METER.power(0) == DIMENSIONLESS

    def test_fractional_power(self):
        sqrt_meter = METER.power(Fraction(1, 2))
        assert sqrt_meter == DimensionalSignature(L=Fraction(1, 2))
        assert sqrt_meter.multiply(sqrt_meter) == METER

    def test_unknown_dimension_propagates_through_arithmetic(self):
        assert UNKNOWN_DIMENSION.multiply(METER).is_unknown
        assert METER.divide(UNKNOWN_DIMENSION).is_unknown
        assert UNKNOWN_DIMENSION.power(2).is_unknown

    def test_force_composition(self):
        # F = m * a => kg * m/s²
        assert KILOGRAM.multiply(ACCELERATION) == NEWTON

    def test_energy_composition(self):
        # E = F * d => N * m = J
        assert NEWTON.multiply(METER) == JOULE

    def test_power_composition(self):
        # P = E / t => J / s = W
        assert JOULE.divide(SECOND) == WATT

    def test_voltage_composition(self):
        # V = W / A
        assert WATT.divide(AMPERE) == VOLT

    def test_ohm_composition(self):
        # Ω = V / A
        assert VOLT.divide(AMPERE) == OHM

    def test_coulomb_composition(self):
        # C = A * s
        assert AMPERE.multiply(SECOND) == COULOMB

    def test_farad_composition(self):
        # F = C / V
        assert COULOMB.divide(VOLT) == FARAD


class TestCompatibility:
    def test_same_is_compatible(self):
        assert WATT.is_compatible(POWER)

    def test_different_not_compatible(self):
        assert not WATT.is_compatible(VOLT)

    def test_dimensionless_self_compatible(self):
        assert DIMENSIONLESS.is_compatible(DimensionalSignature())

    def test_unknown_not_compatible_with_known_or_unknown(self):
        assert not UNKNOWN_DIMENSION.is_compatible(METER)
        assert not UNKNOWN_DIMENSION.is_compatible(UNKNOWN_DIMENSION)


# ---------------------------------------------------------------------------
# Serialisation roundtrip
# ---------------------------------------------------------------------------


class TestCompactSerialization:
    def test_dimensionless(self):
        assert DIMENSIONLESS.to_compact() == "1"
        assert DimensionalSignature.from_compact("1") == DIMENSIONLESS

    def test_power(self):
        s = WATT.to_compact()
        assert "M1" in s
        assert "L2" in s
        assert "T-3" in s
        assert DimensionalSignature.from_compact(s) == WATT

    def test_roundtrip_all_predefined(self):
        for dim in [METER, KILOGRAM, SECOND, AMPERE, KELVIN, HERTZ,
                     NEWTON, PASCAL, JOULE, WATT, COULOMB, VOLT,
                     FARAD, OHM, VELOCITY, ACCELERATION]:
            assert DimensionalSignature.from_compact(dim.to_compact()) == dim

    def test_fractional_compact_roundtrip(self):
        dim = DimensionalSignature(L=Fraction(1, 2), T=Fraction(-3, 2))
        assert dim.to_compact() == "L1/2T-3/2"
        assert DimensionalSignature.from_compact("L1/2T-3/2") == dim

    def test_rational_compact_roundtrip_preserves_all_axes(self):
        dim = DimensionalSignature(
            M=Fraction(5, 3),
            L=Fraction(-1, 2),
            T=Fraction(7, 4),
            I=Fraction(-2, 5),
            Theta=Fraction(3, 8),
            N=Fraction(-4, 9),
            J=Fraction(11, 6),
        )

        compact = dim.to_compact()

        assert compact == "M5/3L-1/2T7/4I-2/5Th3/8N-4/9J11/6"
        assert DimensionalSignature.from_compact(compact) == dim

    def test_invalid_compact_signature_does_not_become_dimensionless(self):
        with pytest.raises(ValueError):
            DimensionalSignature.from_compact("not-a-dimension")
        with pytest.raises(ValueError):
            DimensionalSignature.from_compact("L1/2unknown")

    def test_unknown_compact_roundtrip(self):
        assert UNKNOWN_DIMENSION.to_compact() == "?"
        assert DimensionalSignature.from_compact("?") == UNKNOWN_DIMENSION
        assert DimensionalSignature.from_compact("unknown") == UNKNOWN_DIMENSION

    def test_empty_string(self):
        assert DimensionalSignature.from_compact("") == DIMENSIONLESS


# ---------------------------------------------------------------------------
# parse_units_string
# ---------------------------------------------------------------------------


class TestParseUnitsString:
    def test_known_strings(self):
        assert parse_units_string("volts") == VOLT
        assert parse_units_string("power") == WATT
        assert parse_units_string("Hz") == HERTZ
        assert parse_units_string("seconds") == SECOND
        assert parse_units_string("meter") == METER
        assert parse_units_string("kelvin") == KELVIN

    def test_normalized_prefix(self):
        assert parse_units_string("normalized_power") == WATT
        assert parse_units_string("rms_power") == WATT

    def test_dimensionless_strings(self):
        assert parse_units_string("dimensionless") == DIMENSIONLESS
        assert parse_units_string("ratio") == DIMENSIONLESS
        assert parse_units_string("coefficient") == DIMENSIONLESS

    def test_unknown_returns_none(self):
        assert parse_units_string("frobnicators") is None
        assert parse_units_string("") is None

    def test_case_insensitive(self):
        assert parse_units_string("VOLTS") == VOLT
        assert parse_units_string("Hertz") == HERTZ


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


def test_repr():
    r = repr(WATT)
    assert "DimensionalSignature" in r
