"""Tests for sciona.ghost.dimensions – DimensionalSignature type system."""

from __future__ import annotations

import pytest

from sciona.ghost.dimensions import (
    ACCELERATION,
    AMPERE,
    COULOMB,
    DIMENSIONLESS,
    ENERGY,
    FARAD,
    FORCE,
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
