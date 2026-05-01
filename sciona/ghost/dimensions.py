"""Dimensional signature type system for physical quantity tracking.

Every physical quantity can be expressed as a product of powers of the
seven SI base dimensions.  ``DimensionalSignature`` stores those exponents
and provides arithmetic so that witnesses and the compiler can propagate
and verify dimensional consistency across CDG edges.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

from pydantic import BaseModel, Field, field_validator


DimensionExponent = Fraction


def _coerce_exponent(value: Any) -> Fraction:
    """Coerce supported exponent representations to an exact Fraction."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        return Fraction(int(value), 1)
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        return Fraction(str(value))
    if isinstance(value, str):
        return Fraction(value.strip())

    # SymPy Rational/Integer expose exact ``p``/``q`` attributes.
    numerator = getattr(value, "p", None)
    denominator = getattr(value, "q", None)
    if numerator is not None and denominator is not None:
        return Fraction(int(numerator), int(denominator))

    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if numerator is not None and denominator is not None:
        return Fraction(int(numerator), int(denominator))

    return Fraction(value)


def _format_exponent(exp: Fraction) -> str:
    """Deterministic compact representation for an exact exponent."""
    if exp.denominator == 1:
        return str(exp.numerator)
    return f"{exp.numerator}/{exp.denominator}"


class DimensionalSignature(BaseModel, frozen=True):
    """Product of SI base dimension exponents.

    For example, Power (W = kg·m²·s⁻³) is represented as
    ``DimensionalSignature(M=1, L=2, T=-3)``.

    All exponents default to 0 (dimensionless).  Exponents are exact
    rational numbers so signatures such as sqrt(length) can be represented
    without float drift.
    """

    M: DimensionExponent = Field(default=Fraction(0), description="Mass (kg)")
    L: DimensionExponent = Field(default=Fraction(0), description="Length (m)")
    T: DimensionExponent = Field(default=Fraction(0), description="Time (s)")
    I: DimensionExponent = Field(default=Fraction(0), description="Electric current (A)")  # noqa: E741
    Theta: DimensionExponent = Field(default=Fraction(0), description="Temperature (K)")
    N: DimensionExponent = Field(default=Fraction(0), description="Amount of substance (mol)")
    J: DimensionExponent = Field(default=Fraction(0), description="Luminous intensity (cd)")
    unknown: bool = Field(
        default=False,
        description="True when a source explicitly marks dimensionality as unknown.",
    )

    @field_validator("M", "L", "T", "I", "Theta", "N", "J", mode="before")
    @classmethod
    def _validate_exponent(cls, value: Any) -> Fraction:
        return _coerce_exponent(value)

    # ----- arithmetic -----

    def multiply(self, other: DimensionalSignature) -> DimensionalSignature:
        """Dimension of ``self * other`` (add exponents)."""
        if self.is_unknown or other.is_unknown:
            return UNKNOWN_DIMENSION
        return DimensionalSignature(
            M=self.M + other.M,
            L=self.L + other.L,
            T=self.T + other.T,
            I=self.I + other.I,
            Theta=self.Theta + other.Theta,
            N=self.N + other.N,
            J=self.J + other.J,
        )

    def divide(self, other: DimensionalSignature) -> DimensionalSignature:
        """Dimension of ``self / other`` (subtract exponents)."""
        if self.is_unknown or other.is_unknown:
            return UNKNOWN_DIMENSION
        return DimensionalSignature(
            M=self.M - other.M,
            L=self.L - other.L,
            T=self.T - other.T,
            I=self.I - other.I,
            Theta=self.Theta - other.Theta,
            N=self.N - other.N,
            J=self.J - other.J,
        )

    def power(self, n: int | Fraction | str | float) -> DimensionalSignature:
        """Dimension of ``self ** n`` (scale exponents)."""
        if self.is_unknown:
            return UNKNOWN_DIMENSION
        exponent = _coerce_exponent(n)
        return DimensionalSignature(
            M=self.M * exponent,
            L=self.L * exponent,
            T=self.T * exponent,
            I=self.I * exponent,
            Theta=self.Theta * exponent,
            N=self.N * exponent,
            J=self.J * exponent,
        )

    def is_compatible(self, other: DimensionalSignature) -> bool:
        """True when both signatures have identical exponents."""
        if self.is_unknown or other.is_unknown:
            return False
        return (
            self.M == other.M
            and self.L == other.L
            and self.T == other.T
            and self.I == other.I
            and self.Theta == other.Theta
            and self.N == other.N
            and self.J == other.J
        )

    @property
    def is_dimensionless(self) -> bool:
        return (
            not self.is_unknown
            and self.M == self.L == self.T == self.I == self.Theta == self.N == self.J == 0
        )

    @property
    def is_unknown(self) -> bool:
        return self.unknown

    # ----- serialisation -----

    def to_compact(self) -> str:
        """Compact string like ``"M1L2T-3"`` for storage in IOSpec."""
        if self.is_unknown:
            return "?"
        if self.is_dimensionless:
            return "1"
        parts: list[str] = []
        for label, exp in [
            ("M", self.M), ("L", self.L), ("T", self.T),
            ("I", self.I), ("Th", self.Theta), ("N", self.N), ("J", self.J),
        ]:
            if exp != 0:
                parts.append(f"{label}{_format_exponent(exp)}")
        return "".join(parts) or "1"

    @classmethod
    def from_compact(cls, s: str) -> DimensionalSignature:
        """Parse a compact string back into a DimensionalSignature."""
        s = s.strip()
        if not s or s == "1":
            return DIMENSIONLESS
        if s in {"?", "unknown", "UNKNOWN"}:
            return UNKNOWN_DIMENSION
        pattern = re.compile(r"(Th|[MLTNIJ])(-?\d+(?:/\d+)?)")
        kwargs: dict[str, Fraction] = {}
        label_map = {"M": "M", "L": "L", "T": "T", "I": "I", "Th": "Theta", "N": "N", "J": "J"}
        matches = list(pattern.finditer(s))
        consumed = "".join(match.group(0) for match in matches)
        if consumed != s:
            raise ValueError(f"unsupported compact dimensional signature: {s!r}")
        for match in matches:
            label, exp = match.group(1), _coerce_exponent(match.group(2))
            field = label_map[label]
            kwargs[field] = exp
        return cls(**kwargs)

    def __repr__(self) -> str:
        return f"DimensionalSignature({self.to_compact()})"


# ---------------------------------------------------------------------------
# Predefined dimensional constants
# ---------------------------------------------------------------------------

DIMENSIONLESS = DimensionalSignature()
UNKNOWN_DIMENSION = DimensionalSignature(unknown=True)

# Base dimensions
METER = DimensionalSignature(L=1)
KILOGRAM = DimensionalSignature(M=1)
SECOND = DimensionalSignature(T=1)
AMPERE = DimensionalSignature(I=1)
KELVIN = DimensionalSignature(Theta=1)
MOLE = DimensionalSignature(N=1)
CANDELA = DimensionalSignature(J=1)

# Derived SI dimensions
HERTZ = DimensionalSignature(T=-1)                         # 1/s
NEWTON = DimensionalSignature(M=1, L=1, T=-2)              # kg·m/s²
PASCAL = DimensionalSignature(M=1, L=-1, T=-2)             # N/m² = kg/(m·s²)
JOULE = DimensionalSignature(M=1, L=2, T=-2)               # N·m = kg·m²/s²
WATT = DimensionalSignature(M=1, L=2, T=-3)                # J/s = kg·m²/s³
COULOMB = DimensionalSignature(T=1, I=1)                    # A·s
VOLT = DimensionalSignature(M=1, L=2, T=-3, I=-1)          # W/A = kg·m²/(A·s³)
FARAD = DimensionalSignature(M=-1, L=-2, T=4, I=2)         # C/V
OHM = DimensionalSignature(M=1, L=2, T=-3, I=-2)           # V/A
SIEMENS = DimensionalSignature(M=-1, L=-2, T=3, I=2)       # 1/Ω
WEBER = DimensionalSignature(M=1, L=2, T=-2, I=-1)         # V·s
TESLA = DimensionalSignature(M=1, T=-2, I=-1)              # Wb/m²
HENRY = DimensionalSignature(M=1, L=2, T=-2, I=-2)         # Wb/A

# Common compound dimensions
VELOCITY = DimensionalSignature(L=1, T=-1)                  # m/s
ACCELERATION = DimensionalSignature(L=1, T=-2)              # m/s²
AREA = DimensionalSignature(L=2)                            # m²
VOLUME = DimensionalSignature(L=3)                          # m³
DENSITY = DimensionalSignature(M=1, L=-3)                   # kg/m³
PRESSURE = PASCAL
ENERGY = JOULE
POWER = WATT
FORCE = NEWTON
FREQUENCY = HERTZ
ELECTRIC_POTENTIAL = VOLT
RESISTANCE = OHM
CAPACITANCE = FARAD
INDUCTANCE = HENRY
MAGNETIC_FLUX = WEBER
MAGNETIC_FLUX_DENSITY = TESLA

# Angles (dimensionless in SI, but useful to track semantically)
RADIAN = DIMENSIONLESS
STERADIAN = DIMENSIONLESS


# ---------------------------------------------------------------------------
# Free-form unit string parser (best-effort, for migration)
# ---------------------------------------------------------------------------

_UNIT_STRING_MAP: dict[str, DimensionalSignature] = {
    # base
    "m": METER, "meter": METER, "meters": METER, "metre": METER, "metres": METER,
    "kg": KILOGRAM, "kilogram": KILOGRAM, "kilograms": KILOGRAM,
    "s": SECOND, "sec": SECOND, "second": SECOND, "seconds": SECOND,
    "a": AMPERE, "amp": AMPERE, "ampere": AMPERE, "amperes": AMPERE,
    "k": KELVIN, "kelvin": KELVIN,
    "mol": MOLE, "mole": MOLE,
    "cd": CANDELA, "candela": CANDELA,
    # derived
    "hz": HERTZ, "hertz": HERTZ, "frequency": HERTZ,
    "n": NEWTON, "newton": NEWTON, "newtons": NEWTON, "force": NEWTON,
    "pa": PASCAL, "pascal": PASCAL, "pressure": PASCAL,
    "j": JOULE, "joule": JOULE, "joules": JOULE, "energy": JOULE,
    "w": WATT, "watt": WATT, "watts": WATT, "power": WATT,
    "c": COULOMB, "coulomb": COULOMB, "coulombs": COULOMB, "charge": COULOMB,
    "v": VOLT, "volt": VOLT, "volts": VOLT, "voltage": VOLT,
    "f": FARAD, "farad": FARAD, "farads": FARAD, "capacitance": FARAD,
    "ohm": OHM, "ohms": OHM, "resistance": OHM,
    "wb": WEBER, "weber": WEBER,
    "t": TESLA, "tesla": TESLA,
    "h": HENRY, "henry": HENRY, "henrys": HENRY, "inductance": HENRY,
    # compound
    "m/s": VELOCITY, "velocity": VELOCITY, "speed": VELOCITY,
    "m/s2": ACCELERATION, "m/s^2": ACCELERATION, "acceleration": ACCELERATION,
    # dimensionless
    "dimensionless": DIMENSIONLESS, "unitless": DIMENSIONLESS,
    "ratio": DIMENSIONLESS, "fraction": DIMENSIONLESS,
    "rad": DIMENSIONLESS, "radian": DIMENSIONLESS, "radians": DIMENSIONLESS,
    "sr": DIMENSIONLESS, "steradian": DIMENSIONLESS,
    "db": DIMENSIONLESS, "decibel": DIMENSIONLESS, "decibels": DIMENSIONLESS,
    "coefficient": DIMENSIONLESS,
}


def parse_units_string(s: str) -> DimensionalSignature | None:
    """Best-effort parse of a free-form units string.

    Returns ``None`` when the string is not recognised, allowing callers to
    distinguish "unknown" from "intentionally dimensionless".
    """
    if not s:
        return None
    key = s.strip().lower().replace("_", " ").replace("-", " ")
    # Try direct lookup first
    if key in _UNIT_STRING_MAP:
        return _UNIT_STRING_MAP[key]
    # Try with common prefixes stripped (e.g. "normalized_power" -> "power")
    for prefix in ("normalized ", "relative ", "absolute ", "mean ", "rms ", "peak "):
        if key.startswith(prefix):
            suffix = key[len(prefix):]
            if suffix in _UNIT_STRING_MAP:
                return _UNIT_STRING_MAP[suffix]
    return None
