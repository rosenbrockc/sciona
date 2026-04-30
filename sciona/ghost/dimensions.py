"""Dimensional signature type system for physical quantity tracking.

Every physical quantity can be expressed as a product of powers of the
seven SI base dimensions.  ``DimensionalSignature`` stores those exponents
and provides arithmetic so that witnesses and the compiler can propagate
and verify dimensional consistency across CDG edges.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


class DimensionalSignature(BaseModel, frozen=True):
    """Product of SI base dimension exponents.

    For example, Power (W = kg·m²·s⁻³) is represented as
    ``DimensionalSignature(M=1, L=2, T=-3)``.

    All exponents default to 0 (dimensionless).
    """

    M: int = Field(default=0, description="Mass (kg)")
    L: int = Field(default=0, description="Length (m)")
    T: int = Field(default=0, description="Time (s)")
    I: int = Field(default=0, description="Electric current (A)")
    Theta: int = Field(default=0, description="Temperature (K)")
    N: int = Field(default=0, description="Amount of substance (mol)")
    J: int = Field(default=0, description="Luminous intensity (cd)")

    # ----- arithmetic -----

    def multiply(self, other: DimensionalSignature) -> DimensionalSignature:
        """Dimension of ``self * other`` (add exponents)."""
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
        return DimensionalSignature(
            M=self.M - other.M,
            L=self.L - other.L,
            T=self.T - other.T,
            I=self.I - other.I,
            Theta=self.Theta - other.Theta,
            N=self.N - other.N,
            J=self.J - other.J,
        )

    def power(self, n: int) -> DimensionalSignature:
        """Dimension of ``self ** n`` (scale exponents)."""
        return DimensionalSignature(
            M=self.M * n,
            L=self.L * n,
            T=self.T * n,
            I=self.I * n,
            Theta=self.Theta * n,
            N=self.N * n,
            J=self.J * n,
        )

    def is_compatible(self, other: DimensionalSignature) -> bool:
        """True when both signatures have identical exponents."""
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
        return self == DIMENSIONLESS

    # ----- serialisation -----

    def to_compact(self) -> str:
        """Compact string like ``"M1L2T-3"`` for storage in IOSpec."""
        if self.is_dimensionless:
            return "1"
        parts: list[str] = []
        for label, exp in [
            ("M", self.M), ("L", self.L), ("T", self.T),
            ("I", self.I), ("Th", self.Theta), ("N", self.N), ("J", self.J),
        ]:
            if exp != 0:
                parts.append(f"{label}{exp}")
        return "".join(parts) or "1"

    @classmethod
    def from_compact(cls, s: str) -> DimensionalSignature:
        """Parse a compact string back into a DimensionalSignature."""
        s = s.strip()
        if not s or s == "1":
            return DIMENSIONLESS
        pattern = re.compile(r"(Th|[MLTNIJ])(-?\d+)")
        kwargs: dict[str, int] = {}
        label_map = {"M": "M", "L": "L", "T": "T", "I": "I", "Th": "Theta", "N": "N", "J": "J"}
        for match in pattern.finditer(s):
            label, exp = match.group(1), int(match.group(2))
            field = label_map[label]
            kwargs[field] = exp
        return cls(**kwargs)

    def __repr__(self) -> str:
        return f"DimensionalSignature({self.to_compact()})"


# ---------------------------------------------------------------------------
# Predefined dimensional constants
# ---------------------------------------------------------------------------

DIMENSIONLESS = DimensionalSignature()

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
