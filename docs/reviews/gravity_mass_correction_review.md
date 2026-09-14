# Spherical-source gravitational mass reconstruction

Source284349d7-2884-5455-b8f1-d6b7897d91ee, version
8d6afa2c-187c-5852-a936-815d8416125d remains draft. Fresh audit checks ten steps,
twenty-one bindings, ten stored equations and two recovered equations5345738321
and6935745841. All three source pins and exact equation fields/feeds checked.

Source renaming uses target mass identity twice instead of renaming m1=pdg0005022
to source mass=pdg0005458. Unit feeds encode acceleration as(m/s)^2, omit kg^-1
from G, and encode the meter unit as a mass symbol. Numerical ASTs truncate9.80665
to9, leave a substituted expression empty, or reduce the final computation to6.3781.
Reconstruct from displayed LaTeX with explicit SI quantities and exact decimals.

Newtonian force equality and positive test-mass cancellation give g=GM/r²,
then M=g*r²/G for positive gravitational acceleration, radius and G. Spherical
symmetry and a surface/exterior point are required. This is pure gravitational
acceleration, not effective gravity including rotation; test mass is nonzero.

The displayed g=9.80665m/s²,r=6378100m,G=6.67430e-11m³/kg/s² yield
5.977197417548005034235800e24kg, not the source5.972e24kg. The discrepancy is
not final rounding. Retain the computed result and flag the original endpoint.
Standard gravity and equatorial radius do not form a precision Earth ephemeris.

Thirteen synthetic proof tests pass: ten mutated steps, independent Decimal
endpoint and effective-gravity counterexample. Runtime and Tier3 remain pending.
Implement generic positive g,r,G -> inferred mass (and optionally gravitational
parameter g*r² if full output scope is clear), exact rational arithmetic before
rounding. Include full source constants in serialized execution evidence. Never
hard-code the catalog Earth mass to make the source endpoint pass.

Runtime completion:35 combined tests and6 serialized runner cases/29 synthetic
states pass. Exact Fraction computation on converted binary64 inputs, one output
rounding, independently verified with Decimal2500. Full source example produces
5.9771974e24kg and explicitly fails to reproduce the wrong5.972e24 endpoint.
Intermediate overflow/underflow, subnormal output, shape and domain checks pass.
Reference: OpenStax University Physics Volume1,13.2 Gravitation Near Earth Surface:
https://openstax.org/books/university-physics-volume-1/pages/13-2-gravitation-near-earths-surface
Source-stage pending notes are historical; final approval is recorded separately.
