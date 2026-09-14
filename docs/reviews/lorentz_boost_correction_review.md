# Lorentz boost: source and invariant proof review

Status: source/proof and runtime validated; Tier 3 promotion pending.
Source version e2b3b291-b9f3-50bf-be6c-fe2d390f27b7, content hash
c03df95d9bff24d13e35624a4157a790cadd266c8679d4e6e25f919071aa52b4.
Fresh read-only audit checks 23 steps, 55 active bindings and 30 stored public
expressions against pinned files. No missing snapshots. Original remains draft;
the reconstruction does not certify 23 literal source transitions as valid.

Corrections include blank time-transform/cross-coefficient ASTs, scalar
multiplication parsed as function application, spatial-x aliases, and a factor
feed using the angle-like x identifier. Step 11 loses a squared coefficient and
uses gamma where velocity belongs in denominators. Step 22 divides by the
dimensionally invalid c²-gamma²; the corrected denominator is c²-v².

The derivation assumes a linear reciprocal standard boost along x, common
origin, unchanged transverse coordinates, and preservation of the light cone
for every null event. A single null event is insufficient for coefficient
comparison. The source branch divides by v and by 1-gamma²; establish v=0
separately by continuity from the identity. The x² coefficient alone admits
gamma²=1 for nonzero v, but that extra branch violates the time and cross
coefficients. Enforce all three conditions. Select positive gamma using the
identity limit and future time orientation, not squaring alone.

For c>0 and |v|<c, beta=v/c and gamma=1/sqrt(1-beta²). On coordinates
(ct,x,y,z), the standard matrix maps ct to gamma*(ct-beta*x), x to
gamma*(x-beta*ct), and leaves y,z unchanged. Ten algebraic checks verify
preservation of diag(1,-1,-1,-1), inverse under velocity reversal, determinant
one, zero-velocity identity, the positive-root square and identity value, all
three source coefficient conditions, and the reconstructed time correction.
Twelve synthetic tests cover these checks and mutations, the extraneous branch,
negative-root identity failure, superluminal/singular domain, and a mixed-sign
matrix that fails interval preservation.

The scope is a standard inertial x-axis boost. No acceleration, general
relativity, arbitrary boost direction or superluminal frame is claimed.
Independent rounding of numerical outputs will not preserve exact algebraic
invariants in every ill-conditioned case; runtime validation must reflect that.
Initial provider search found no Lorentz or rapidity implementation.

Reference checked: [OpenStax Lorentz transformation](https://openstax.org/books/university-physics-volume-3/pages/5-5-the-lorentz-transformation).
It supports the standard inertial-frame transformation and inverse; retained
algebraic checks supply the corrected source evidence.

Added the complete lorentz_boost provider with inputs light_speed,
relative_velocity, time, x, y, z and outputs gamma, time_prime, x_prime,
y_prime, z_prime. All are equal-shaped finite real arrays, including scalars,
without broadcasting. c must be positive and |v|<c. Signed event times and
positions are supported. Zero velocity is exactly the identity; transverse
coordinates are copied to independent outputs. The graph has one boost node
and eleven SI ports.

Exact Fraction intermediates form c²-v² and the coordinate subtraction terms
before local 450-digit square-root and multiplication evaluation. Outputs are
independently rounded to float64; exact zero and subnormals are accepted,
nonzero underflow and overflow rejected. This avoids losing cancellations due
to prematurely rounded intermediate products. It does not guarantee exact
interval preservation or inversion after rounding.

Thirty-seven combined proof/runtime tests pass. The independent 2,500-digit
Decimal oracle scales the null coordinates ct+x and ct-x by reciprocal Doppler
factors, then reconstructs time and x. It uses a different representation from
the direct runtime boost. Test cases include signed boosts, zero velocity,
near-light-speed null events, cancellation, extreme c and subnormal coordinates.
Two initial negative fixtures were still representable: they were retained as
positive cases, and stricter overflow/underflow fixtures added. No provider
behavior was weakened to accommodate those test corrections.

Six serialized full-runner cases cover 29 synthetic states with all five saved
outputs agreeing at zero ULP. Graph roundtrip and fresh source evidence pass.
The runner report is refreshed after the final test changes. Next: Tier3
promotion dry-run/apply, independent catalog verification, repeat-apply check.
