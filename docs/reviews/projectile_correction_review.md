# Automated review: corrected ideal-projectile trajectory

The original two-step graph has five bindings to four equations. Two bound
expressions lack stored symbolic snapshots. Both were recovered from the exact
Cypher file pinned by the original ingestion snapshot; no source records were
invented. The review checks the full file hash, four exact LaTeX pairs, scalar
definitions and dimensions, rule pin, and fresh evidence for the two stored
expressions. Original artifacts and bindings remain unchanged.

Two source AST defects are explicit: the vertical-position premise references
instantaneous vertical velocity while its LaTeX specifies initial vertical
velocity, and the final trajectory squares gravity while its LaTeX uses linear
gravity. The former changes meaning; the latter also violates dimensional
consistency. The corrected proof uses the source-defined initial-velocity
identity and linear gravity, divides horizontal displacement by nonzero initial
horizontal velocity, and substitutes that time into vertical position. This
matches the ideal-motion derivation in [OpenStax University Physics](https://openstax.org/books/university-physics-volume-1/pages/4-3-projectile-motion).

The numerical provider returns elapsed time and height at a requested horizontal
position. SI coordinates/initial velocities and nonnegative downward gravity
must be finite real arrays of identical nonempty shape. Nonzero horizontal
velocity and nonnegative elapsed time are required; either horizontal direction
is allowed. Zero gravity gives the inertial limit. Exact rational arithmetic on
float64-converted inputs forms both outputs before independent rounding, so
height uses exact time rather than the rounded time output. Exact zeros and
subnormal results are admitted; nonzero underflow to zero and overflow fail.

Twenty-six provider/proof tests passed, including independent 2500-digit Decimal
reference, direction reversal, initial position, large intermediate cancellation,
subnormal output, invalid inputs and rejected squared-gravity proof corruption.
Six serialized CDG runner cases cover 29 synthetic points with zero ULP error
for both outputs. No real dataset records or metadata were used.

The domain is ideal no-drag constant-acceleration motion in one Cartesian frame.
There is no collision/terrain cutoff, wind, varying gravity or accuracy claim
outside that regime. Generic runner diagnostics are not an additional numerical
guarantee at extreme magnitudes; complete output arrays were checked. This is
automated Tier 3 evidence for an explicit corrected realization, not approval of
the original malformed symbolic graph or human-reviewed Tier 1 certification.
