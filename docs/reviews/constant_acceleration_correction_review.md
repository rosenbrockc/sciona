# Constant-acceleration source and proof review

Status: source/proof and runtime validated; Tier 3 promotion pending.
Source version 41c11625-ad96-5738-82bc-aa6fb7f95a10, content hash
2c8a5c5ef6fd2b7bc5c989dcaabb25e199c19e1895acbe29c4ced1d09df1d27b.
Original source remains draft; no literal source AST parity or Tier 1 claim.

Fresh read-only audit checks 23 steps, 51 active bindings and 24 expressions
against pinned public files. Twenty-three expressions are stored at the bound
versions; missing average-velocity expression 3411994811 is recovered from the
pinned source. Expression 5144263777 has a missing RHS and loses the square on
velocity in its AST; reconstruct it from the reviewed LaTex. Several other
ASTs have already combined terms displayed separately in LaTex.

The equations describe signed one-dimensional motion with constant acceleration.
The displacement d is not path length if velocity changes sign. The endpoint
mean velocity equals time-average velocity because acceleration is constant;
the average-acceleration quotient alone is insufficient to establish that.
The original time quotients require t nonzero, and steps 14–16 divide by
acceleration and require a nonzero. Direct integration separately proves
v=v0+a*t and d=v0*t+a*t²/2 for a=0 and t=0. Use finite signed v0 and a and
nonnegative elapsed time for the runtime.

Squaring the velocity equation establishes v²=v0²+2*a*d as a consequence, but
does not make it a unique inverse for signed velocity. The runtime must not
discard negative velocities or infer their sign from a square root.

Twenty-seven synthetic proof tests pass: all 23 reconstructed equation
mutations, integral/derivative and endpoint checks, and counterexamples for
variable acceleration, displacement versus path length, and squared-velocity
sign ambiguity. The symbolic residuals are evaluated on the constant-
acceleration solution with quotient domains explicitly retained.

Reference checked: [OpenStax motion with constant acceleration](https://openstax.org/books/university-physics-volume-1/pages/3-4-motion-with-constant-acceleration).
It supports the constant-acceleration assumption and signed kinematic relations;
the retained tests supply the reconstruction evidence.

A broader provider search found endpoint work_energy, which requires final
velocity and does not evolve motion from acceleration and time. Added a complete
constant_acceleration provider: initial_velocity, acceleration, elapsed_time
arrays return final_velocity, displacement and average_velocity. The serialized
motion node has six ordered SI ports. Inputs must be finite real, equal-shaped
and nonempty, with nonnegative time; scalars are supported without broadcasting.

Exact Fraction arithmetic on copied float64 inputs precedes independent output
rounding. Exact zeros and subnormals are accepted; nonzero output underflow and
overflow reject the call. At t=0, displacement is zero and mean velocity is the
continuous extension u, not a computed zero-duration quotient. Signed reversal
and a=0 are supported without division by acceleration or square-root inference.

Forty-nine combined tests pass. An independent 2,500-digit Decimal reference
constructs final velocity, then integrates using the endpoint mean. Six full
serialized runner cases cover 29 synthetic states, including reversal, zero
acceleration, zero time, overflow-prone cancellation and subnormals. All three
saved outputs match at zero ULP. Graph roundtrip and fresh pinned source audit
also pass. No universal throughput claim is made.

Next: promotion dry-run/apply, independent catalog verification, and repeat-
apply check; retain all domain and zero-duration semantics in Tier 3 bounds.
