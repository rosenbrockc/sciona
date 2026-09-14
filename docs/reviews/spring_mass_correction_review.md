# Spring–mass source correction review

Source artifact `ba65b6de-5304-5289-bba5-decd3220c565`, version
`cba4334a-0e0e-5a0e-855f-7aa7c41a5514`, remains unchanged and unapproved.
The audit checks 11 steps, 28 active bindings, 14 stored equations and the missing
Newton-law equation `5345738321` recovered from the pinned public expression file.
All five reviewed LaTeX/AST fields, rule IDs, binding identities and feed ASTs
are checked against retained expectations. Fresh source evidence and all three
public file hashes are retained in `spring_mass_source_review.json`.

The source endpoint writes `x(t) = A cos((k/m)t)`. Substituting its preceding
positive frequency branch requires `x(t) = A cos(sqrt(k/m)t)` instead.
Stiffness has dimensions mass/time², so only the corrected argument is
dimensionless. For the synthetic example k=8, m=2 and A=3, the uncorrected
endpoint has Newton/Hooke residual -72 at t=0; the corrected residual is zero.
The frequency agrees with [OpenStax University Physics, section 15.1](https://openstax.org/books/university-physics-volume-1/pages/15-1-simple-harmonic-motion).

Additional reconstruction is necessary: the source inconsistently aliases
position, time, stiffness and amplitude; several derivative ASTs are scalar
products or derivatives of a constant Symbol. Net force and spring force have
different source IDs. Identifying them requires the explicit physical assumption
that the spring is the net restoring force, in a coordinate measured from
equilibrium. These are semantic corrections, not literal AST parity.

The cosine expression is an ansatz for release from rest, not a general solution
with arbitrary initial velocity. Differentiation supplies velocity and
acceleration. The source step 9 feed encodes time instead of its displayed
multiplier -1/(A cos(omega t)). That multiplier is singular at every displacement
zero and for A=0. The corrected derivation evaluates the identity at t=0 and
divides by nonzero A there. The endpoint is independently checked for zero A and
all times, including zero crossings. Material parameters define natural
frequency even when the motion is identically zero; zero motion alone does not
identify frequency.

The two square-root equations are alternative branches. They cannot both hold
for the same nonzero frequency. Both yield identical cosine motion; positive
frequency is the conventional reported natural frequency. The proof retains
both alternatives and checks their equivalent motion.

The reconstruction includes all eleven transitions. Independent checks cover
the differential equation, displacement and velocity at t=0, conserved energy,
the negative-frequency branch, zero amplitude, and the initial-time frequency
constraint. Fifteen tests pass, including corruption of every transition and
regressions for the missing square root and displacement zero crossings.

Runtime and promotion are pending. Search of the core and physics provider
source trees found no spring-mass/harmonic-oscillator implementation. The next
realization must cover displacement, velocity and acceleration as well as natural
frequency for positive constant mass/stiffness, real time and signed initial
displacement. Use independent high-precision evaluation and explicit numerical
representability limits, including large phase arguments; a frequency-only
helper would not realize this source graph. Follow with serialized runner and
catalog round-trip evidence before any Tier 3 approval. No damping, forcing,
nonlinear spring, or arbitrary-initial-velocity claim is made.


## Runtime completion (supersedes runtime-pending notes)

The provider now implements all four outputs with local 1200-digit mpmath
arithmetic. Ratios, phase, trigonometric functions, and each dimensional output
are evaluated before float64 rounding. Largest possible finite-input phase has
fewer than 625 decimal digits; high precision avoids direct float64 phase
overflow. Reference tests use an independent 1800-digit complex exponential and
separate square roots. Inputs are treated as exact after conversion to float64;
this cannot recover precision absent from physical measurements.

All 38 proof/runtime tests pass. They include signed motion, preserved inputs,
scalar and multidimensional arrays, large phase, overflow-balanced products,
zero amplitude, release-time zeros, subnormal output, energy conservation,
time reversal, invalid domains, and rejected nonzero underflow/overflow.
Seven complete serialized CDG runner cases cover 30 synthetic states with exact
float64 agreement (0 ULP) against the reference. All four saved outputs and the
execution graph round trip are checked. Evidence is retained in
spring_mass_execution.json and spring_mass_test_review.json.

Promotion remains pending; no catalog status was changed by this validation.
