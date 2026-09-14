# Radial infall speed: corrected source and runtime review

Source version e70acd5a-17a6-5e30-b307-78d427d5a207, content hash
79f180692b324066977be6ec88a4e526321e8526c3b437bbde5b31985ab643d6.
Original source remains draft. Automated source/proof and runtime validation
support a corrected realization; catalog promotion is pending, with Tier 3
intended. No human-reviewed Tier 1 or literal source AST parity claim.

Fresh pinned-source audit checks 17 steps, 43 active bindings and 26 stored
expressions. No source expressions require recovery. Important corrections:

- In an outward radial coordinate the force is -G*m*M/x². The source force
  LaTex is outward-positive and its AST also loses vector structure and uses
  x^-1. The intended inward-force integral appears in the next expression.
- The vector integral AST conflates its dummy coordinate and final radius.
  Use the scalar radial force integrated from a finite R to r, then R→∞.
- Restore the missing boundary/antiderivative terms and the mass symbol
  incorrectly parsed as a function in intermediate expressions.
- Restore the zero RHS and infinity evaluation lost from boundary ASTs
  3214170322 and 2924222857. Rest at infinity is a zero-energy asymptotic
  condition, not a launch event at finite time.
- Both work aliases mean gravitational work on the particle. They are not
  interchangeable opposing thermodynamic work conventions.
- The square-root outputs are alternative signed branches. Speed is positive;
  inward radial velocity is negative. Renaming alone does not select a branch.

Assume positive G, test mass m, fixed central mass M and radius r, an exterior
spherical/point-source field, negligible test-mass backreaction, inward radial
motion and no other forces. The caller establishes physical applicability and
clearance. Gravitational work is GMm/r, potential is -GMm/r with zero at infinity,
speed is sqrt(2GM/r), and inward radial velocity is its negative. The formula
also gives ideal escape-speed magnitude by time reversal; it does not certify
a collision-free escape trajectory. Finite-mass relative speed would instead
depend on M+m, so that regime is explicitly excluded. No relativity, drag,
general extended-body field or finite fall-time result is claimed.

Thirteen proof tests cover the improper integral, force/potential sign,
work-energy identity, branch signs, radial acceleration, asymptotic boundary,
test-mass cancellation and mutations. Negative examples distinguish an outward
force, finite release radius and finite two-body mass.

Provider infall_speed takes G, test mass, central mass and radius arrays and
returns speed, inward_radial_velocity, gravitational_work and potential_energy.
Inputs are copied to float64, must be positive finite and nonempty with identical
shapes; scalar arrays are supported, broadcasting is rejected. Local 450-digit
arithmetic precedes independent float64 rounding, accepting subnormals but
rejecting nonzero underflow or overflow. No universal correct-rounding proof
or throughput claim is made. Existing provider search found no infall/escape
implementation covering these outputs.

Forty-four combined proof/runtime tests pass. The independent Decimal reference
uses 2,500-digit arithmetic, splits work into infinity→2r and 2r→r, then derives
speed from kinetic work. Six full serialized runner cases cover 29 synthetic states with all four
outputs matching at zero ULP; graph roundtrip and fresh source audit pass.
Evidence is retained in infall_speed_execution.json. The serialized graph has one
infall node, four inputs and four outputs with SI dimensional signatures.

Reference: [OpenStax University Physics 1, gravitational potential and total energy](https://openstax.org/books/university-physics-volume-1/pages/13-3-gravitational-potential-energy-and-total-energy).
This supports the zero-at-infinity potential and ideal escape-energy model;
the retained source reconstruction and tests supply the automated evidence.
