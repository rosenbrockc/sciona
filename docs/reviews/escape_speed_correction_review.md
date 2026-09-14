# Escape-speed source and proof review

Status: source/proof and reused-provider runtime validated; Tier 3 promotion pending.
Source version 8d31398d-2174-589d-9624-c72abbf1a3d5, content hash
4acf6f6151a4f16c0aeecf3f9a32bb09d6661b36d45efebcac9cc67e91990960.
Original remains draft; no literal source AST parity or human-reviewed Tier 1.

Fresh audit checks 18 steps, 45 active bindings and 28 public expressions against
pinned symbol, rule and expression files. Four exact-version stored expressions
are missing: 1590774089, 5404822208, 6935745841 and 8357234146. They are recovered
from the pinned expression file; other versions elsewhere are not substituted.

Positive force in the work integral means quasistatic outward external force
against gravity. Its work from radius r to infinity is GMm/r; gravitational work
on an outward-moving particle is the negative. Potential at r is -GMm/r when
zero is chosen at infinity. These work conventions must stay distinct.

Step 3 needs antiderivative -1/x, not +1/x; step 4 must retain -1/x rather than
changing to 1/x². Intermediate integral/boundary ASTs lose important structure,
including a blank RHS. Interpret the named infty symbol as mathematical infinity.
The E2 definition and subsequent equations use different final-kinetic-energy
symbols, which are unified explicitly. Step 14 divides by positive test mass;
it is not an unrestricted simplification. The square-root branches are
alternative signed velocities; positive speed is selected. In the last source
equation generic m denotes central mass, not the cancelled test-particle mass.

Assume positive G, central mass M, test mass m and radius r in a fixed Newtonian
exterior spherical/point-source field with negligible backreaction and no
dissipation. At threshold, kinetic energy balances negative potential and the
terminal speed approaches zero at infinity. A fraction a of threshold speed
has energy GMm/r*(a²-1), negative for 0<a<1, so cannot reach infinity with
nonnegative kinetic energy. Radius is center-based; caller establishes an
outward path without collision. No finite-time arrival, atmospheric loss,
rotating-surface correction, general two-body or relativistic claim.

Fourteen synthetic proof tests pass, including improper integration, work sign,
energy threshold, mass cancellation, branch distinctions and central/test-mass
identity. The reconstructed speed and energy expressions match the approved
infall model. That provider remains approved/latest/served Tier 3 at version
23b6922d-d058-5dbc-aee2-2276694c7cdd and content hash
485f1797fab10da7d52f2ef79af046281f01db9abadff2070b520363cb5bcd04;
its local file matches retained evidence.

Reuse requires an explicit escape graph contract and runner validation. Positive
infall speed equals escape-speed magnitude, and positive infall gravitational
work equals the required launch energy. The existing negative inward velocity
must never be described as outward escape velocity. Preserve this distinction
when selecting or adapting outputs, and avoid changing the approved provider.

Reference: [OpenStax gravitational potential and total energy](https://openstax.org/books/university-physics-volume-1/pages/13-3-gravitational-potential-energy-and-total-energy).
The reference supports the ideal zero-at-infinity escape-energy model; retained
proofs supply the corrected source evidence.

Runtime reuse is now validated. The new escape node binds unchanged provider
sciona.atoms.physics.infall_speed.infall_speed with its pinned approved version.
Input names and ordering remain G, test mass, central mass, radius. Output
aliases by ordinal are escape_speed, inward_counterpart_velocity,
required_launch_energy and potential_energy. Descriptions explicitly prohibit
interpreting the negative counterpart as outward escape velocity and distinguish
positive required launch energy from negative gravitational work during escape.

Forty-eight combined tests pass (escape contract/proof and existing provider
runtime tests). Six serialized runner cases cover 29 synthetic states; all four
saved outputs agree with a 2,500-digit Decimal zero-total-energy reference at
zero ULP. Source validation, graph roundtrip, exact provider serving state and
local provider hash also pass. No existing provider code or catalog audit was
changed. Input domain, underflow/overflow handling and fixed-central-field
limitations remain those of the approved provider.

Next: promote only the new escape CDG, with exact existing-provider binding and
original-source provenance. Its renamed ports and escape-specific evidence
belong to the new graph, not the existing atom. Verify existing provider state
before and after, graph serving and roundtrip, then repeat-apply idempotency.
