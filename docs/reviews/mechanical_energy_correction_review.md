# Constant-force mechanical energy: reconstructed execution review

Source version: ab2e405e-eb52-54f6-b73b-f4a1450a78f9.
Source graph hash: 49f52e25a02fe4e323beb9d7325542cbfbf4d6f78c6eb973886e31d1c040f624.

The 23-step, 56-binding source derives E2=E1 for a particle in a constant
one-dimensional force field. Four unique missing equations (seven binding
occurrences) are recovered from the hash-pinned public expression file: general
and endpoint kinetic energies and Newton's law. The initial-energy identifier
`pg5579` is malformed in one AST; the reviewed LaTex identifies E1. At step20
the stored right side has already simplified vF-Fv to zero. The original graph
remains draft; this review covers an explicitly reconstructed realization.

The endpoint kinetic energies use instantaneous velocities. The velocity in
finite-interval power equations denotes interval-average velocity. Its equality
to the endpoint mean requires constant acceleration, established here by
positive constant mass and constant net force. Quotient expressions require
positive duration; polynomial trajectories establish zero duration separately.
The force is signed, and reversal is allowed. Displacement is not path length.

The potential is U=-F*x with its zero at x=0. The model assumes a conservative,
time-independent field with no other work or dissipation. An additive constant
would shift both endpoint energies without affecting conservation, but this
implementation chooses zero constant. No variable-force, relativistic or
time-dependent-potential result is claimed. These conditions agree with
[OpenStax's conservation law](https://openstax.org/books/university-physics-volume-1/pages/8-3-conservation-of-energy)
and its [constant-acceleration equations](https://openstax.org/books/university-physics-volume-1/pages/3-4-motion-with-constant-acceleration).

The symbolic proof checks all 23 reconstructed equations plus 12 integral and
derivative conditions, including Newton's law, work-energy, potential gradient,
energy derivative, and zero-duration and zero-force limits. Tests distinguish
average from final velocity, show endpoint averaging fails for variable
acceleration, cover reversal, and exhibit energy change with a time-dependent
potential despite an unchanged force.

The numerical provider accepts equal-shaped, nonempty real arrays (including
scalars): mass, force, initial_position, initial_velocity, elapsed_time. It
returns final_position, final_velocity, kinetic_energy, potential_energy,
total_energy, work. SI units and all eleven ports are explicit in the graph.
Exact rational arithmetic after float64 conversion prevents intermediate
overflow and cancellation from rounded endpoints. Every output is independently
rounded: rounded K+U need not equal rounded total_energy, and energies are based
on exact model coordinates, not the rounded position and velocity. Zero and
subnormal outputs are accepted; nonzero underflow or overflow rejects the call.

Validation uses synthetic inputs only. The independent Decimal reference uses
2500-digit impulse and endpoint-average displacement, initial energy plus work,
and initial conserved total energy. Numeric tests include extreme cancellation,
subnormals, zero duration, zero force, reversal and invalid domains. Retained
test and serialized-runner reports bind the exact provider, tests, proof and
execution code. Approval is automated Tier 3; no Tier 1 or literal source AST
parity claim is made. Catalog promotion and its independent verification are
recorded separately.
