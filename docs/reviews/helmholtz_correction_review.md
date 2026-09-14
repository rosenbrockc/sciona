# Time-harmonic wave to Helmholtz reconstruction

Source 24adfd57-ff5a-57e9-9c01-cc9416029fe1, version
ba802b17-88be-59f5-a399-a53a3c385c9c remains draft. Fresh read-only audit verifies
seven steps, seventeen active bindings, ten stored equations and all three
public source file pins, exact symbolic fields and once-decoded source LaTeX.

Source nabla² and partial derivatives are encoded as scalar factors. Reconstruct
a Cartesian Laplacian and actual time derivatives componentwise. Interpret the
imaginary-unit identity as i and the exponential function identity as exp. The
guess step is a monochromatic ansatz, not a derivation from the PDE. The two
steps named differentiate evaluate time derivatives already on the right-hand
side; differentiating both sides of the full equation would be a different step.

For a time-independent C2 spatial component U, real constant omega and positive
constant c, set E=U exp(i omega t), with mu epsilon=1/c². The wave residual equals
exp(i omega t) times [Laplacian(U)+(omega/c)² U]. This proves equivalence within
the harmonic ansatz; it does not assert that arbitrary U has zero residual.
Cancel the nonzero exponential, never U or omega. Zero frequency is included
and reduces to Laplace equation. Complex representation allows taking the real
part. Electric-field components share electric-field units; phase is dimensionless;
Helmholtz terms have field/length² units. Maxwell divergence is a separate condition.

Seven reconstructed steps and residual equivalence pass twelve synthetic tests:
all corrupt steps, plane-wave solution, non-solution, static harmonic amplitude,
longitudinal-wave divergence counterexample and time-dependent-amplitude failure.
No literal source AST parity, full Maxwell, boundary-value solver or broadband claim.

Runtime implementation, serialized runner and Tier 3 promotion are pending.
Use a full three-component Cartesian amplitude realization, preserving residuals
for non-solutions rather than returning a constant zero. Existing vacuum_wave
constructs a general time-domain equation and does not replace harmonic reduction.

Runtime completion: full three-component provider now returns harmonic field,
spatial Laplacian and Helmholtz residual. 36 combined tests and six serialized
runner cases passed with independent derivatives and residual equivalence.
Positive, zero and negative frequency, complex/composed fields, non-solutions and
longitudinal divergence counterexample covered. Guarded parsing, time independence,
constant coefficients and coordinate assumptions verified. All inputs synthetic.
Evidence supports automated Tier 3 within the stated scope; transaction outcome
is recorded separately in the catalog verification report.

Reference: MIT 6.013 Electromagnetics and Applications, course notes, time-harmonic
Helmholtz wave equation and k=omega/c:
https://ocw.mit.edu/courses/6-013-electromagnetics-and-applications-spring-2009/d3be4ea78b036a6362230fb41780cf54_MIT6_013S09_notes.pdf
