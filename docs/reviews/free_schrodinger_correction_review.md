# Free Schrödinger equation: source and consistency proof

Source version: 3e30bcec-9752-5d2f-90f9-c5e806d421d9.
Source graph hash: 72c2732025dd2f9b0873efb81fb5c5ee6f9fa8e2d898c64a2558693e64746619.

The 29-step graph has 70 bindings and 37 unique stored equations. The fresh
read-only audit verifies the draft state, graph hash, source fields, rule/feed
inventory and public file pins. No missing expression records were found.
This review does not approve the original graph or claim literal AST parity.

The source's k/(2*pi)=lambda is incorrect: the right side is 1/lambda.
The Laplacian coefficient -p²/hbar must be -p²/hbar². Exponential functions,
the imaginary unit, vector dot products, and gradient/divergence operators are
misparsed in several ASTs. Some whole equations occur in the left field with
an empty right field. Energy and Hamiltonian aliases need consistent meanings.
Step20 lacks the gradient-output binding, which the reconstructed differentiation
supplies explicitly. An operator nabla² cannot be treated as multiplication by
an ordinary scalar symbol.

The reviewed mode is A*exp(i*(p dot r-E*t)/hbar), with E=(p dot p)/(2*m), in
three-dimensional Cartesian coordinates. Mass and hbar are positive constants;
momentum is a constant real vector; A is a constant finite complex amplitude.
The potential is chosen identically zero. Scalar de Broglie wavelength formulas
require nonzero momentum magnitude, but the vector mode and its derivative
identities also cover zero momentum without dividing by wavelength.

This is consistency of a free nonrelativistic plane wave with the Schrödinger
equation. It does not derive quantum dynamics from classical laws, prove that
every wavefunction solves the equation, or supply a potential, boundary,
relativistic or spin model. Operator linearity extends valid modes to finite
linear superpositions. Infinite sums or Fourier integrals require separate
convergence and differentiation arguments. Nonzero plane waves have constant
density and are not square-integrable on infinite space; no normalized-state
claim is made. The scope agrees with
[OpenStax's free-particle equation and plane-wave solution](https://openstax.org/books/university-physics-volume-3/pages/7-3-the-schrodinger-equation).

Sixteen symbolic checks validate all three spatial derivatives, time derivative,
Laplacian, kinetic Hamiltonian, energy generator, free equation, zero momentum,
constant modulus, wavelength/frequency conversions, operator linearity,
two-mode superposition and probability continuity. These are reconstructed
consistency checks, not 29 literal source-step replays. Seven synthetic tests
also reject the missing hbar power, wrong energy dispersion and temporal sign,
show that arbitrary functions need not solve the PDE, and distinguish constant
density from infinite-space normalization. All seven tests pass.

The symbolic runtime accepts four guarded srepr strings for mass, hbar,
a three-component momentum Tuple and amplitude. It rejects coordinate-dependent
parameters and conflicting symbol assumptions. Fixed real coordinates x,y,z,t
are reserved. Parsed floating constants become exact binary-value rationals
before arithmetic; general symbolic parameters with provable domains are retained.
It returns wavefunction and energy srepr, three gradient components in JSON,
Laplacian and time-derivative srepr, and a certificate containing input/coordinate
ASTs, Hamiltonian action and seven verified differential identities. Ten ports
carry explicit dimensions; amplitude uses L^(-3/2) as a dimensional convention,
which does not normalize a nonzero infinite-space plane wave. The certificate
is metadata, not a physical scalar.

All 31 combined tests pass: 24 runtime/contract cases and seven proof/scope cases.
The independent reference builds the wave as separable Cartesian space and time
factors and differentiates that product. A composite-parameter test initially
failed because heuristic simplification did not combine equivalent phases.
The final comparison checks a quotient of nonzero analytic expressions with
normalized rational exponents; the cleared identity extends to coefficient zeros.
The test retains the original general symbolic case and all derivative checks.

Six full serialized-runner cases pass, covering exact three-dimensional momentum,
complex amplitude, zero momentum, zero amplitude, rational scales, independent
symbolic parameters and composite symbolic expressions. Parameter preservation,
all output formulas, certificate identities, graph roundtrip and cached AST
strings are checked. Retained evidence hashes bind the implementation, source
review, proof, tests, parser, codec and runner. All inputs are synthetic.
Automated Tier 3 promotion and independent catalog verification are recorded
separately; no Tier 1 designation or literal source AST parity is claimed.
